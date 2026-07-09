from __future__ import annotations

import json

from openai import AsyncOpenAI

from src.config.settings import settings
from src.models.schemas import HighlightCandidate, TranscriptSegment


class HighlightDetectionError(RuntimeError):
    """Base error for highlight detection issues."""

    pass


class HighlightConfigError(HighlightDetectionError):
    """Raised when required highlight configuration is missing."""

    pass


class HighlightResponseError(HighlightDetectionError):
    """Raised when model output is malformed or unusable."""

    pass


def _build_transcript_prompt(transcript_segments: list[TranscriptSegment]) -> str:
    """Serialize transcript segments into a compact timestamped text block."""

    lines: list[str] = []
    for segment in transcript_segments:
        lines.append(
            f"[{segment.start_seconds:.2f} - {segment.end_seconds:.2f}] {segment.text}",
        )
    return "\n".join(lines)


def _extract_json_payload(raw_content: str) -> str:
    """Strip optional markdown fences and return a JSON candidate string."""

    content = raw_content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        if content.endswith("```"):
            content = content[:-3]
    return content.strip()


async def _request_highlights_json(
    transcript_segments: list[TranscriptSegment],
    top_k: int,
    min_seconds: int,
    max_seconds: int,
) -> str:
    """Request highlight candidates from DeepSeek and return raw text content."""

    if not settings.deepseek_api_key:
        raise HighlightConfigError("DEEPSEEK_API_KEY is not configured")

    # DeepSeek exposes an OpenAI-compatible endpoint, so we use this client transport.
    client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    transcript_payload = _build_transcript_prompt(transcript_segments)

    system_prompt = (
        "You select short highlight segments from a transcript. "
        "Return strict JSON only with no markdown."
    )
    user_prompt = (
        f"From the transcript below, select up to {top_k} self-contained highlights suitable for short clips. "
        f"Each highlight must be between {min_seconds} and {max_seconds} seconds when possible. "
        "Return JSON in this shape: "
        "{\"highlights\": [{\"start_seconds\": number, \"end_seconds\": number, "
        "\"hook_title\": string, \"rationale\": string, \"confidence\": number}]} "
        f"Confidence must be 0 to 1. Transcript:\n{transcript_payload}"
    )

    try:
        response = await client.chat.completions.create(
            model=settings.highlight_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise HighlightDetectionError(f"DeepSeek request failed: {exc}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise HighlightResponseError("DeepSeek returned an empty response")

    return content


def _validate_and_convert_candidates(
    payload: dict,
    transcript_segments: list[TranscriptSegment],
    top_k: int,
) -> list[HighlightCandidate]:
    """Validate parsed JSON payload and convert items into HighlightCandidate objects."""

    highlights = payload.get("highlights")
    if not isinstance(highlights, list):
        raise HighlightResponseError("Response JSON must include a 'highlights' list")

    timeline_start = min(segment.start_seconds for segment in transcript_segments)
    timeline_end = max(segment.end_seconds for segment in transcript_segments)

    candidates: list[HighlightCandidate] = []
    for item in highlights[:top_k]:
        if not isinstance(item, dict):
            raise HighlightResponseError("Each highlight item must be an object")

        try:
            candidate = HighlightCandidate(
                start_seconds=float(item["start_seconds"]),
                end_seconds=float(item["end_seconds"]),
                hook_title=str(item["hook_title"]).strip(),
                rationale=str(item["rationale"]).strip(),
                confidence=float(item["confidence"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HighlightResponseError(f"Invalid highlight fields: {exc}") from exc

        if candidate.end_seconds <= candidate.start_seconds:
            raise HighlightResponseError("Highlight end_seconds must be greater than start_seconds")
        if candidate.start_seconds < timeline_start or candidate.end_seconds > timeline_end + 1.0:
            raise HighlightResponseError("Highlight timestamps are outside transcript timeline")
        if not candidate.hook_title:
            raise HighlightResponseError("Highlight hook_title cannot be empty")
        if not candidate.rationale:
            raise HighlightResponseError("Highlight rationale cannot be empty")

        candidates.append(candidate)

    if not candidates:
        raise HighlightResponseError("No valid highlights returned by model")

    return candidates


async def detect_highlights(
    transcript_segments: list[TranscriptSegment],
    top_k: int,
    min_seconds: int,
    max_seconds: int,
) -> list[HighlightCandidate]:
    """Detect transcript highlights using DeepSeek and return validated clip candidates."""
    if not transcript_segments:
        raise HighlightDetectionError("Transcript is empty")
    if top_k <= 0:
        raise HighlightDetectionError("top_k must be greater than 0")
    if min_seconds <= 0 or max_seconds <= 0:
        raise HighlightDetectionError("Clip duration bounds must be greater than 0")
    if min_seconds > max_seconds:
        raise HighlightDetectionError("min_seconds cannot be greater than max_seconds")

    raw_content = await _request_highlights_json(
        transcript_segments=transcript_segments,
        top_k=top_k,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )

    cleaned = _extract_json_payload(raw_content)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise HighlightResponseError("DeepSeek response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise HighlightResponseError("Top-level response JSON must be an object")

    return _validate_and_convert_candidates(payload, transcript_segments, top_k)
