"""Optional rewrite pass for reducing templated rhythm."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from src.style.style_profile import StyleProfile


async def rewrite_for_natural_rhythm(
    draft: str,
    style_profile: StyleProfile,
    client: Any | None = None,
    model: str | None = None,
) -> str:
    """Rewrite a draft when AI-tone checks flag rhythm as too uniform.

    The rewrite is constrained to preserve factual claims from the original draft
    while increasing sentence-length variance and matching cadence from
    style-profile anchor snippets.
    """

    if not draft.strip():
        raise ValueError("draft must not be empty")

    resolved_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    active_client = client or _create_default_client()
    temperature = _env_float("DEEPSEEK_HUMANIZE_TEMPERATURE", 0.72)
    frequency_penalty = _env_float("DEEPSEEK_HUMANIZE_FREQUENCY_PENALTY", 0.35)
    presence_penalty = _env_float("DEEPSEEK_HUMANIZE_PRESENCE_PENALTY", 0.15)

    system_prompt = (
        "You are an expert editor. Rewrite text to sound naturally human and less templated. "
        "Keep all factual content grounded in the original draft. "
        "Do not introduce any new facts, technologies, employers, dates, or achievements."
    )
    user_prompt = _build_humanize_prompt(draft=draft, style_profile=style_profile)

    try:
        response = await active_client.chat.completions.create(
            model=resolved_model,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except OpenAIError as exc:
        raise RuntimeError(f"humanization pass failed: {exc}") from exc

    rewritten = _extract_response_text(response)
    if not rewritten:
        raise RuntimeError("humanization pass returned empty content")
    return rewritten


def _build_humanize_prompt(draft: str, style_profile: StyleProfile) -> str:
    """Build a rewrite prompt anchored to profile cadence and constraints."""

    anchor_block = "\n\n".join(
        f"- {snippet}" for snippet in style_profile.anchor_snippets[:4]
    ) or "- No anchor snippets provided"

    return (
        "Rewrite the draft below with these constraints:\n"
        "1) Vary sentence length noticeably (mix short, medium, and longer sentences).\n"
        "2) Remove generic or boilerplate cover-letter sounding lines.\n"
        "3) Preserve all factual claims from the original draft.\n"
        "4) Do not add any new facts or credentials.\n"
        "5) Match sentence-length variance, cadence, and structural rhythm from the anchor snippets.\n"
        "6) Prefer concrete verbs and specific phrasing over abstract motivational language.\n"
        "7) Keep paragraph lengths uneven and natural; avoid repeated sentence templates.\n"
        "8) Keep role/company references factual and concise, avoiding filler transitions.\n\n"
        "Style profile:\n"
        f"- Tone: {style_profile.tone_description}\n"
        f"- Avg sentence length: {style_profile.avg_sentence_length}\n"
        f"- Sentence variance: {style_profile.sentence_length_variance}\n"
        f"- Characteristic phrases: {', '.join(style_profile.characteristic_phrases) or 'N/A'}\n"
        f"- Phrases to avoid: {', '.join(style_profile.phrases_to_avoid) or 'N/A'}\n"
        f"- Structural notes: {style_profile.structural_notes}\n"
        f"- Anchor snippets:\n{anchor_block}\n\n"
        f"Original draft:\n{draft}\n"
    )


def _env_float(name: str, default: float) -> float:
    """Read a float from environment with safe fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _create_default_client() -> AsyncOpenAI:
    """Create Deepseek-compatible async client from environment settings."""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for humanization pass")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _extract_response_text(response: Any) -> str:
    """Extract normalized assistant text from OpenAI-compatible responses."""

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(getattr(part, "text", "")) for part in content]
        return "".join(parts).strip()
    return str(content).strip()
