"""Async job ad parser boundary for the Deepseek-backed implementation."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .schema import JobAd


class JobAdParser:
    """Parse raw job ad text into a structured JobAd model."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    async def parse(self, raw_text: str, source_url: str | None = None) -> JobAd:
        """Parse one posting into a JobAd.

        The method validates input locally before the Deepseek call is wired in.
        """

        cleaned_text = raw_text.strip()
        if not cleaned_text:
            raise ValueError("job ad text cannot be empty")

        company_name = _extract_company_name(cleaned_text)
        role_title = _extract_role_title(cleaned_text)
        required_skills = _extract_skills(cleaned_text, required=True)
        nice_to_have_skills = _extract_skills(cleaned_text, required=False)
        tone_signals = _infer_tone_signals(cleaned_text)
        key_responsibilities = _extract_bullets(cleaned_text)

        return JobAd(
            company_name=company_name,
            role_title=role_title,
            required_skills=required_skills,
            nice_to_have_skills=nice_to_have_skills,
            tone_signals=tone_signals,
            key_responsibilities=key_responsibilities,
            source_text=cleaned_text,
            source_url=source_url,
        )

    async def parse_many(self, items: Sequence[str]) -> list[JobAd]:
        """Parse multiple postings sequentially through the async parser boundary."""

        parsed: list[JobAd] = []
        for item in items:
            parsed.append(await self.parse(item))
        return parsed


def _extract_company_name(text: str) -> str:
    """Infer a company name from common job-ad patterns."""

    for pattern in (r"company[:\s]+(.+)", r"about\s+(.+?)\s+(?:is|are|we)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().splitlines()[0][:120]
    first_line = text.splitlines()[0].strip()
    return first_line[:120] if first_line else "Unknown Company"


def _extract_role_title(text: str) -> str:
    """Infer the role title from the first line or common title markers."""

    for pattern in (r"role[:\s]+(.+)", r"position[:\s]+(.+)", r"hiring[:\s]+(.+)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().splitlines()[0][:120]
    first_line = text.splitlines()[0].strip()
    return first_line[:120] if first_line else "Unknown Role"


def _extract_skills(text: str, required: bool) -> list[str]:
    """Pull skills from skill-oriented sections and keyword-like tokens."""

    lower_text = text.lower()
    section_markers = ["requirements", "must have", "required", "nice to have", "preferred"]
    if required and not any(marker in lower_text for marker in section_markers[:3]):
        return _find_technology_tokens(text)
    if not required and not any(marker in lower_text for marker in section_markers[3:]):
        return []

    skills = _find_technology_tokens(text)
    return skills


def _find_technology_tokens(text: str) -> list[str]:
    """Detect concise technology-like tokens from the posting text."""

    candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9+.#-]{1,}\b", text)
    common_filters = {
        "the", "and", "with", "that", "this", "from", "have", "will", "for", "you",
        "role", "job", "team", "work", "we", "our", "are", "company", "position",
        "requirements", "preferred", "responsibilities", "experience", "skills",
    }
    skills: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized.lower() in common_filters:
            continue
        if len(normalized) < 2:
            continue
        if normalized.lower() in seen:
            continue
        if normalized[0].isupper() or any(char in normalized for char in ("+", ".", "#", "-")):
            skills.append(normalized)
            seen.add(normalized.lower())
    return skills[:12]


def _infer_tone_signals(text: str) -> str:
    """Summarize the overall tone of the posting in a compact label."""

    lower_text = text.lower()
    if any(marker in lower_text for marker in ("startup", "fast-paced", "friendly", "casual")):
        return "startup casual"
    if any(marker in lower_text for marker in ("corporate", "professional", "formal", "global")):
        return "corporate formal"
    return "neutral professional"


def _extract_bullets(text: str) -> list[str]:
    """Collect bullet-like responsibilities from the posting text."""

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = [line.lstrip("-• \t").strip() for line in raw_lines if line.startswith(("-", "•"))]
    section_lines = [line.rstrip(":").strip() for line in raw_lines if line.endswith(":")]
    combined = bullets + section_lines
    if combined:
        return combined[:10]
    return raw_lines[:5]
