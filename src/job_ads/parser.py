"""Async job ad parser boundary for the Deepseek-backed implementation."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .schema import JobAd


_KNOWN_TECH_TERMS = {
    "python", "sql", "postgresql", "mysql", "mongodb", "docker", "git", "gitlab",
    "github", "fastapi", "flask", "django", "streamlit", "react", "typescript",
    "javascript", "java", "c", "c++", "c/c++", "matlab", "simulink", "matlab/simulink",
    "autosar", "vector", "vectors", "airflow", "kubernetes", "scikit-learn", "numpy",
    "pandas", "opencv", "langchain", "reflex", "ffmpeg", "faster-whisper", "deepseek",
    "gemini", "claude", "gpt", "ollama", "mcp", "rag", "embeddings", "ci/cd",
    "home assistant", "autohotkey", "mqtt", "grafana", "raspberry", "broadlink",
}

_NOISE_SKILL_TOKENS = {
    "och/eller",
    "e-post",
    "observera",
    "da",
    "kanner",
    "annonsen",
    "ansokan",
    "ansokningar",
    "kandidaten",
    "om",
    "eller",
}

_CANONICAL_SKILL_FORMS = {
    "python": "Python",
    "c/c++": "C/C++",
    "matlab/simulink": "Matlab/Simulink",
    "autosar": "Autosar",
    "vector": "Vector",
    "vectors": "Vectors",
}

_PHRASE_PATTERNS: list[tuple[str, str]] = [
    (r"\bc\s*/\s*c\s*\+\+\b", "C/C++"),
    (r"\bmatlab\s*/\s*simulink\b", "Matlab/Simulink"),
    (r"\bautosar\b", "Autosar"),
    (r"\bvectors?\b", "Vectors"),
    (r"\bpython\b", "Python"),
]


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

    for pattern in (
        r"company[:\s]+(.+)",
        r"about\s+(.+?)\s+(?:is|are|we)",
        r"om\s+([A-ZA-Za-z0-9&\- ]+?)\s+i\s+[A-ZA-Za-z0-9&\- ]+",
        r"([A-ZA-Za-z0-9&\- ]+?)\s+vaxer\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().splitlines()[0][:120]
    first_line = _meaningful_lines(text)[0] if _meaningful_lines(text) else ""
    return first_line[:120] if first_line else "Unknown Company"


def _extract_role_title(text: str) -> str:
    """Infer the role title from the first line or common title markers."""

    for pattern in (
        r"role[:\s]+(.+)",
        r"position[:\s]+(.+)",
        r"hiring[:\s]+(.+)",
        r"positionen\s+som\s+(.+)",
        r"rollen\s+som\s+(.+)",
        r"som\s+(mjukvaruutvecklare|software engineer|backend engineer|embedded engineer|developer)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().splitlines()[0][:120]
    first_line = _meaningful_lines(text)[0] if _meaningful_lines(text) else ""
    return first_line[:120] if first_line else "Unknown Role"


def _extract_skills(text: str, required: bool) -> list[str]:
    """Pull skills from skill-oriented sections and keyword-like tokens."""

    lower_text = text.lower()
    required_markers = (
        "requirements",
        "must have",
        "required",
        "krav",
        "vi ser garna",
        "programmeringssprak",
        "har anvant",
        "arbetat med",
    )
    nice_to_have_markers = ("nice to have", "preferred", "meriterande", "plus om")

    candidate_lines = _meaningful_lines(text)
    if required:
        scoped_lines = [line for line in candidate_lines if any(marker in line.lower() for marker in required_markers)]
        if scoped_lines:
            return _find_technology_tokens("\n".join(scoped_lines + _neighbor_lines(candidate_lines, scoped_lines)))
        return _find_technology_tokens(text)

    scoped_lines = [line for line in candidate_lines if any(marker in line.lower() for marker in nice_to_have_markers)]
    if not scoped_lines:
        return []
    return _find_technology_tokens("\n".join(scoped_lines + _neighbor_lines(candidate_lines, scoped_lines)))


def _find_technology_tokens(text: str) -> list[str]:
    """Detect concise technology-like tokens from the posting text."""

    candidates = re.findall(r"(?<![A-Za-z0-9+.#/\-])[A-Za-z][A-Za-z0-9+.#/\-]{1,}(?![A-Za-z0-9+.#/\-])", text)
    common_filters = {
        "the", "and", "with", "that", "this", "from", "have", "will", "for", "you",
        "role", "job", "team", "work", "we", "our", "are", "company", "position",
        "requirements", "preferred", "responsibilities", "experience", "skills",
        "om", "jobbet", "rollen", "ansokan", "ansökan", "vi", "du", "att", "och",
        "som", "har", "det", "den", "med", "vara", "vart", "inom", "allt", "dig",
        "goteborg", "göteborg", "nexer", "engineering", "vaxer", "växer", "ars",
        "mjukvaruutvecklare", "framtidens", "kunder", "personal", "kultur", "vision",
    }
    skills: list[str] = []
    seen: set[str] = set()

    # First, capture common multi-token technical phrases so token-level parsing
    # does not degrade expressions like C/C++ into partial fragments.
    for pattern, canonical in _PHRASE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            canonical_lower = canonical.lower()
            if canonical_lower not in seen:
                skills.append(canonical)
                seen.add(canonical_lower)

    for candidate in candidates:
        normalized = _normalize_skill_token(candidate)
        if normalized is None:
            continue
        normalized_lower = normalized.lower()
        if normalized_lower in common_filters:
            continue
        if len(normalized_lower) < 2:
            continue
        if normalized_lower in seen:
            continue
        if _looks_like_technology_token(normalized, normalized_lower):
            skills.append(normalized)
            seen.add(normalized_lower)
    return skills[:12]


def _normalize_skill_token(token: str) -> str | None:
    """Normalize a candidate token and drop obvious non-skill fragments."""

    cleaned = token.strip(".,;:()[]{}!?")
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in _NOISE_SKILL_TOKENS:
        return None
    if "@" in lowered:
        return None

    if lowered in _CANONICAL_SKILL_FORMS:
        return _CANONICAL_SKILL_FORMS[lowered]

    # Keep connectors like "x/y" only when both sides look technical.
    if "/" in lowered:
        parts = [part for part in lowered.split("/") if part]
        if not parts:
            return None
        if any(part in _NOISE_SKILL_TOKENS for part in parts):
            return None

    return cleaned


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

    raw_lines = _meaningful_lines(text)
    bullets = [line.lstrip("-• \t").strip() for line in raw_lines if line.startswith(("-", "•"))]
    section_lines = [line.rstrip(":").strip() for line in raw_lines if line.endswith(":")]
    combined = bullets + section_lines
    if combined:
        return combined[:10]
    return raw_lines[:5]


def _meaningful_lines(text: str) -> list[str]:
    """Return non-empty lines without markdown heading noise."""

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[#*\s]+", "", line).strip()
        line = re.sub(r"[*\s]+$", "", line).strip()
        if not line:
            continue
        if line.lower() in {"om jobbet", "om rollen", "ansokan", "ansökan", "var kultur", "vår kultur"}:
            continue
        lines.append(line)
    return lines


def _neighbor_lines(lines: Sequence[str], scoped_lines: Sequence[str]) -> list[str]:
    """Collect nearby lines after matched section markers for better skill extraction."""

    scoped_set = set(scoped_lines)
    neighbors: list[str] = []
    for index, line in enumerate(lines):
        if line not in scoped_set:
            continue
        for offset in (1, 2):
            next_index = index + offset
            if next_index < len(lines):
                candidate = lines[next_index]
                if _looks_like_skill_line(candidate):
                    neighbors.append(candidate)
    return neighbors


def _looks_like_skill_line(line: str) -> bool:
    """Heuristically detect whether a line is likely to enumerate technologies."""

    if line.startswith(("-", "•")):
        return True
    if any(symbol in line for symbol in ("/", "+", ",")):
        return True
    line_lower = line.lower()
    return any(term in line_lower for term in _KNOWN_TECH_TERMS)


def _looks_like_technology_token(normalized: str, normalized_lower: str) -> bool:
    """Decide whether a token is likely to be a real technology/skill term."""

    if normalized_lower in _NOISE_SKILL_TOKENS:
        return False
    if normalized_lower in _KNOWN_TECH_TERMS:
        return True
    if "/" in normalized_lower:
        return False
    if any(char in normalized for char in ("+", ".", "#", "-", "/")):
        return True
    if normalized.isupper() and len(normalized) <= 8:
        return True
    return False
