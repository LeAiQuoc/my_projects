"""Cover letter generation template."""

from __future__ import annotations

import os
import re
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from src.facts.facts_schema import FactsEntry
from src.facts.facts_schema import FactsDatabase
from src.generation.cv_generator import CVGenerator
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


_SOFT_SKILL_HINTS: tuple[str, ...] = (
    "customer service",
    "teamwork",
    "technical support",
    "independently",
    "under pressure",
    "accuracy",
    "service",
)

_TECH_DIFFERENTIATOR_HINTS: tuple[str, ...] = (
    "RAG",
    "AI API integration",
    "MCP",
    "Prompt Engineering",
    "embeddings",
)

_SOFT_SKILL_LANGUAGE_HINTS: tuple[str, ...] = (
    "under pressure",
    "independent",
    "independently",
    "accuracy",
    "team",
    "teamwork",
    "collaboration",
    "stakeholder",
    "customer",
    "service",
    "communication",
)


class CoverLetterGenerator:
    """Generate a cover letter grounded strictly in verified facts."""

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self.client = client
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    async def generate(
        self,
        facts: FactsDatabase,
        job_ad: JobAd,
        style_profile: StyleProfile,
        correction_note: str | None = None,
    ) -> str:
        """Generate a grounded cover letter draft from verified inputs.

        The correction note is appended on retries when the evaluator finds issues.
        """

        selected_facts = CVGenerator().select_relevant_facts(facts, job_ad, limit=6)
        selected_facts = _ensure_soft_skill_coverage(selected_facts, facts)
        if not selected_facts:
            raise ValueError("cannot generate cover letter without at least one facts entry")

        system_prompt = (
            "You are an expert cover-letter writer. "
            "Use only the provided facts. "
            "Do not invent achievements, metrics, companies, or experience details. "
            "You may summarize patterns across multiple facts when the meaning stays faithful to those facts."
        )
        user_prompt = self._build_prompt(
            selected_facts=selected_facts,
            job_ad=job_ad,
            style_profile=style_profile,
            correction_note=correction_note,
        )

        client = self.client or _create_default_client()
        temperature = _env_float("DEEPSEEK_COVER_TEMPERATURE", 0.82)
        frequency_penalty = _env_float("DEEPSEEK_COVER_FREQUENCY_PENALTY", 0.15)
        presence_penalty = _env_float("DEEPSEEK_COVER_PRESENCE_PENALTY", 0.35)

        try:
            response = await client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OpenAIError as exc:
            raise RuntimeError(f"cover letter generation request failed: {exc}") from exc

        content = _extract_response_text(response)
        if not content:
            raise RuntimeError("cover letter generation returned empty content")
        content = _ensure_differentiator_mention(content, selected_facts)
        content = _ensure_soft_skill_grounding(content, selected_facts)
        return _remove_low_value_license_line(content, job_ad)

    def _build_prompt(
        self,
        selected_facts: list[FactsEntry],
        job_ad: JobAd,
        style_profile: StyleProfile,
        correction_note: str | None,
    ) -> str:
        """Build a grounded prompt for writing a targeted cover letter."""

        anchor_snippets = "\n\n".join(
            f"- {snippet}" for snippet in style_profile.anchor_snippets[:4]
        ) or "- No anchor snippets provided"
        correction_section = (
            f"\nRetry correction note:\n{correction_note.strip()}\n"
            if correction_note and correction_note.strip()
            else ""
        )
        facts_block = "\n".join(_format_fact(entry) for entry in selected_facts)
        soft_skill_fact_ids = [entry.id for entry in selected_facts if _is_soft_skill_entry(entry)]
        soft_skill_ids_text = ", ".join(soft_skill_fact_ids) if soft_skill_fact_ids else "N/A"
        differentiator_terms = _collect_differentiator_terms(selected_facts)
        differentiator_text = ", ".join(differentiator_terms) if differentiator_terms else "N/A"
        focus_points = _extract_job_ad_focus_points(job_ad)
        focus_points_text = ", ".join(focus_points) if focus_points else "N/A"

        return (
            "Task: Write a tailored cover letter in Markdown for the target role.\n\n"
            "Hard constraints:\n"
            "1) Use ONLY the selected facts entries.\n"
            "2) Do NOT invent skills, outcomes, dates, companies, or metrics.\n"
            "3) If required information is missing, avoid the claim entirely.\n"
            "4) Keep it concise, specific, and role-matched.\n"
            "5) Do NOT include disclaimers, meta commentary, or notes about missing requirements.\n"
            "6) Avoid opinion/intent language (for example: I believe, I look forward, eager to) unless explicit in facts.\n"
            "7) Do NOT convert company requirements into personal skill/readiness claims unless explicit in selected facts.\n"
            "8) Output only the cover letter content in Markdown, with no preface text.\n"
            "9) Prefer action-first factual sentences (who did what) over abstract claims.\n"
            "10) Include at least one short soft-skill paragraph grounded in selected facts (communication, service, teamwork, or independent work under pressure).\n"
            "11) Avoid binary contrast templates like 'not X but Y' and dramatic rhetorical framing.\n"
            "12) Avoid canned scaffolding phrases such as 'I am writing to express my interest', 'At the end of the day', and 'It is important to note that'.\n"
            "13) Keep technical mentions focused: avoid long tool lists; prioritize up to 4 role-relevant technologies.\n\n"
            "14) Vary sentence length on purpose: follow a long sentence with a short punchy one, and avoid making every sentence the same length.\n"
            "15) Prefer a mix of short sentences (about 6-10 words) and medium sentences (about 14-22 words).\n\n"
            "16) If selected facts include differentiator tooling (for example RAG, AI API integration, MCP), include at least one in a concise factual sentence.\n\n"
            "17) Do not describe yourself as 'an AI'; use the plain student wording from the facts instead.\n\n"
            "18) Mention the company name exactly at least once in a factual sentence (for example in the opening or closing paragraph).\n\n"
            "19) Address at least two concrete job-ad focus points from the provided list below.\n\n"
            "Recommended structure:\n"
            "- Greeting line must be exactly: Dear Hiring Team,\n"
            "- Exactly 4 paragraphs plus greeting line.\n"
            "- Keep one blank line between paragraphs.\n"
            "- Paragraph 1: start with one short sentence, then add one longer sentence about fit grounded in facts.\n"
            "- Paragraph 2: use one concrete technical example, followed by a short clarifying sentence about the outcome.\n"
            "- Paragraph 3: use one concise collaboration/service example from a non-technical work fact, then one supporting sentence.\n"
            "- Paragraph 4: end with 1-2 concise closing lines, and make at least one of them short.\n"
            "- Do not output the letter as a single block paragraph.\n\n"
            f"Role target:\n"
            f"- Company: {job_ad.company_name}\n"
            f"- Role: {job_ad.role_title}\n"
            f"- Required skills: {', '.join(job_ad.required_skills) or 'N/A'}\n"
            f"- Nice-to-have skills: {', '.join(job_ad.nice_to_have_skills) or 'N/A'}\n"
            f"- Tone signals: {job_ad.tone_signals}\n"
            f"- Responsibilities: {', '.join(job_ad.key_responsibilities) or 'N/A'}\n\n"
            f"- Focus points to address explicitly: {focus_points_text}\n\n"
            f"Style profile:\n"
            f"- Tone: {style_profile.tone_description}\n"
            f"- Avg sentence length: {style_profile.avg_sentence_length}\n"
            f"- Sentence variance: {style_profile.sentence_length_variance}\n"
            f"- Characteristic phrases: {', '.join(style_profile.characteristic_phrases) or 'N/A'}\n"
            f"- Phrases to avoid: {', '.join(style_profile.phrases_to_avoid) or 'N/A'}\n"
            f"- Structural notes: {style_profile.structural_notes}\n"
            f"- Anchor snippets:\n{anchor_snippets}\n\n"
            f"Soft-skill facts to use explicitly (cite by content, not id): {soft_skill_ids_text}\n"
            "You MUST include at least one sentence that names one employer from those soft-skill facts.\n\n"
            f"Technical differentiators available in selected facts: {differentiator_text}\n"
            "If this list is not N/A, mention at least one item explicitly.\n\n"
            f"Selected facts entries:\n{facts_block}\n"
            f"{correction_section}"
        )


def _create_default_client() -> AsyncOpenAI:
    """Create a Deepseek-compatible async client from environment settings."""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for generation")
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


def _format_fact(entry: FactsEntry) -> str:
    """Serialize one fact entry into compact prompt-friendly text."""

    technologies = ", ".join(entry.technologies) if entry.technologies else "N/A"
    start_date = str(entry.start_date) if entry.start_date else "N/A"
    end_date = str(entry.end_date) if entry.end_date else "Present"
    evidence = entry.evidence_url or "N/A"
    return (
        f"- id={entry.id}; category={entry.category}; title={entry.title}; "
        f"description={entry.description}; technologies={technologies}; "
        f"start={start_date}; end={end_date}; evidence={evidence}"
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


def _ensure_soft_skill_coverage(
    selected_facts: list[FactsEntry],
    facts: FactsDatabase,
) -> list[FactsEntry]:
    """Ensure cover-letter context includes at least one soft-skill evidence entry."""

    if not selected_facts:
        return selected_facts

    selected_ids = {entry.id for entry in selected_facts}
    if any(_is_soft_skill_entry(entry) for entry in selected_facts):
        return selected_facts

    for candidate in facts.entries:
        if candidate.id in selected_ids:
            continue
        if _is_soft_skill_entry(candidate):
            return [*selected_facts, candidate]

    return selected_facts


def _is_soft_skill_entry(entry: FactsEntry) -> bool:
    """Detect facts likely to support grounded people-skill statements."""

    if entry.category != "experience":
        return False

    haystack = f"{entry.title} {entry.description} {' '.join(entry.technologies)}".lower()
    return any(token in haystack for token in _SOFT_SKILL_HINTS)


def _collect_differentiator_terms(selected_facts: list[FactsEntry]) -> list[str]:
    """Collect concise advanced-tooling terms available in selected facts."""

    found: list[str] = []
    seen: set[str] = set()
    for entry in selected_facts:
        blob = f"{entry.title} {entry.description} {' '.join(entry.technologies)}".lower()
        for term in _TECH_DIFFERENTIATOR_HINTS:
            normalized = term.lower()
            if normalized in blob and normalized not in seen:
                found.append(term)
                seen.add(normalized)
    return found[:3]


def _ensure_differentiator_mention(text: str, selected_facts: list[FactsEntry]) -> str:
    """Inject one concise factual sentence when key differentiators are omitted."""

    draft = text.strip()
    if not draft:
        return text

    differentiators = _collect_differentiator_terms(selected_facts)
    if not differentiators:
        return text

    draft_lower = draft.lower()
    present = [term for term in differentiators if term.lower() in draft_lower]
    if present:
        return text

    sentence = f"Relevant tooling in my experience includes {differentiators[0]}."

    if draft.endswith("\n"):
        return f"{draft}{sentence}\n"
    return f"{draft}\n\n{sentence}"


def _remove_low_value_license_line(text: str, job_ad: JobAd) -> str:
    """Drop driver's-license statements unless the role explicitly requires it."""

    role_context = " ".join(
        [
            job_ad.role_title,
            job_ad.source_text or "",
            " ".join(job_ad.required_skills),
            " ".join(job_ad.nice_to_have_skills),
        ]
    ).lower()

    requires_license = any(
        marker in role_context
        for marker in ("driver", "driving license", "korkort", "körkort", "class b")
    )
    if requires_license:
        return text

    lines = text.splitlines()
    filtered = [
        line
        for line in lines
        if not any(
            marker in line.lower()
            for marker in ("driver's license", "driving license", "class b", "korkort", "körkort")
        )
    ]
    return "\n".join(filtered).strip()


def _ensure_soft_skill_grounding(text: str, selected_facts: list[FactsEntry]) -> str:
    """Ensure at least one explicit soft-skill sentence grounded in selected experience facts."""

    draft = text.strip()
    if not draft:
        return text

    soft_entries = [entry for entry in selected_facts if _is_soft_skill_entry(entry)]
    if not soft_entries:
        return text

    # If the draft already contains soft-skill language, do not append a fixed
    # fallback sentence that can become repetitive across job ads.
    if _has_soft_skill_language(draft):
        return text

    draft_lower = draft.lower()
    for entry in soft_entries:
        company = _extract_company_from_title(entry.title)
        if company and company.lower() in draft_lower:
            return text

    anchor = soft_entries[0]
    company = _extract_company_from_title(anchor.title) or "a prior role"
    summary = _soft_skill_summary(anchor.description)
    sentence = _soft_skill_fallback_sentence(company, summary, draft)
    return f"{draft}\n\n{sentence}"


def _has_soft_skill_language(text: str) -> bool:
    """Detect whether the draft already contains concrete people-skill wording."""

    lower = text.lower()
    return any(token in lower for token in _SOFT_SKILL_LANGUAGE_HINTS)


def _soft_skill_fallback_sentence(company: str, summary: str, draft: str) -> str:
    """Build a varied fallback sentence to avoid repeating one fixed phrasing."""

    templates = (
        "At {company}, I developed {summary}.",
        "My experience at {company} strengthened {summary}.",
        "Working at {company}, I built {summary}.",
    )
    idx = sum(ord(char) for char in draft) % len(templates)
    return templates[idx].format(company=company, summary=summary)


def _extract_company_from_title(title: str) -> str | None:
    """Extract employer name from titles shaped like 'Role at Company ...'."""

    match = re.search(r"\bat\s+([^,]+)", title, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _soft_skill_summary(description: str) -> str:
    """Build a compact soft-skill summary grounded in the fact description."""

    lower = description.lower()
    if "accuracy" in lower and "under pressure" in lower:
        return "accuracy and independent execution under pressure"
    if "customer" in lower or "service" in lower or "teamwork" in lower:
        return "customer-facing communication and teamwork"
    if "technical support" in lower or "maintenance" in lower:
        return "clear communication and practical troubleshooting"
    return "reliable collaboration and day-to-day ownership"


def _extract_job_ad_focus_points(job_ad: JobAd) -> list[str]:
    """Collect concise, actionable focus points from the parsed job ad."""

    collected: list[str] = []
    seen: set[str] = set()

    for item in [*job_ad.required_skills, *job_ad.nice_to_have_skills, *job_ad.key_responsibilities]:
        cleaned = re.sub(r"\s+", " ", item.strip())
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        if len(cleaned) > 64:
            cleaned = cleaned[:64].rstrip() + "..."
        seen.add(lowered)
        collected.append(cleaned)
        if len(collected) >= 8:
            break

    return collected
