"""Cover letter generation template."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from src.facts.facts_schema import FactsEntry
from src.facts.facts_schema import FactsDatabase
from src.generation.cv_generator import CVGenerator
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


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
        if not selected_facts:
            raise ValueError("cannot generate cover letter without at least one facts entry")

        system_prompt = (
            "You are an expert cover-letter writer. "
            "Use only the provided facts. "
            "Do not invent achievements, metrics, companies, or experience details."
        )
        user_prompt = self._build_prompt(
            selected_facts=selected_facts,
            job_ad=job_ad,
            style_profile=style_profile,
            correction_note=correction_note,
        )

        client = self.client or _create_default_client()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                temperature=0.45,
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
        return content

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

        return (
            "Task: Write a tailored cover letter in Markdown for the target role.\n\n"
            "Hard constraints:\n"
            "1) Use ONLY the selected facts entries.\n"
            "2) Do NOT invent skills, outcomes, dates, companies, or metrics.\n"
            "3) If required information is missing, avoid the claim entirely.\n"
            "4) Keep it concise, specific, and role-matched.\n\n"
            f"Role target:\n"
            f"- Company: {job_ad.company_name}\n"
            f"- Role: {job_ad.role_title}\n"
            f"- Required skills: {', '.join(job_ad.required_skills) or 'N/A'}\n"
            f"- Nice-to-have skills: {', '.join(job_ad.nice_to_have_skills) or 'N/A'}\n"
            f"- Tone signals: {job_ad.tone_signals}\n"
            f"- Responsibilities: {', '.join(job_ad.key_responsibilities) or 'N/A'}\n\n"
            f"Style profile:\n"
            f"- Tone: {style_profile.tone_description}\n"
            f"- Avg sentence length: {style_profile.avg_sentence_length}\n"
            f"- Sentence variance: {style_profile.sentence_length_variance}\n"
            f"- Characteristic phrases: {', '.join(style_profile.characteristic_phrases) or 'N/A'}\n"
            f"- Phrases to avoid: {', '.join(style_profile.phrases_to_avoid) or 'N/A'}\n"
            f"- Structural notes: {style_profile.structural_notes}\n"
            f"- Anchor snippets:\n{anchor_snippets}\n\n"
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
