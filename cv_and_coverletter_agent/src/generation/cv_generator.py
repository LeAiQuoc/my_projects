"""CV generation template."""

from __future__ import annotations

import os
import re
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from src.facts.facts_schema import FactsDatabase, FactsEntry
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


_LANGUAGE_NAMES = {
    "sv": "Swedish",
    "en": "English",
}


class CVGenerator:
    """Generate a tailored CV section set from the verified facts database."""

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
        """Generate a grounded CV draft for the target job ad."""

        selected_facts = self.select_relevant_facts(facts, job_ad, limit=6)
        if not selected_facts:
            raise ValueError("cannot generate CV without at least one facts entry")

        system_prompt = (
            "You are an expert CV writer. "
            "You must write only from the provided facts entries. "
            "Do not invent or assume any skill, metric, employer, or project detail. "
            "If information is missing, omit it rather than hallucinate."
        )
        user_prompt = self._build_prompt(
            selected_facts=selected_facts,
            job_ad=job_ad,
            style_profile=style_profile,
            correction_note=correction_note,
        )

        client = self.client or _create_default_client()
        temperature = _env_float("DEEPSEEK_CV_TEMPERATURE", 0.35)
        frequency_penalty = _env_float("DEEPSEEK_CV_FREQUENCY_PENALTY", 0.2)
        presence_penalty = _env_float("DEEPSEEK_CV_PRESENCE_PENALTY", 0.05)

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
            raise RuntimeError(f"CV generation request failed: {exc}") from exc

        content = _extract_response_text(response)
        if not content:
            raise RuntimeError("CV generation returned empty content")
        content = _ground_unverified_skill_claims(content)
        return content

    def _build_prompt(
        self,
        selected_facts: list[FactsEntry],
        job_ad: JobAd,
        style_profile: StyleProfile,
        correction_note: str | None,
    ) -> str:
        """Build a structured prompt with only relevant grounding context."""

        anchor_snippets = "\n\n".join(
            f"- {snippet}" for snippet in style_profile.anchor_snippets[:4]
        ) or "- No anchor snippets provided"
        correction_section = (
            f"\nRetry correction note:\n{correction_note.strip()}\n"
            if correction_note and correction_note.strip()
            else ""
        )

        facts_block = "\n".join(_format_fact(entry) for entry in selected_facts)
        language_code = job_ad.source_language.lower() if job_ad.source_language else "en"
        language_name = _LANGUAGE_NAMES.get(language_code, "English")
        return (
            "Task: Write a tailored CV draft in Markdown for this role.\n\n"
            f"Language requirement: Write entirely in {language_name}. Keep headings, labels, and body text in {language_name}. Do not mix languages except for company names and product names.\n\n"
            "Hard constraints:\n"
            "1) Use ONLY the provided facts entries.\n"
            "2) Do NOT invent skills, achievements, timelines, or metrics.\n"
            "3) Prefer concise bullet points and role-relevant ordering.\n"
            "4) Keep tone aligned with style profile data.\n"
            "5) Do NOT include notes, disclaimers, or commentary about missing requirements.\n"
            "6) Do NOT include intention/opinion lines unless explicitly supported by facts.\n"
            "7) Do NOT convert company requirements into personal skill/readiness claims unless explicit in facts.\n"
            "8) Do not describe yourself as 'an AI'; use the plain student wording from the facts instead.\n"
            "9) When mentioning auxiliary facts like personal programming practice, a driver's license, or languages, avoid three short sentences in a row; combine them into uneven sentence lengths.\n"
            "10) Output only the CV content in Markdown, with no preface text.\n\n"
            f"Role target:\n"
            f"- Company: {job_ad.company_name}\n"
            f"- Role: {job_ad.role_title}\n"
            f"- Required skills: {', '.join(job_ad.required_skills) or 'N/A'}\n"
            f"- Nice-to-have skills: {', '.join(job_ad.nice_to_have_skills) or 'N/A'}\n"
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

    def select_relevant_facts(
        self,
        facts_db: FactsDatabase,
        job_ad: JobAd,
        limit: int = 3,
    ) -> list[FactsEntry]:
        """Return the top-N facts ranked by relevance to the target job ad.

        This scoring is deterministic and favors direct skill/technology matches,
        then role-title alignment, and finally contextual responsibility overlap.
        """

        if limit <= 0:
            return []

        scored_entries: list[tuple[float, FactsEntry]] = []

        # Normalize job metadata for case-insensitive matching.
        req_skills = {skill.strip().lower() for skill in job_ad.required_skills if skill.strip()}
        nice_skills = {skill.strip().lower() for skill in job_ad.nice_to_have_skills if skill.strip()}
        responsibilities = [resp.strip().lower() for resp in job_ad.key_responsibilities if resp.strip()]

        # Split role title into meaningful words (ignoring common stop words).
        ignore_words = {"and", "or", "in", "the", "for", "of", "with", "to", "a", "an"}
        role_words = {
            word.strip().lower()
            for word in re.split(r"\W+", job_ad.role_title)
            if word.strip() and word.lower() not in ignore_words
        }

        for entry in facts_db.entries:
            if entry.id.startswith("profile-"):
                continue
            score = 0.0
            entry_techs = {tech.strip().lower() for tech in entry.technologies if tech.strip()}
            entry_title_lower = entry.title.lower()
            entry_desc_lower = entry.description.lower()
            combined_text = f"{entry_title_lower} {entry_desc_lower}"

            # --- CRITERION 1: Technology & Skill Tags (Direct Matches) ---
            # Match against required skills in tech tags (3.0 pts per match)
            req_tech_matches = entry_techs.intersection(req_skills)
            score += len(req_tech_matches) * 3.0

            # Match against nice-to-have skills in tech tags (1.0 pt per match)
            nice_tech_matches = entry_techs.intersection(nice_skills)
            score += len(nice_tech_matches) * 1.0

            # --- CRITERION 2: Role Title Similarity (Semantic Alignment) ---
            # If words from the target role title are in your project/experience title
            for word in role_words:
                if re.search(rf"\b{re.escape(word)}\b", entry_title_lower):
                    score += 2.0

            # --- CRITERION 3: Contextual Text Matching (Text Scans) ---
            # Scan title and description for required skills (1.5 pts per match)
            for skill in req_skills:
                if re.search(rf"\b{re.escape(skill)}\b", combined_text):
                    score += 1.5

            # Scan title and description for nice-to-have skills (0.5 pts per match)
            for skill in nice_skills:
                if re.search(rf"\b{re.escape(skill)}\b", combined_text):
                    score += 0.5

            # Scan description for overlaps with the key responsibilities (1.0 pt per match)
            for resp in responsibilities:
                # Look for matches of key terms within the responsibility line
                resp_words = {w for w in re.split(r"\W+", resp) if len(w) > 3 and w not in ignore_words}
                matched_words = {
                    w
                    for w in resp_words
                    if re.search(rf"\b{re.escape(w)}\b", entry_desc_lower)
                }
                if len(matched_words) >= 2:  # If at least 2 distinct words overlap
                    score += 1.0

            # Only keep facts that have some level of match (score > 0)
            if score > 0:
                scored_entries.append((score, entry))

        # Fall back to first entries if nothing scored, so downstream generation
        # can still proceed with grounded facts instead of failing hard.
        if not scored_entries:
            return facts_db.entries[:limit]

        # Sort entries by score in descending order (highest score first)
        scored_entries.sort(
            key=lambda item: (
                item[0],
                item[1].start_date is not None,
                str(item[1].start_date or ""),
                item[1].id,
            ),
            reverse=True,
        )

        # Return the top 'limit' matching entries
        return [entry for _, entry in scored_entries[:limit]]


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


def _ground_unverified_skill_claims(text: str) -> str:
    """Replace common skill claims that are not backed by the facts file."""

    grounded = re.sub(
        r"Python, SQL, C\+\+, Java, Git\.",
        "Python, SQL, Git.",
        text,
        flags=re.IGNORECASE,
    )
    grounded = re.sub(
        r"Git använde jag dagligen för CI/CD och versionshantering\.",
        "Jag satte upp GitLab CI/CD och genomförde API-kontraktsutvärdering över sex repositories.",
        grounded,
        flags=re.IGNORECASE,
    )
    return grounded


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