"""Draft evaluation for hallucination, coverage, style, and AI tone."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from statistics import mean, pstdev
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, field_validator

from src.facts.facts_schema import FactsDatabase
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


class EvaluationResult(BaseModel):
    """Structured result returned by the evaluator."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    per_check_scores: dict[str, float] = Field(default_factory=dict)

    @field_validator("issues")
    @classmethod
    def _trim_issues(cls, value: list[str]) -> list[str]:
        """Keep evaluator issue text compact and readable."""

        return [issue.strip() for issue in value if issue.strip()]


@dataclass(slots=True)
class Evaluator:
    """Coordinate the checks that decide whether a draft is ready to ship."""

    client: Any | None = None
    model: str = "deepseek-chat"

    async def evaluate(
        self,
        draft: str,
        facts: FactsDatabase,
        job_ad: JobAd,
        style_profile: StyleProfile,
    ) -> EvaluationResult:
        """Run all evaluator checks and aggregate pass/fail with per-check scores."""

        issues: list[str] = []
        scores: dict[str, float] = {}

        hallucination = await self._hallucination_check(draft, facts)
        scores["hallucination"] = hallucination.score
        if not hallucination.passed:
            issues.extend(hallucination.issues)

        coverage = self._requirement_coverage_check(draft, job_ad)
        scores["requirement_coverage"] = coverage.score
        if not coverage.passed:
            issues.extend(coverage.issues)

        style = self._style_match_check(draft, style_profile)
        scores["style_match"] = style.score
        if not style.passed:
            issues.extend(style.issues)

        ai_tone = self._ai_tone_check(draft)
        scores["ai_tone"] = ai_tone.score
        if not ai_tone.passed:
            issues.extend(ai_tone.issues)

        cliche = self._corporate_cliche_filter(draft)
        scores["cliche_filter"] = cliche.score
        if not cliche.passed:
            issues.extend(cliche.issues)

        return EvaluationResult(
            passed=len(issues) == 0,
            issues=issues,
            per_check_scores=scores,
        )

    async def _hallucination_check(self, draft: str, facts: FactsDatabase) -> "_CheckResult":
        """Use Deepseek to identify unsupported claims against the facts database."""

        client = self.client or _create_default_client()
        if client is None:
            return _CheckResult(
                passed=False,
                score=0.0,
                issues=["hallucination check unavailable: Deepseek client not configured"],
            )

        facts_block = "\n".join(_format_fact(entry) for entry in facts.entries)
        system_prompt = (
            "You are a strict factual verifier. "
            "Given draft text and a facts database, detect every unsupported claim. "
            "Return JSON only with keys: unsupported_claims (list[str]), confidence (0-1)."
        )
        user_prompt = (
            "Facts database:\n"
            f"{facts_block}\n\n"
            "Draft:\n"
            f"{draft}\n\n"
            "Respond in JSON only."
        )

        try:
            response = await client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OpenAIError as exc:
            return _CheckResult(
                passed=False,
                score=0.0,
                issues=[f"hallucination check failed: {exc}"],
            )

        content = _extract_response_text(response)
        unsupported = _parse_unsupported_claims(content)
        if not unsupported:
            return _CheckResult(passed=True, score=1.0, issues=[])

        penalty = min(1.0, len(unsupported) * 0.2)
        return _CheckResult(
            passed=False,
            score=max(0.0, 1.0 - penalty),
            issues=[f"unsupported claim: {claim}" for claim in unsupported],
        )

    def _requirement_coverage_check(self, draft: str, job_ad: JobAd) -> "_CheckResult":
        """Ensure the draft addresses required skills from the job ad."""

        required = [skill.strip().lower() for skill in job_ad.required_skills if skill.strip()]
        if not required:
            return _CheckResult(passed=True, score=1.0, issues=[])

        draft_lower = draft.lower()
        matched = [
            skill
            for skill in required
            if re.search(rf"\b{re.escape(skill)}\b", draft_lower)
        ]
        ratio = len(matched) / len(required)
        if ratio >= 0.6:
            return _CheckResult(passed=True, score=ratio, issues=[])

        missing = [skill for skill in required if skill not in matched]
        return _CheckResult(
            passed=False,
            score=ratio,
            issues=[f"requirement coverage too low; missing: {', '.join(missing)}"],
        )

    def _style_match_check(self, draft: str, style_profile: StyleProfile) -> "_CheckResult":
        """Compare sentence-length statistics against the style profile."""

        sentence_lengths = _sentence_lengths(draft)
        if not sentence_lengths:
            return _CheckResult(passed=False, score=0.0, issues=["style check failed: empty draft"])

        draft_avg = mean(sentence_lengths)
        draft_variance = pstdev(sentence_lengths) ** 2 if len(sentence_lengths) > 1 else 0.0

        avg_delta = abs(draft_avg - style_profile.avg_sentence_length)
        variance_delta = abs(draft_variance - style_profile.sentence_length_variance)

        avg_score = max(0.0, 1.0 - (avg_delta / max(1.0, style_profile.avg_sentence_length)))
        variance_score = max(
            0.0,
            1.0 - (variance_delta / max(1.0, style_profile.sentence_length_variance + 1.0)),
        )
        score = (avg_score + variance_score) / 2

        if score >= 0.5:
            return _CheckResult(passed=True, score=score, issues=[])

        return _CheckResult(
            passed=False,
            score=score,
            issues=[
                "style mismatch: sentence-length profile differs from style profile "
                f"(avg delta={avg_delta:.2f}, variance delta={variance_delta:.2f})"
            ],
        )

    def _ai_tone_check(self, draft: str) -> "_CheckResult":
        """Flag templated filler language and overly uniform rhythm."""

        lower = draft.lower()
        generic_phrases = (
            "i am excited to",
            "dynamic environment",
            "team player",
            "leverage my",
            "passionate about",
        )
        found_phrases = [phrase for phrase in generic_phrases if phrase in lower]

        paragraph_lengths = [len(paragraph.split()) for paragraph in _paragraphs(draft)]
        uniform_rhythm = False
        if len(paragraph_lengths) >= 3:
            spread = max(paragraph_lengths) - min(paragraph_lengths)
            uniform_rhythm = spread <= 8

        issues: list[str] = []
        if found_phrases:
            issues.append(f"AI-tone risk: generic phrases found ({', '.join(found_phrases)})")
        if uniform_rhythm:
            issues.append("AI-tone risk: paragraph rhythm appears overly uniform")

        if not issues:
            return _CheckResult(passed=True, score=1.0, issues=[])

        penalty = min(1.0, len(issues) * 0.4)
        return _CheckResult(passed=False, score=max(0.0, 1.0 - penalty), issues=issues)

    def _corporate_cliche_filter(self, draft: str) -> "_CheckResult":
        """Deterministically fail drafts containing banned cliche words."""

        banned_words = (
            "delve",
            "tapestry",
            "testament",
            "pioneer",
            "bespoke",
            "seamlessly",
            "foster",
            "ultimate",
            "furthermore",
            "moreover",
        )
        lower = draft.lower()
        hits = [word for word in banned_words if re.search(rf"\b{re.escape(word)}\b", lower)]
        if not hits:
            return _CheckResult(passed=True, score=1.0, issues=[])

        return _CheckResult(
            passed=False,
            score=0.0,
            issues=[f"cliche filter fail: banned words present ({', '.join(hits)})"],
        )


@dataclass(slots=True)
class _CheckResult:
    """Internal container for one evaluator check outcome."""

    passed: bool
    score: float
    issues: list[str]


def _create_default_client() -> AsyncOpenAI | None:
    """Create Deepseek-compatible async client from environment settings."""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _extract_response_text(response: Any) -> str:
    """Extract normalized text from OpenAI-compatible completion payloads."""

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


def _parse_unsupported_claims(text: str) -> list[str]:
    """Parse unsupported claims from JSON-like model output."""

    if not text.strip():
        return []
    try:
        payload = json.loads(text)
        claims = payload.get("unsupported_claims", [])
        if isinstance(claims, list):
            return [str(claim).strip() for claim in claims if str(claim).strip()]
        return []
    except json.JSONDecodeError:
        # Fallback: parse simple bullet lines if the model did not return valid JSON.
        lines = [line.strip("- \t") for line in text.splitlines() if line.strip()]
        return [line for line in lines if line]


def _format_fact(entry: Any) -> str:
    """Serialize one fact entry to compact verifier context text."""

    technologies = ", ".join(entry.technologies) if entry.technologies else "N/A"
    start_date = str(entry.start_date) if entry.start_date else "N/A"
    end_date = str(entry.end_date) if entry.end_date else "Present"
    evidence = entry.evidence_url or "N/A"
    return (
        f"id={entry.id}; category={entry.category}; title={entry.title}; "
        f"description={entry.description}; technologies={technologies}; "
        f"start={start_date}; end={end_date}; evidence={evidence}"
    )


def _sentence_lengths(text: str) -> list[int]:
    """Return sentence lengths in words for style matching."""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return [len(sentence.split()) for sentence in sentences if sentence.split()]


def _paragraphs(text: str) -> list[str]:
    """Split draft into non-empty paragraphs."""

    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
