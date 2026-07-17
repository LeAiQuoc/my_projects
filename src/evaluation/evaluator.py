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


_AI_TONE_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bi am excited to leverage my skills in\b", "i am excited to leverage my skills in"),
    (r"\bexcited to leverage\b", "excited to leverage"),
    (r"\baligns with\b", "aligns with"),
    (r"\bplays a crucial role\b", "plays a crucial role"),
    (r"\blook forward to the opportunity to discuss further\b", "look forward to the opportunity to discuss further"),
    (r"\brelevant tooling in my experience includes\b", "relevant tooling in my experience includes"),
    (r"\bmy technical foundation\b", "my technical foundation"),
)

_STRUCTURAL_CLOSING_PHRASES: tuple[tuple[str, str], ...] = (
    (r"\bi am excited to leverage my skills in\b", "i am excited to leverage my skills in"),
    (r"\bmy technical foundation .*? aligns with\b", "my technical foundation ... aligns with"),
    (r"\brelevant tooling in my experience includes\b", "relevant tooling in my experience includes"),
    (r"\bi look forward to the opportunity to discuss further\b", "i look forward to the opportunity to discuss further"),
    (r"\blook forward to the opportunity to discuss further\b", "look forward to the opportunity to discuss further"),
)


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

        fingerprints = self._fingerprint_check(draft)
        scores["fingerprint_check"] = fingerprints.score
        if not fingerprints.passed:
            issues.extend(fingerprints.issues)

        hallucination = await self._hallucination_check(draft, facts, job_ad)
        scores["hallucination"] = hallucination.score
        if not hallucination.passed:
            issues.extend(hallucination.issues)

        coverage = self._requirement_coverage_check(draft, job_ad)
        scores["requirement_coverage"] = coverage.score
        if not coverage.passed:
            issues.extend(coverage.issues)

        language_match = self._language_match_check(draft, job_ad)
        scores["language_match"] = language_match.score
        if not language_match.passed:
            issues.extend(language_match.issues)

        cover_length = self._cover_letter_length_check(draft, job_ad)
        scores["cover_letter_length"] = cover_length.score
        if not cover_length.passed:
            issues.extend(cover_length.issues)

        style = self._style_match_check(draft, style_profile)
        scores["style_match"] = style.score
        if not style.passed:
            issues.extend(style.issues)

        ai_tone = self._ai_tone_check(draft)
        scores["ai_tone"] = ai_tone.score
        if not ai_tone.passed:
            issues.extend(ai_tone.issues)

        structural_tone = self._cover_letter_structural_check(draft)
        scores["cover_letter_structure"] = structural_tone.score
        if not structural_tone.passed:
            issues.extend(structural_tone.issues)

        rhythm_uniformity = self._rhythm_uniformity_check(draft)
        scores["rhythm_uniformity"] = rhythm_uniformity.score
        if not rhythm_uniformity.passed:
            issues.extend(rhythm_uniformity.issues)

        cliche = self._corporate_cliche_filter(draft)
        scores["cliche_filter"] = cliche.score
        if not cliche.passed:
            issues.extend(cliche.issues)

        return EvaluationResult(
            passed=len(issues) == 0,
            issues=issues,
            per_check_scores=scores,
        )

    async def _hallucination_check(self, draft: str, facts: FactsDatabase, job_ad: JobAd) -> "_CheckResult":
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
            "Treat paraphrases as supported when a facts entry conveys the same meaning. "
            "Do not flag wording differences alone as unsupported. "
            "Only flag a claim when no semantically equivalent support exists in any facts entry. "
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
        unsupported = _filter_supported_claims(unsupported, facts, job_ad)
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

    def _language_match_check(self, draft: str, job_ad: JobAd) -> "_CheckResult":
        """Ensure Swedish ads produce Swedish draft language."""

        if not job_ad.source_language.lower().startswith("sv"):
            return _CheckResult(passed=True, score=1.0, issues=[])

        english_markers = (
            "dear hiring team",
            "i am ",
            "i’m ",
            "my background",
            "during my internship",
            "i also",
            "i built",
            "i work",
            "my experience",
            "i’m ready",
        )
        lower = draft.lower()
        hits = sum(1 for marker in english_markers if marker in lower)
        if hits == 0:
            return _CheckResult(passed=True, score=1.0, issues=[])

        score = max(0.0, 1.0 - (hits / len(english_markers)))
        return _CheckResult(
            passed=False,
            score=score,
            issues=["language mismatch: cover letter should be written in Swedish for this job ad"],
        )

    def _cover_letter_length_check(self, draft: str, job_ad: JobAd) -> "_CheckResult":
        """Keep the cover letter within the requested 200-300 word range."""

        cover_letter = _extract_cover_letter_body(draft) or draft
        word_count = len(cover_letter.split())
        if 200 <= word_count <= 300:
            return _CheckResult(passed=True, score=1.0, issues=[])

        if word_count < 200:
            return _CheckResult(
                passed=False,
                score=max(0.0, word_count / 200.0),
                issues=[f"cover letter too short; got {word_count} words, target 200-300"],
            )

        return _CheckResult(
            passed=False,
            score=max(0.0, 300.0 / word_count),
            issues=[f"cover letter too long; got {word_count} words, target 200-300"],
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
            r"\bi am excited to leverage\b",
            r"\bdynamic environment\b",
            r"\bteam player\b",
            r"\bleverage my\b",
            r"\bpassionate about\b",
            r"\blook forward to the opportunity to discuss further\b",
            r"\bplays a crucial role\b",
            r"\brelevant tooling in my experience includes\b",
            r"\baligns with\b",
        )
        found_phrases = [
            pattern.replace(r"\b", "")
            for pattern in generic_phrases
            if re.search(pattern, lower)
        ]

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

    def _cover_letter_structural_check(self, draft: str) -> "_CheckResult":
        """Catch stock cover-letter structure that reads generic across runs."""

        cover_letter = _extract_cover_letter_body(draft)
        if not cover_letter:
            return _CheckResult(passed=True, score=1.0, issues=[])

        lower = cover_letter.lower()
        phrase_patterns: tuple[tuple[str, str], ...] = _STRUCTURAL_CLOSING_PHRASES + (
            (r"\bi am ready to contribute\b", "i am ready to contribute"),
            (r"\bcompetence development opportunities\b", "competence development opportunities"),
            (r"\bproduction-facing tools\b", "production-facing tools"),
        )
        found_phrases = [
            label
            for pattern, label in phrase_patterns
            if re.search(pattern, lower)
        ]

        body_paragraphs = [p for p in _paragraphs(cover_letter) if not p.lower().startswith("dear hiring team")]
        symmetry_issue = None
        if len(body_paragraphs) >= 3:
            paragraph_lengths = [len(paragraph.split()) for paragraph in body_paragraphs]
            avg = mean(paragraph_lengths)
            if avg > 0:
                spread = max(paragraph_lengths) - min(paragraph_lengths)
                if spread <= max(10, int(avg * 0.15)):
                    symmetry_issue = (
                        "template structure risk: cover-letter paragraphs are too even in length "
                        f"({', '.join(str(length) for length in paragraph_lengths)} words)"
                    )

        issues: list[str] = []
        if found_phrases:
            issues.append(
                "template structure risk: stock phrases found "
                f"({', '.join(found_phrases)})"
            )
        if symmetry_issue is not None:
            issues.append(symmetry_issue)

        if not issues:
            return _CheckResult(passed=True, score=1.0, issues=[])

        penalty = min(1.0, len(issues) * 0.35)
        return _CheckResult(passed=False, score=max(0.0, 1.0 - penalty), issues=issues)

    def _rhythm_uniformity_check(self, draft: str) -> "_CheckResult":
        """Detect metronomic sentence rhythm that often reads as templated AI prose."""

        paragraphs = _paragraphs(draft)
        if not paragraphs:
            return _CheckResult(passed=True, score=1.0, issues=[])

        min_cv = 1.0
        uniform_details: list[str] = []
        for idx, paragraph in enumerate(paragraphs, start=1):
            lengths = _sentence_lengths(paragraph)
            if len(lengths) < 3:
                continue
            avg = mean(lengths)
            if avg <= 0:
                continue
            cv = pstdev(lengths) / avg
            min_cv = min(min_cv, cv)
            if cv < 0.22:
                uniform_details.append(f"p{idx} cv={cv:.2f}")

        if not uniform_details:
            return _CheckResult(passed=True, score=1.0, issues=[])

        score = max(0.0, min(1.0, min_cv / 0.22))
        return _CheckResult(
            passed=False,
            score=score,
            issues=[
                "rhythm uniformity risk: sentence lengths are too even "
                f"({', '.join(uniform_details)})"
            ],
        )

    def _fingerprint_check(self, draft: str) -> "_CheckResult":
        """Detect high-confidence AI copy/paste fingerprints deterministically."""

        placeholder_patterns = (
            r"\[(?:Your|Insert|Add|Enter|Recipient|Sender|Subject|Position|Company Name)[^\]\n]{0,80}\]",
            r"\b(?:19|20)\d{2}-XX-XX\b",
            r"\bXX/XX/(?:19|20)\d{2}\b",
            r"<!--\s*(?:add|fill\s+in|insert|todo|placeholder)[^>]{0,120}-->",
        )
        citation_patterns = (
            r"\boaicite\b",
            r"\bcontentReference\s*\[oaicite:[^\]]+\]",
            r"\boai_citation\b",
            r"\bgrok_card\b",
        )
        utm_patterns = (
            r"[?&]utm_source=(?:chatgpt|openai|copilot|claude|grok|gemini|perplexity)(?:\.com|\.ai)?\b",
            r"[?&]referrer=(?:chatgpt|copilot|grok|claude|gemini|perplexity)\.(?:com|ai)\b",
        )

        hits: list[str] = []
        for pattern in placeholder_patterns:
            if re.search(pattern, draft, flags=re.IGNORECASE):
                hits.append("placeholder")
                break
        for pattern in citation_patterns:
            if re.search(pattern, draft, flags=re.IGNORECASE):
                hits.append("citation-markup")
                break
        for pattern in utm_patterns:
            if re.search(pattern, draft, flags=re.IGNORECASE):
                hits.append("ai-tracking-link")
                break

        if not hits:
            return _CheckResult(passed=True, score=1.0, issues=[])

        unique_hits = sorted(set(hits))
        penalty = min(1.0, 0.5 + (0.2 * len(unique_hits)))
        return _CheckResult(
            passed=False,
            score=max(0.0, 1.0 - penalty),
            issues=[f"fingerprint check fail: found {', '.join(unique_hits)}"],
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


def _filter_supported_claims(claims: list[str], facts: FactsDatabase, job_ad: JobAd) -> list[str]:
    """Drop unsupported-claim hits that are already supported by facts."""

    additional_texts = [
        job_ad.company_context or "",
        job_ad.source_text or "",
        job_ad.role_title,
        job_ad.company_name,
        job_ad.tone_signals,
        " ".join(job_ad.required_skills),
        " ".join(job_ad.nice_to_have_skills),
        " ".join(job_ad.key_responsibilities),
    ]
    return [
        claim
        for claim in claims
        if not _claim_is_supported_by_facts(claim, facts, additional_texts)
    ]


def _claim_is_supported_by_facts(claim: str, facts: FactsDatabase, additional_texts: list[str] | None = None) -> bool:
    """Heuristically accept fact-backed paraphrases and multi-fact summaries."""

    claim_norm = _normalize_text(claim)
    if not claim_norm:
        return False

    fact_blobs = [
        _normalize_text(f"{entry.title} {entry.description} {' '.join(entry.technologies)}")
        for entry in facts.entries
    ]
    combined_blob = " ".join(fact_blobs)
    extra_blobs = [_normalize_text(text) for text in (additional_texts or []) if _normalize_text(text)]
    combined_support_blob = " ".join([combined_blob, *extra_blobs])

    if any(claim_norm in blob for blob in fact_blobs):
        return True
    if claim_norm in combined_support_blob:
        return True

    claim_tokens = _meaningful_tokens(claim_norm)
    if not claim_tokens:
        return False

    combined_tokens = set(_meaningful_tokens(combined_support_blob))
    overlap = sum(1 for token in claim_tokens if token in combined_tokens)
    overlap_ratio = overlap / len(claim_tokens)

    if overlap_ratio >= 0.72:
        return True

    summary_markers = ("technical projects", "independent programming work", "spare time")
    if any(marker in claim_norm for marker in summary_markers) and overlap_ratio >= 0.55:
        return True

    if "independent programming work" in claim_norm:
        has_personal_programming_fact = any(
            "spare time" in _normalize_text(entry.description)
            or "personal programming practice" in _normalize_text(entry.title)
            for entry in facts.entries
        )
        has_project_fact = any(entry.category == "project" for entry in facts.entries)
        if has_personal_programming_fact and has_project_fact:
            return True

    return False


def _normalize_text(text: str) -> str:
    """Normalize text for simple semantic-support heuristics."""

    lowered = text.lower().replace("-", " ").replace("/", " ")
    lowered = re.sub(r"[^a-z0-9+\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _meaningful_tokens(text: str) -> list[str]:
    """Extract content tokens while ignoring generic summary glue."""

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "both", "build", "built",
        "by", "can", "clearly", "combine", "combined", "consistent", "currently",
        "demonstrate", "demonstrates", "experience", "for", "from", "have", "has", "i",
        "in", "including", "into", "is", "it", "my", "of", "on", "or", "that",
        "the", "their", "these", "this", "through", "to", "various", "where", "with",
        "work", "worked", "working",
    }
    return [token for token in text.split() if len(token) > 2 and token not in stop_words]


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


def _extract_cover_letter_body(text: str) -> str:
    """Extract the cover-letter slice from a combined CV + cover-letter draft."""

    match = re.search(r"(?:^|\n\n)(dear hiring team,|hej rekryteringsteamet,|hej,)(.*)\Z", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return f"{match.group(1)}{match.group(2)}".strip()
