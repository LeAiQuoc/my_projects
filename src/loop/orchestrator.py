"""Template for generate-evaluate-retry orchestration (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.facts.facts_schema import FactsDatabase
from src.generation.cover_letter_generator import CoverLetterGenerator, _cap_cover_letter_length
from src.generation.cv_generator import CVGenerator
from src.generation.humanize_pass import apply_deterministic_humanize_cleanup, rewrite_for_natural_rhythm
from src.generation.sanitizer import sanitize_draft
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


@dataclass(slots=True)
class OrchestrationResult:
    """Final output from the generate-evaluate loop."""

    cv_draft: str
    cover_letter_draft: str
    evaluation: EvaluationResult
    attempts: int
    unresolved_issues: list[str]


class GenerationOrchestrator:
    """Generate, sanitize, evaluate, and retry drafts in a bounded loop."""

    def __init__(
        self,
        cv_generator: CVGenerator,
        cover_letter_generator: CoverLetterGenerator,
        evaluator: Evaluator,
        max_retries: int = 3,
        logger: Any | None = None,
        enable_humanize_pass: bool = True,
    ) -> None:
        self.cv_generator = cv_generator
        self.cover_letter_generator = cover_letter_generator
        self.evaluator = evaluator
        self.max_retries = max_retries
        self.logger = logger
        self.enable_humanize_pass = enable_humanize_pass

    async def run(
        self,
        facts: FactsDatabase,
        job_ad: JobAd,
        style_profile: StyleProfile,
    ) -> OrchestrationResult:
        """Run bounded orchestration with deterministic sanitization before evaluation."""

        last_evaluation: EvaluationResult | None = None
        unresolved_issues: list[str] = []
        cv_draft = ""
        cover_letter_draft = ""

        for attempt in range(1, self.max_retries + 1):
            correction_note = self._build_correction_note(unresolved_issues)
            cv_raw = await self.cv_generator.generate(
                facts,
                job_ad,
                style_profile,
                correction_note=correction_note,
            )
            cover_letter_raw = await self.cover_letter_generator.generate(
                facts,
                job_ad,
                style_profile,
                correction_note=correction_note,
            )

            # Deterministic cleanup runs before the evaluator to avoid avoidable
            # retries caused by known banned vocabulary.
            cv_draft = sanitize_draft(cv_raw)
            cover_letter_draft = sanitize_draft(cover_letter_raw)
            cover_letter_draft = _ensure_cover_letter_differentiators(cover_letter_draft, facts)
            cover_letter_draft = _strip_driver_license_for_software_roles(cover_letter_draft, job_ad)
            cover_letter_draft = _cap_cover_letter_length(cover_letter_draft)
            combined_draft = f"{cv_draft}\n\n{cover_letter_draft}".strip()

            last_evaluation = await self.evaluator.evaluate(
                combined_draft,
                facts,
                job_ad,
                style_profile,
            )
            unresolved_issues = list(last_evaluation.issues)

            # Phase 7: run the humanization rewrite when tone/rhythm signals are
            # flagged (AI-tone and/or sentence-profile style mismatch).
            if (
                self.enable_humanize_pass
                and not last_evaluation.passed
                and _should_run_humanize_pass(unresolved_issues)
            ):
                # Stage 1: deterministic residue cleanup before any LLM rewrite.
                pre_humanize_cv = _apply_pre_humanize_cleanup(cv_draft)
                pre_humanize_cover = _apply_pre_humanize_cleanup(cover_letter_draft)

                protected_spans = _collect_protected_spans(facts, job_ad)
                cv_stage1_drift = _find_protected_span_drift(cv_draft, pre_humanize_cv, protected_spans)
                cover_stage1_drift = _find_protected_span_drift(cover_letter_draft, pre_humanize_cover, protected_spans)
                cv_stage1_owner_drift = _find_ownership_marker_drift(cv_draft, pre_humanize_cv)
                cover_stage1_owner_drift = _find_ownership_marker_drift(cover_letter_draft, pre_humanize_cover)

                if cv_stage1_drift or cover_stage1_drift or cv_stage1_owner_drift or cover_stage1_owner_drift:
                    if self.logger is not None:
                        drift_details = "; ".join(
                            cv_stage1_drift + cover_stage1_drift + cv_stage1_owner_drift + cover_stage1_owner_drift
                        )
                        self.logger.warning("deterministic cleanup blocked by fidelity gate: %s", drift_details)
                    pre_humanize_cv = cv_draft
                    pre_humanize_cover = cover_letter_draft

                cv_scene_mode, cv_voice_mode = _select_rewrite_strategy(job_ad, document_type="cv")
                cover_scene_mode, cover_voice_mode = _select_rewrite_strategy(job_ad, document_type="cover_letter")

                cv_rewritten = await rewrite_for_natural_rhythm(
                    pre_humanize_cv,
                    style_profile,
                    voice_mode=cv_voice_mode,
                    scene_mode=cv_scene_mode,
                    language_code=job_ad.source_language,
                )
                cover_rewritten = await rewrite_for_natural_rhythm(
                    pre_humanize_cover,
                    style_profile,
                    voice_mode=cover_voice_mode,
                    scene_mode=cover_scene_mode,
                    language_code=job_ad.source_language,
                )

                # Stage 2: fidelity gate after voice pull.
                cv_drift = _find_protected_span_drift(pre_humanize_cv, cv_rewritten, protected_spans)
                cover_drift = _find_protected_span_drift(pre_humanize_cover, cover_rewritten, protected_spans)
                cv_owner_drift = _find_ownership_marker_drift(pre_humanize_cv, cv_rewritten)
                cover_owner_drift = _find_ownership_marker_drift(pre_humanize_cover, cover_rewritten)

                if cv_drift or cover_drift or cv_owner_drift or cover_owner_drift:
                    if self.logger is not None:
                        drift_details = "; ".join(cv_drift + cover_drift + cv_owner_drift + cover_owner_drift)
                        self.logger.warning("humanize fidelity gate blocked rewrite: %s", drift_details)
                    cv_draft = pre_humanize_cv
                    cover_letter_draft = pre_humanize_cover
                else:
                    # Stage 3: final deterministic residue scan.
                    cv_draft = _apply_pre_humanize_cleanup(cv_rewritten)
                    cover_letter_draft = _apply_pre_humanize_cleanup(cover_rewritten)
                    cover_letter_draft = _ensure_cover_letter_differentiators(cover_letter_draft, facts)
                    cover_letter_draft = _strip_driver_license_for_software_roles(cover_letter_draft, job_ad)
                    cover_letter_draft = _cap_cover_letter_length(cover_letter_draft)

                combined_rewritten = f"{cv_draft}\n\n{cover_letter_draft}".strip()

                rewritten_eval = await self.evaluator.evaluate(
                    combined_rewritten,
                    facts,
                    job_ad,
                    style_profile,
                )
                if rewritten_eval.passed:
                    return OrchestrationResult(
                        cv_draft=cv_draft,
                        cover_letter_draft=cover_letter_draft,
                        evaluation=rewritten_eval,
                        attempts=attempt,
                        unresolved_issues=[],
                    )
                last_evaluation = rewritten_eval
                unresolved_issues = list(rewritten_eval.issues)

            if self.logger is not None:
                self.logger.info(
                    "orchestrator attempt %s complete (passed=%s)",
                    attempt,
                    last_evaluation.passed,
                )

            if last_evaluation.passed:
                return OrchestrationResult(
                    cv_draft=cv_draft,
                    cover_letter_draft=cover_letter_draft,
                    evaluation=last_evaluation,
                    attempts=attempt,
                    unresolved_issues=[],
                )

        if last_evaluation is None:
            raise RuntimeError("orchestrator executed no attempts")

        return OrchestrationResult(
            cv_draft=cv_draft,
            cover_letter_draft=cover_letter_draft,
            evaluation=last_evaluation,
            attempts=self.max_retries,
            unresolved_issues=unresolved_issues,
        )

    def _build_correction_note(self, issues: list[str]) -> str | None:
        """Convert evaluation issues into a concise retry directive."""

        if not issues:
            return None

        directives: list[str] = []
        for issue in issues:
            issue_text = issue.strip()
            if not issue_text:
                continue
            lower = issue_text.lower()

            if lower.startswith("unsupported claim"):
                directives.append("Remove unsupported claims and keep wording strictly aligned with facts entries.")
                continue

            if lower.startswith("requirement coverage too low"):
                directives.append(
                    "Do not fabricate missing required skills; only mention skills explicitly present in facts."
                )
                continue

            if "rhythm uniformity risk" in lower:
                directives.append(
                    "Break the repeated rhythm: mix short punchy sentences with longer flowing ones across paragraphs."
                )
                continue

            if lower.startswith("template structure risk"):
                directives.append(
                    "Avoid stock fit phrases, vary paragraph lengths, and remove any tacked-on tooling summary sentence."
                )
                continue

            directives.append(issue_text)

        if not directives:
            return None

        unique_directives = list(dict.fromkeys(directives))
        return "; ".join(unique_directives)


def _should_run_humanize_pass(issues: list[str]) -> bool:
    """Detect whether issues suggest templated rhythm that rewriting can improve."""

    return any(
        "ai-tone" in issue.lower() or "style mismatch" in issue.lower()
        for issue in issues
    )


def _collect_protected_spans(facts: FactsDatabase, job_ad: JobAd) -> set[str]:
    """Collect selective factual anchors that should not disappear during humanization."""

    spans: set[str] = set()

    company = job_ad.company_name.strip()
    if company:
        spans.add(company)

    for entry in facts.entries:
        for tech in entry.technologies:
            cleaned = tech.strip()
            if cleaned:
                spans.add(cleaned)

        spans.update(_extract_numeric_spans(entry.description))

    return spans


def _find_protected_span_drift(original: str, rewritten: str, protected_spans: set[str]) -> list[str]:
    """Return missing protected spans that existed before rewrite but not after."""

    original_lower = original.lower()
    rewritten_lower = rewritten.lower()
    missing: list[str] = []

    for span in sorted(protected_spans):
        span_lower = span.lower()
        if span_lower not in original_lower:
            continue
        if span_lower in rewritten_lower:
            continue

        # Try word-boundary style matching for alphanumeric spans.
        escaped = re.escape(span_lower)
        pattern = rf"\b{escaped}\b"
        if re.search(pattern, original_lower) and not re.search(pattern, rewritten_lower):
            missing.append(f"missing protected span: {span}")
        elif span_lower in original_lower and span_lower not in rewritten_lower:
            missing.append(f"missing protected span: {span}")

    return missing


def _find_ownership_marker_drift(original: str, rewritten: str) -> list[str]:
    """Ensure explicit ownership markers are not fully erased by rewriting."""

    ownership_pattern = re.compile(r"\bi\s+(built|developed|handled|configured|worked|speak|hold)\b", re.IGNORECASE)
    original_hits = ownership_pattern.findall(original)
    rewritten_hits = ownership_pattern.findall(rewritten)
    if original_hits and not rewritten_hits:
        return ["missing ownership marker: first-person action verb"]
    return []


def _extract_numeric_spans(text: str) -> set[str]:
    """Extract numeric/date/unit spans that should remain stable."""

    return set(re.findall(r"\b\d+(?::\d+)?%?\b", text))


def _apply_pre_humanize_cleanup(text: str) -> str:
    """Run deterministic cleanup stage used before and after voice rewriting."""

    return apply_deterministic_humanize_cleanup(sanitize_draft(text))


def _ensure_cover_letter_differentiators(text: str, facts: FactsDatabase) -> str:
    """Ensure final cover letter keeps at least two advanced tooling terms from facts."""

    draft = text.strip()
    if not draft:
        return text

    available = _available_differentiators_from_facts(facts)
    if not available:
        return text

    draft_lower = draft.lower()
    present = [term for term in available if term.lower() in draft_lower]
    if present:
        return text

    sentence = f"Relevant tooling in my experience includes {available[0]}."
    return f"{draft}\n\n{sentence}"


def _available_differentiators_from_facts(facts: FactsDatabase) -> list[str]:
    """Collect advanced tooling differentiators explicitly present in facts."""

    candidates = ("RAG", "AI API integration", "MCP", "Prompt Engineering", "embeddings")
    blob = "\n".join(
        f"{entry.title} {entry.description} {' '.join(entry.technologies)}"
        for entry in facts.entries
    ).lower()

    found: list[str] = []
    for term in candidates:
        if term.lower() in blob:
            found.append(term)
    return found


def _strip_driver_license_for_software_roles(text: str, job_ad: JobAd) -> str:
    """Remove driver's-license lines for software roles unless explicitly required."""

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


def _select_rewrite_strategy(job_ad: JobAd, document_type: str) -> tuple[str, str]:
    """Choose scene + voice strategy from job tone and document type."""

    env_cv_mode = os.getenv("HUMANIZE_VOICE_MODE_CV")
    env_cover_mode = os.getenv("HUMANIZE_VOICE_MODE_COVER")
    env_cv_scene = os.getenv("HUMANIZE_SCENE_MODE_CV")
    env_cover_scene = os.getenv("HUMANIZE_SCENE_MODE_COVER")
    if document_type == "cv" and env_cv_mode:
        scene = (env_cv_scene or "docs").strip().lower()
        return scene, env_cv_mode.strip().lower()
    if document_type == "cover_letter" and env_cover_mode:
        scene = (env_cover_scene or "public-writing").strip().lower()
        return scene, env_cover_mode.strip().lower()

    tone = (job_ad.tone_signals or "").lower()
    if document_type == "cv":
        if any(token in tone for token in ("startup", "casual", "friendly")):
            return "docs", "professional"
        return "docs", "technical"

    if any(token in tone for token in ("startup", "casual", "friendly")):
        return "public-writing", "warm"
    return "public-writing", "professional"
