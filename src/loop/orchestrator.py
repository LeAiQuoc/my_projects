"""Template for generate-evaluate-retry orchestration (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.facts.facts_schema import FactsDatabase
from src.generation.cover_letter_generator import CoverLetterGenerator
from src.generation.cv_generator import CVGenerator
from src.generation.humanize_pass import rewrite_for_natural_rhythm
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
                cv_draft = await rewrite_for_natural_rhythm(cv_draft, style_profile)
                cover_letter_draft = await rewrite_for_natural_rhythm(cover_letter_draft, style_profile)
                combined_rewritten = f"{sanitize_draft(cv_draft)}\n\n{sanitize_draft(cover_letter_draft)}".strip()

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
        return "; ".join(issue for issue in issues if issue)


def _should_run_humanize_pass(issues: list[str]) -> bool:
    """Detect whether issues suggest templated rhythm that rewriting can improve."""

    return any(
        "ai-tone" in issue.lower() or "style mismatch" in issue.lower()
        for issue in issues
    )
