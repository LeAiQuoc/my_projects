"""Template for generate-evaluate-retry orchestration (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.facts.facts_schema import FactsDatabase
from src.generation.cover_letter_generator import CoverLetterGenerator
from src.generation.cv_generator import CVGenerator
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
    """Template orchestrator for the future bounded retry loop."""

    def __init__(
        self,
        cv_generator: CVGenerator,
        cover_letter_generator: CoverLetterGenerator,
        evaluator: Evaluator,
        max_retries: int = 3,
        logger: Any | None = None,
    ) -> None:
        self.cv_generator = cv_generator
        self.cover_letter_generator = cover_letter_generator
        self.evaluator = evaluator
        self.max_retries = max_retries
        self.logger = logger

    async def run(
        self,
        facts: FactsDatabase,
        job_ad: JobAd,
        style_profile: StyleProfile,
    ) -> OrchestrationResult:
        """Run the phase-6 orchestration flow once implemented."""

        _ = facts, job_ad, style_profile
        raise NotImplementedError

    def _build_correction_note(self, issues: list[str]) -> str | None:
        """Template for converting evaluator issues into retry guidance."""

        _ = issues
        raise NotImplementedError
