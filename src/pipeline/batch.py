"""Template batch pipeline for post-phase-3 implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.facts.facts_schema import FactsDatabase
from src.job_ads.schema import JobAd
from src.loop.orchestrator import GenerationOrchestrator, OrchestrationResult
from src.style.style_profile import StyleProfile


@dataclass(slots=True)
class RankedBatchResult:
    """A single job ad result with a computed fit score."""

    job_ad: JobAd
    result: OrchestrationResult
    fit_score: float


class BatchPipeline:
    """Template batch runner for future multi-job orchestration."""

    def __init__(self, orchestrator: GenerationOrchestrator, max_concurrency: int = 3) -> None:
        self.orchestrator = orchestrator
        self.max_concurrency = max_concurrency

    async def run(
        self,
        facts: FactsDatabase,
        job_ads: Sequence[JobAd],
        style_profile: StyleProfile,
    ) -> list[RankedBatchResult]:
        """Run batch processing once the orchestration phase is implemented."""

        _ = facts, job_ads, style_profile
        raise NotImplementedError

    def _compute_fit_score(self, facts: FactsDatabase, job_ad: JobAd) -> float:
        """Template for future fit-score calculation."""

        _ = facts, job_ad
        raise NotImplementedError
