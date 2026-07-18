"""Template batch pipeline for post-phase-3 implementation."""

from __future__ import annotations

import asyncio
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
        """Run the full loop across job ads with bounded concurrency."""

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run_one(job_ad: JobAd) -> RankedBatchResult:
            async with semaphore:
                result = await self.orchestrator.run(facts, job_ad, style_profile)
                return RankedBatchResult(
                    job_ad=job_ad,
                    result=result,
                    fit_score=self._compute_fit_score(facts, job_ad),
                )

        results = await asyncio.gather(*[_run_one(job_ad) for job_ad in job_ads])
        return sorted(results, key=lambda item: item.fit_score, reverse=True)

    def _compute_fit_score(self, facts: FactsDatabase, job_ad: JobAd) -> float:
        """Score how well the facts database overlaps the job requirements."""

        required_skills = _normalize_text_items(job_ad.required_skills)
        nice_to_have_skills = _normalize_text_items(job_ad.nice_to_have_skills)
        if not required_skills and not nice_to_have_skills:
            return 0.0

        fact_terms = _collect_fact_terms(facts)
        required_overlap = sum(1 for skill in required_skills if skill in fact_terms)
        nice_overlap = sum(1 for skill in nice_to_have_skills if skill in fact_terms)

        weighted_matches = (2 * required_overlap) + nice_overlap
        return float(weighted_matches)


def _collect_fact_terms(facts: FactsDatabase) -> set[str]:
    """Collect normalized searchable terms from the facts database."""

    terms: set[str] = set()
    for entry in facts.entries:
        terms.update(_normalize_text_items(entry.technologies))
        terms.update(_tokenize(entry.title))
        terms.update(_tokenize(entry.description))
    return terms


def _normalize_text_items(items: Sequence[str]) -> set[str]:
    """Normalize a list of strings into lower-case canonical values."""

    return {_normalize_token(item) for item in items if _normalize_token(item)}


def _tokenize(text: str) -> set[str]:
    """Split text into searchable tokens."""

    cleaned = text.replace("/", " ").replace("-", " ").lower()
    return {
        token
        for token in cleaned.split()
        if token and len(token) > 2
    }


def _normalize_token(value: str) -> str:
    """Normalize a single skill or term for comparison."""

    return value.strip().lower().replace("/", " ").replace("-", " ")
