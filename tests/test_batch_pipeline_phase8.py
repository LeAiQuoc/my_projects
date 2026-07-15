from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from src.evaluation.evaluator import EvaluationResult
from src.facts.facts_schema import FactsDatabase, FactsEntry
from src.job_ads.schema import JobAd
from src.loop.orchestrator import OrchestrationResult
from src.pipeline.batch import BatchPipeline
from src.style.style_profile import StyleProfile


def _facts() -> FactsDatabase:
    return FactsDatabase.from_entries(
        [
            FactsEntry(
                id="exp-1",
                category="experience",
                title="Backend Engineer",
                description="Built Python ETL pipelines and SQL quality checks.",
                technologies=["Python", "SQL"],
            ),
            FactsEntry(
                id="proj-1",
                category="project",
                title="Automation Platform",
                description="Created Docker-based automation scripts.",
                technologies=["Docker"],
            ),
        ]
    )


def _style() -> StyleProfile:
    return StyleProfile(
        tone_description="direct and professional",
        avg_sentence_length=14.0,
        sentence_length_variance=3.0,
        characteristic_phrases=["I focused on"],
        phrases_to_avoid=["I am passionate about"],
        structural_notes="brief intro, evidence-heavy body, concise close",
        anchor_snippets=["I prioritize clear outcomes and measurable impact."],
    )


def _job(company_name: str, required_skills: list[str], nice_to_have_skills: list[str] | None = None) -> JobAd:
    return JobAd(
        company_name=company_name,
        role_title="Data Engineer",
        required_skills=required_skills,
        nice_to_have_skills=nice_to_have_skills or [],
        tone_signals="neutral professional",
        key_responsibilities=["build pipelines"],
    )


@dataclass
class _TrackingOrchestrator:
    active: int = 0
    max_active: int = 0

    async def run(self, facts: FactsDatabase, job_ad: JobAd, style_profile: StyleProfile) -> OrchestrationResult:
        _ = facts, style_profile
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.05)
            return OrchestrationResult(
                cv_draft=f"cv::{job_ad.company_name}",
                cover_letter_draft=f"letter::{job_ad.company_name}",
                evaluation=EvaluationResult(
                    passed=True,
                    issues=[],
                    per_check_scores={"fit": 1.0},
                ),
                attempts=1,
                unresolved_issues=[],
            )
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_batch_pipeline_limits_concurrency_and_ranks_by_fit_score() -> None:
    orchestrator = _TrackingOrchestrator()
    pipeline = BatchPipeline(orchestrator=orchestrator, max_concurrency=2)

    job_ads = [
        _job("Alpha", ["Python", "SQL"]),
        _job("Beta", ["Python"]),
        _job("Gamma", ["Docker"]),
        _job("Delta", ["SQL"], ["Docker"]),
    ]

    results = await pipeline.run(_facts(), job_ads, _style())

    assert orchestrator.max_active <= 2
    assert [item.job_ad.company_name for item in results] == ["Alpha", "Delta", "Beta", "Gamma"]
    assert results[0].fit_score > results[1].fit_score > results[2].fit_score >= results[3].fit_score


def test_batch_pipeline_fit_score_reflects_overlap() -> None:
    pipeline = BatchPipeline(orchestrator=_TrackingOrchestrator(), max_concurrency=2)

    high_fit = pipeline._compute_fit_score(_facts(), _job("Alpha", ["Python", "SQL"], ["Docker"]))
    low_fit = pipeline._compute_fit_score(_facts(), _job("Omega", ["Kubernetes"]))

    assert high_fit > low_fit
    assert low_fit == 0.0