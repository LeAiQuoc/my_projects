from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.evaluation.evaluator import EvaluationResult
from src.facts.facts_schema import FactsDatabase, FactsEntry
from src.job_ads.schema import JobAd
from src.loop.orchestrator import GenerationOrchestrator
from src.style.style_profile import StyleProfile


@dataclass
class _CountingGenerator:
    prefix: str
    calls: int = 0

    async def generate(self, *args, **kwargs) -> str:
        _ = args, kwargs
        self.calls += 1
        return f"{self.prefix}-{self.calls}"


class _AlwaysFailEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, *args, **kwargs) -> EvaluationResult:
        _ = args, kwargs
        self.calls += 1
        return EvaluationResult(
            passed=False,
            issues=["requirement coverage too low; missing: python"],
            per_check_scores={"requirement_coverage": 0.25},
        )


def _facts() -> FactsDatabase:
    return FactsDatabase.from_entries(
        [
            FactsEntry(
                id="fact-1",
                category="experience",
                title="Backend Engineer",
                description="Built async ingestion pipeline.",
                technologies=["python", "asyncio"],
            )
        ]
    )


def _job() -> JobAd:
    return JobAd(
        company_name="Acme",
        role_title="Software Engineer",
        required_skills=["python"],
        nice_to_have_skills=[],
        tone_signals="pragmatic",
        key_responsibilities=["ship features"],
        source_text="Role description",
    )


def _style() -> StyleProfile:
    return StyleProfile(
        tone_description="Direct and grounded",
        avg_sentence_length=13.0,
        sentence_length_variance=5.0,
        characteristic_phrases=["shipped"],
        phrases_to_avoid=["leverage"],
        structural_notes="Lead with impact",
        anchor_snippets=["I built it quickly, then refined it in production."],
    )


@pytest.mark.asyncio
async def test_orchestrator_stops_after_max_retries_and_returns_unresolved_issues() -> None:
    cv_generator = _CountingGenerator("cv")
    cover_letter_generator = _CountingGenerator("cover")
    evaluator = _AlwaysFailEvaluator()

    orchestrator = GenerationOrchestrator(
        cv_generator=cv_generator,
        cover_letter_generator=cover_letter_generator,
        evaluator=evaluator,
        max_retries=3,
        enable_humanize_pass=False,
    )

    result = await orchestrator.run(_facts(), _job(), _style())

    assert result.evaluation.passed is False
    assert result.attempts == 3
    assert result.unresolved_issues == ["requirement coverage too low; missing: python"]
    assert cv_generator.calls == 3
    assert cover_letter_generator.calls == 3
    assert evaluator.calls == 3