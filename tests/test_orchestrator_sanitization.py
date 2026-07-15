from __future__ import annotations

import pytest

from src.evaluation.evaluator import EvaluationResult
from src.facts.facts_schema import FactsDatabase
from src.job_ads.schema import JobAd
from src.loop.orchestrator import GenerationOrchestrator
from src.style.style_profile import StyleProfile


class _FakeCVGenerator:
    async def generate(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return "We leverage data and delve into results."


class _FakeCoverLetterGenerator:
    async def generate(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return "Furthermore, this is a bespoke approach."


class _FakeEvaluator:
    def __init__(self) -> None:
        self.last_draft: str | None = None

    async def evaluate(self, draft: str, *args, **kwargs) -> EvaluationResult:  # noqa: ANN002, ANN003
        self.last_draft = draft
        return EvaluationResult(passed=True, issues=[], per_check_scores={"overall": 1.0})


@pytest.mark.asyncio
async def test_orchestrator_sanitizes_before_evaluation() -> None:
    evaluator = _FakeEvaluator()
    orchestrator = GenerationOrchestrator(
        cv_generator=_FakeCVGenerator(),
        cover_letter_generator=_FakeCoverLetterGenerator(),
        evaluator=evaluator,
        max_retries=1,
    )

    facts = FactsDatabase(entries=[])
    job_ad = JobAd(
        company_name="Example Co",
        role_title="Engineer",
        tone_signals="neutral professional",
    )
    style_profile = StyleProfile(
        tone_description="direct",
        avg_sentence_length=12.0,
        sentence_length_variance=2.0,
        structural_notes="short paragraphs",
    )

    result = await orchestrator.run(facts, job_ad, style_profile)

    assert evaluator.last_draft is not None
    assert "leverage" not in evaluator.last_draft.lower()
    assert "delve" not in evaluator.last_draft.lower()
    assert "furthermore" not in evaluator.last_draft.lower()
    assert "bespoke" not in evaluator.last_draft.lower()
    assert result.evaluation.passed is True
