from __future__ import annotations

import pytest

from src.evaluation.evaluator import Evaluator
from src.facts.facts_schema import FactsDatabase, FactsEntry
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


class _FakeResponseMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeResponseChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeResponseMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeResponseChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create(self, **kwargs: object) -> _FakeResponse:  # noqa: ANN401
        _ = kwargs
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


def _sample_facts() -> FactsDatabase:
    return FactsDatabase(
        entries=[
            FactsEntry(
                id="exp-1",
                category="experience",
                title="Data Engineer Intern",
                description="Built Python ETL pipelines and SQL quality checks.",
                technologies=["Python", "SQL"],
            )
        ]
    )


def _sample_job_ad() -> JobAd:
    return JobAd(
        company_name="Example Co",
        role_title="Data Engineer",
        required_skills=["Python", "SQL"],
        nice_to_have_skills=["Docker"],
        tone_signals="neutral professional",
        key_responsibilities=["build data pipelines", "ensure data quality"],
    )


def _sample_style() -> StyleProfile:
    return StyleProfile(
        tone_description="direct and professional",
        avg_sentence_length=14.0,
        sentence_length_variance=5.0,
        characteristic_phrases=["I focused on"],
        phrases_to_avoid=["I am passionate about"],
        structural_notes="brief intro, evidence-heavy body, concise close",
        anchor_snippets=["I prioritize clear outcomes and measurable impact."],
    )


@pytest.mark.asyncio
async def test_evaluator_fails_banned_words_filter() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.93}'))

    draft = "I furthermore deliver bespoke systems and pioneer solutions in teams."
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.passed is False
    assert result.per_check_scores["cliche_filter"] == 0.0
    assert any("cliche filter fail" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_evaluator_fails_requirement_coverage() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.93}'))

    draft = "I built backend services with Go and Kubernetes."
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.passed is False
    assert result.per_check_scores["requirement_coverage"] < 0.6
    assert any("requirement coverage too low" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_evaluator_fails_hallucination_from_llm_output() -> None:
    evaluator = Evaluator(
        client=_FakeClient('{"unsupported_claims": ["Led a 20-person org", "Shipped Rust compiler"], "confidence": 0.91}')
    )

    draft = "I led a 20-person org and shipped a Rust compiler."
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.passed is False
    assert result.per_check_scores["hallucination"] < 1.0
    assert any("unsupported claim" in issue for issue in result.issues)
