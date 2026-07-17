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


@pytest.mark.asyncio
async def test_evaluator_fails_ai_placeholder_fingerprint() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.95}'))

    draft = "Dear [Your Name], I built Python services and SQL pipelines."
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.passed is False
    assert result.per_check_scores["fingerprint_check"] < 1.0
    assert any("fingerprint check fail" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_evaluator_passes_fingerprint_check_for_clean_text() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.95}'))

    draft = "I built Python ETL tooling and validated SQL data quality checks in production."
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.per_check_scores["fingerprint_check"] == 1.0


@pytest.mark.asyncio
async def test_evaluator_flags_uniform_sentence_rhythm() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.95}'))

    draft = (
        "I build Python services every day. "
        "I write SQL checks for data quality. "
        "I ship backend tools for production teams. "
        "I document updates for team members."
    )
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.passed is False
    assert result.per_check_scores["rhythm_uniformity"] < 1.0
    assert any("rhythm uniformity risk" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_evaluator_filters_fact_supported_summary_claims() -> None:
    facts = FactsDatabase(
        entries=[
            FactsEntry(
                id="proj-1",
                category="project",
                title="Home Automation and DIY Systems",
                description="Built a Python and AutoHotkey automation system to control an AV receiver and configured Raspberry Pi 5 running Home Assistant.",
                technologies=["Python", "AutoHotkey", "Raspberry Pi 5", "Home Assistant"],
            ),
            FactsEntry(
                id="skill-1",
                category="skill",
                title="Personal Programming Practice",
                description="I also spend time programming in my spare time, building various programs.",
                technologies=["Python", "Personal Projects"],
            ),
        ]
    )
    evaluator = Evaluator(
        client=_FakeClient(
            '{"unsupported_claims": ['
            '"I also spend time programming in my spare time, building various programs.", '
            '"My technical projects, including a Python-based AV receiver automation system and an AI clip highlighter, demonstrate consistent independent programming work."], '
            '"confidence": 0.91}'
        )
    )

    draft = (
        "I also spend time programming in my spare time, building various programs. "
        "My technical projects, including a Python-based AV receiver automation system and an AI clip highlighter, demonstrate consistent independent programming work."
    )
    result = await evaluator.evaluate(draft, facts, _sample_job_ad(), _sample_style())

    assert not any(issue.startswith("unsupported claim: I also spend time programming") for issue in result.issues)
    assert not any("independent programming work" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_evaluator_flags_stock_cover_letter_phrases() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.95}'))

    draft = (
        "Dear Hiring Team,\n\n"
        "My technical foundation in Python and backend engineering aligns with your role.\n\n"
        "Relevant tooling in my experience includes RAG, MCP."
    )
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.passed is False
    assert result.per_check_scores["cover_letter_structure"] < 1.0
    assert any("template structure risk: stock phrases found" in issue for issue in result.issues)

@pytest.mark.asyncio
async def test_evaluator_flags_ai_phrase_tells() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.95}'))

    draft = (
        "Dear Hiring Team,\n\n"
        "I am excited to leverage my skills in Python and SQL. "
        "My technical foundation in backend engineering aligns with the role. "
        "Relevant tooling in my experience includes RAG and MCP. "
        "I look forward to the opportunity to discuss further."
    )
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.passed is False
    assert result.per_check_scores["ai_tone"] < 1.0
    assert result.per_check_scores["cover_letter_structure"] < 1.0
    assert any("generic phrases found" in issue for issue in result.issues)
    assert any("template structure risk: stock phrases found" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_evaluator_flags_even_cover_letter_paragraph_lengths() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.95}'))

    draft = (
        "Dear Hiring Team,\n\n"
        "I built Python tools during internship and documented progress for the team every week.\n\n"
        "I handled technical support tasks at work and solved practical issues under time pressure daily.\n\n"
        "I also built personal automation projects and kept improving them through steady iteration at home."
    )
    result = await evaluator.evaluate(draft, _sample_facts(), _sample_job_ad(), _sample_style())

    assert result.passed is False
    assert result.per_check_scores["cover_letter_structure"] < 1.0
    assert any("template structure risk: cover-letter paragraphs are too even in length" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_evaluator_flags_swedish_language_mismatch_and_short_letter() -> None:
    evaluator = Evaluator(client=_FakeClient('{"unsupported_claims": [], "confidence": 0.95}'))
    job_ad = JobAd(
        company_name="Combitech",
        role_title="mjukvaruutvecklare inom Autonomy & Connectivity",
        source_language="sv",
        required_skills=["Python"],
        nice_to_have_skills=[],
        tone_signals="professionell",
        key_responsibilities=["in-house-verksamheten"],
    )

    draft = (
        "Hej rekryteringsteamet,\n\n"
        "I am applying for the role at Combitech. My background fits well.\n\n"
        "I built a RAG pipeline during my internship.\n\n"
        "Best regards"
    )
    result = await evaluator.evaluate(draft, _sample_facts(), job_ad, _sample_style())

    assert result.passed is False
    assert result.per_check_scores["language_match"] < 1.0
    assert result.per_check_scores["cover_letter_length"] < 1.0
    assert any("language mismatch" in issue for issue in result.issues)
    assert any("cover letter too short" in issue for issue in result.issues)
