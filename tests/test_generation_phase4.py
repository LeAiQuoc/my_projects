from __future__ import annotations

import pytest

from src.facts.facts_schema import FactsDatabase, FactsEntry
from src.generation.cover_letter_generator import CoverLetterGenerator
from src.generation.cv_generator import CVGenerator
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


class _FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> _FakeResponse:
        self.last_kwargs = kwargs
        return _FakeResponse(self.content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions(content)})()


def _sample_facts() -> FactsDatabase:
    return FactsDatabase(
        entries=[
            FactsEntry(
                id="exp-1",
                category="experience",
                title="Data Engineer Intern",
                description="Built Python ETL pipelines and SQL quality checks.",
                technologies=["Python", "SQL"],
            ),
            FactsEntry(
                id="proj-1",
                category="project",
                title="Automation Platform",
                description="Created automation scripts and observability checks.",
                technologies=["Python", "Docker"],
            ),
            FactsEntry(
                id="exp-2",
                category="experience",
                title="Operations Support",
                description="Worked in customer service and coordinated with teammates under pressure.",
                technologies=["Customer Service", "Teamwork"],
            ),
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
        sentence_length_variance=3.0,
        characteristic_phrases=["I focused on"],
        phrases_to_avoid=["I am passionate about"],
        structural_notes="brief intro, evidence-heavy body, concise close",
        anchor_snippets=["I prioritize clear outcomes and measurable impact."],
    )


@pytest.mark.asyncio
async def test_cv_generator_builds_grounded_prompt_and_returns_text() -> None:
    fake_client = _FakeClient("CV DRAFT")
    generator = CVGenerator(client=fake_client)

    output = await generator.generate(
        facts=_sample_facts(),
        job_ad=_sample_job_ad(),
        style_profile=_sample_style(),
        correction_note="remove unsupported metrics",
    )

    assert output == "CV DRAFT"

    kwargs = fake_client.chat.completions.last_kwargs
    assert kwargs is not None
    messages = kwargs["messages"]
    assert isinstance(messages, list)
    user_prompt = messages[1]["content"]
    assert isinstance(user_prompt, str)
    assert "Use ONLY the provided facts entries" in user_prompt
    assert "Do NOT invent skills" in user_prompt
    assert "remove unsupported metrics" in user_prompt


@pytest.mark.asyncio
async def test_cover_letter_generator_builds_grounded_prompt_and_returns_text() -> None:
    fake_client = _FakeClient("COVER LETTER DRAFT")
    generator = CoverLetterGenerator(client=fake_client)

    output = await generator.generate(
        facts=_sample_facts(),
        job_ad=_sample_job_ad(),
        style_profile=_sample_style(),
    )

    assert output == "COVER LETTER DRAFT"

    kwargs = fake_client.chat.completions.last_kwargs
    assert kwargs is not None
    messages = kwargs["messages"]
    assert isinstance(messages, list)
    user_prompt = messages[1]["content"]
    assert isinstance(user_prompt, str)
    assert "Use ONLY the selected facts entries" in user_prompt
    assert "Do NOT invent skills" in user_prompt
    assert "Include at least one short soft-skill paragraph grounded in selected facts" in user_prompt
    assert "Avoid binary contrast templates" in user_prompt
    assert "Exactly 4 paragraphs plus greeting line" in user_prompt
    assert "Greeting line must be exactly: Dear Hiring Team," in user_prompt
    assert "non-technical work fact" in user_prompt
    assert "Soft-skill facts to use explicitly" in user_prompt
    assert "Technical differentiators available in selected facts" in user_prompt
    assert "id=exp-2" in user_prompt


@pytest.mark.asyncio
async def test_cover_letter_generator_injects_missing_differentiators() -> None:
    facts = FactsDatabase(
        entries=[
            FactsEntry(
                id="exp-diff-1",
                category="experience",
                title="AI Engineer",
                description="Worked on practical LLM workflows.",
                technologies=["Python", "RAG", "AI API integration", "MCP"],
            ),
            FactsEntry(
                id="exp-diff-2",
                category="experience",
                title="Ops Support",
                description="Handled customer service under pressure.",
                technologies=["Customer Service", "Teamwork"],
            ),
        ]
    )

    fake_client = _FakeClient("Dear Hiring Team,\n\nI build practical tools in Python.")
    generator = CoverLetterGenerator(client=fake_client)

    output = await generator.generate(
        facts=facts,
        job_ad=_sample_job_ad(),
        style_profile=_sample_style(),
    )

    assert "Relevant tooling in my experience includes" in output
    assert "RAG" in output or "AI API integration" in output or "MCP" in output
