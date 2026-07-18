from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation.humanize_pass import rewrite_for_natural_rhythm
from src.style.style_profile import StyleProfile


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create(self, **kwargs):
        _ = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content),
                )
            ]
        )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def _style_profile() -> StyleProfile:
    return StyleProfile(
        tone_description="Direct, pragmatic, grounded.",
        avg_sentence_length=14.2,
        sentence_length_variance=6.1,
        characteristic_phrases=["shipped", "iterated", "measured"],
        phrases_to_avoid=["I am excited to", "leverage"],
        structural_notes="Open with a concrete impact statement.",
        anchor_snippets=[
            "I built the first version quickly, then cut half the assumptions after real feedback.",
            "The next iteration was slower but cleaner; it held up in production.",
            "I keep explanations short when the data is clear.",
        ],
    )


@pytest.mark.asyncio
async def test_rewrite_for_natural_rhythm_uses_client_response() -> None:
    style = _style_profile()
    client = _FakeClient("Rewritten draft with varied rhythm.")

    rewritten = await rewrite_for_natural_rhythm(
        draft="Original draft text.",
        style_profile=style,
        client=client,
        model="deepseek-chat",
    )

    assert rewritten == "Rewritten draft with varied rhythm."


@pytest.mark.asyncio
async def test_rewrite_for_natural_rhythm_rejects_empty_draft() -> None:
    style = _style_profile()
    client = _FakeClient("Ignored")

    with pytest.raises(ValueError, match="draft must not be empty"):
        await rewrite_for_natural_rhythm(
            draft="   ",
            style_profile=style,
            client=client,
        )


@pytest.mark.asyncio
async def test_rewrite_for_natural_rhythm_normalizes_typographic_dashes() -> None:
    style = _style_profile()
    client = _FakeClient("Line one — line two – line three")

    rewritten = await rewrite_for_natural_rhythm(
        draft="Original draft text.",
        style_profile=style,
        client=client,
        model="deepseek-chat",
    )

    assert "—" not in rewritten
    assert "–" not in rewritten
    assert "Line one - line two - line three" == rewritten


@pytest.mark.asyncio
async def test_rewrite_for_natural_rhythm_splits_very_long_sentence() -> None:
    style = _style_profile()
    long_line = (
        "I built a practical tool during internship and delivered it in production, and "
        "I documented progress clearly for technical stakeholders while keeping the pipeline stable and "
        "maintaining deployment checks across several iterative releases."
    )
    client = _FakeClient(long_line)

    rewritten = await rewrite_for_natural_rhythm(
        draft="Original draft text.",
        style_profile=style,
        client=client,
        model="deepseek-chat",
    )

    assert rewritten.count(".") >= 2


@pytest.mark.asyncio
async def test_rewrite_for_natural_rhythm_removes_scaffolding_phrases() -> None:
    style = _style_profile()
    client = _FakeClient(
        "I am writing to express my interest in the role. "
        "At the end of the day, this is not only practical but also reliable. "
        "Moreover, in conclusion, it is important to note that the result is strong."
    )

    rewritten = await rewrite_for_natural_rhythm(
        draft="Original draft text.",
        style_profile=style,
        client=client,
        model="deepseek-chat",
    )

    assert "I am writing to express my interest in" not in rewritten
    assert "At the end of the day" not in rewritten
    assert "not only" not in rewritten.lower()
    assert "Moreover" not in rewritten
    assert "In conclusion" not in rewritten
    assert "important to note" not in rewritten.lower()


@pytest.mark.asyncio
async def test_rewrite_for_natural_rhythm_keeps_basic_and_phrase_intact() -> None:
    style = _style_profile()
    client = _FakeClient(
        "My background in AI and software development includes practical project delivery, "
        "clear status communication, and iterative implementation with measurable progress across releases."
    )

    rewritten = await rewrite_for_natural_rhythm(
        draft="Original draft text.",
        style_profile=style,
        client=client,
        model="deepseek-chat",
    )

    assert "AI and software development" in rewritten


@pytest.mark.asyncio
async def test_rewrite_for_natural_rhythm_preserves_paragraph_breaks() -> None:
    style = _style_profile()
    client = _FakeClient(
        "Dear Hiring Team,\n\n"
        "I built practical tooling during internship.\n\n"
        "I also handled technical support in prior roles."
    )

    rewritten = await rewrite_for_natural_rhythm(
        draft="Original draft text.",
        style_profile=style,
        client=client,
        model="deepseek-chat",
    )

    assert "\n\n" in rewritten