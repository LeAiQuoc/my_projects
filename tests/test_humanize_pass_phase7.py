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