from __future__ import annotations

import pytest

from src.style.style_extractor import StyleExtractor


@pytest.mark.asyncio
async def test_style_extractor_builds_profile_from_samples() -> None:
    extractor = StyleExtractor()
    profile = await extractor.extract(
        [
            "I worked on automation tools. I value clarity and direct communication.\n\nBest regards, Alex.",
            "Thanks for the update. I focused on practical improvements and concise writing.",
        ]
    )

    assert profile.tone_description
    assert profile.avg_sentence_length >= 0
    assert isinstance(profile.characteristic_phrases, list)
    assert isinstance(profile.anchor_snippets, list)


@pytest.mark.asyncio
async def test_style_extractor_handles_short_sample() -> None:
    extractor = StyleExtractor()

    profile = await extractor.extract(["Thanks."])

    assert profile.tone_description
    assert profile.avg_sentence_length >= 0


@pytest.mark.asyncio
async def test_style_extractor_rejects_empty_samples_gracefully() -> None:
    extractor = StyleExtractor()

    with pytest.raises(ValueError, match="at least one non-empty writing sample is required"):
        await extractor.extract(["   ", "\n"])
