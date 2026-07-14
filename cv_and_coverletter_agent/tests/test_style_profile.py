from __future__ import annotations

from src.style.style_profile import StyleProfile


def test_style_profile_model_builds() -> None:
    profile = StyleProfile(
        tone_description="Clear and direct",
        avg_sentence_length=17.5,
        sentence_length_variance=4.2,
        characteristic_phrases=["I focused on"],
        phrases_to_avoid=["I am passionate about"],
        structural_notes="Short intro, dense middle, concise ending.",
        anchor_snippets=["I ship practical improvements quickly."],
    )

    assert profile.tone_description == "Clear and direct"
    assert profile.anchor_snippets == ["I ship practical improvements quickly."]
