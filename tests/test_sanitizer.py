from __future__ import annotations

from src.generation.sanitizer import sanitize_draft


def test_sanitize_draft_replaces_forbidden_words_case_insensitively() -> None:
    text = "We Delve into systems and UTILIZE tools to LEVERAGE data. Furthermore, this is a testament."

    sanitized = sanitize_draft(text)

    assert "Delve" not in sanitized
    assert "UTILIZE" not in sanitized
    assert "LEVERAGE" not in sanitized
    assert "Furthermore" not in sanitized
    assert "testament" not in sanitized.lower()
    assert "Explore" in sanitized
    assert "USE" in sanitized
    assert "also" in sanitized.lower()


def test_sanitize_draft_only_replaces_whole_words() -> None:
    text = "The word leveragedly should stay, but leverage should change."

    sanitized = sanitize_draft(text)

    assert "leveragedly" in sanitized
    assert " leverage " not in f" {sanitized.lower()} "
