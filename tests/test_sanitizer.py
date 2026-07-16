from __future__ import annotations

from src.generation.sanitizer import sanitize_draft


def test_sanitize_draft_replaces_forbidden_words_case_insensitively() -> None:
    text = (
        "We Delve into systems and UTILIZE tools to LEVERAGE data. Furthermore, this is a testament. "
        "Moreover, it is important to note that this multifaceted approach will optimize results."
    )

    sanitized = sanitize_draft(text)

    assert "Delve" not in sanitized
    assert "UTILIZE" not in sanitized
    assert "LEVERAGE" not in sanitized
    assert "Furthermore" not in sanitized
    assert "Moreover" not in sanitized
    assert "important to note" not in sanitized.lower()
    assert "testament" not in sanitized.lower()
    assert "multifaceted" not in sanitized.lower()
    assert "optimize" not in sanitized.lower()
    assert "Look into" in sanitized
    assert "USE" in sanitized
    assert "also" in sanitized.lower()
    assert "keep in mind" in sanitized.lower()
    assert "complex" in sanitized.lower()
    assert "improve" in sanitized.lower()


def test_sanitize_draft_only_replaces_whole_words() -> None:
    text = "The word leveragedly should stay, but leverage should change."

    sanitized = sanitize_draft(text)

    assert "leveragedly" in sanitized
    assert " leverage " not in f" {sanitized.lower()} "


def test_sanitize_draft_replaces_common_ai_phrases() -> None:
    text = (
        "My technical foundation in Python and backend engineering aligns with the role. "
        "I am excited to leverage my skills in data engineering, and relevant tooling in my experience includes RAG. "
        "I look forward to the opportunity to discuss further."
    )

    sanitized = sanitize_draft(text)

    assert "my technical foundation" not in sanitized.lower()
    assert "aligns with" not in sanitized.lower()
    assert "excited to leverage" not in sanitized.lower()
    assert "relevant tooling in my experience includes" not in sanitized.lower()
    assert "look forward to the opportunity to discuss further" not in sanitized.lower()
    assert "my background" in sanitized.lower()
    assert "fits" in sanitized.lower()
    assert "i use" in sanitized.lower()
    assert "I've used" in sanitized
    assert "happy to discuss further" in sanitized.lower()


def test_sanitize_draft_rewrites_warehouse_triads() -> None:
    text = "Handling picking, packing, and forklift driving at a fast pace."

    sanitized = sanitize_draft(text)

    assert "picking, packing, and forklift driving" not in sanitized.lower()
    assert "warehouse operations" in sanitized.lower()
    assert "picking/packing work" in sanitized.lower()
    assert "forklift operation" in sanitized.lower()


def test_sanitize_draft_normalizes_dash_characters() -> None:
    text = "Teknikhögskolan — Higher Vocational Education Diploma – AI and Software Development"

    sanitized = sanitize_draft(text)

    assert "—" not in sanitized
    assert "–" not in sanitized
    assert "Teknikhögskolan - Higher Vocational Education Diploma - AI and Software Development" in sanitized
