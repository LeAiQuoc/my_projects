"""Pydantic schema for a learned writing style profile."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class StyleProfile(BaseModel):
    """Compact summary of how the user typically writes."""

    tone_description: str
    avg_sentence_length: float
    sentence_length_variance: float
    characteristic_phrases: list[str] = Field(default_factory=list)
    phrases_to_avoid: list[str] = Field(default_factory=list)
    structural_notes: str
    anchor_snippets: list[str] = Field(default_factory=list)

    @field_validator("tone_description", "structural_notes")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        """Keep text fields trimmed and non-empty."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("anchor_snippets")
    @classmethod
    def _validate_anchor_snippets(cls, value: list[str]) -> list[str]:
        """Normalize anchor snippets and keep only non-empty paragraphs."""

        cleaned = [snippet.strip() for snippet in value if snippet.strip()]
        return cleaned
