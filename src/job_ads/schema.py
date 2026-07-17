"""Schema for structured job ad parsing output."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class JobAd(BaseModel):
    """Structured representation of a job ad or posting."""

    company_name: str
    role_title: str
    source_language: str = "en"
    company_context: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    tone_signals: str
    key_responsibilities: list[str] = Field(default_factory=list)
    source_text: str | None = None
    source_url: str | None = None

    @field_validator("company_name", "role_title", "tone_signals", "source_language")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Normalize required text fields and reject blank values."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("source_language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        """Normalize language codes to the supported prompt languages."""

        cleaned = value.strip().lower()
        if cleaned in {"sv", "sv-se", "swedish"}:
            return "sv"
        return "en"
