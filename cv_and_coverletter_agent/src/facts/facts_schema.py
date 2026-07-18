"""Pydantic models for the user's verified facts database."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FactsCategory = Literal["experience", "project", "skill", "education", "certification"]


class FactsEntry(BaseModel):
    """Single verified fact that can be used in generated documents."""

    id: str
    category: FactsCategory
    title: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    evidence_url: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("id", "title", "description")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Normalize text fields so downstream matching is consistent."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("technologies")
    @classmethod
    def _strip_technologies(cls, value: list[str]) -> list[str]:
        """Normalize technology names and drop empty items."""

        cleaned = [technology.strip() for technology in value if technology.strip()]
        return cleaned


class FactsDatabase(BaseModel):
    """Collection of verified facts loaded from a local YAML file."""

    entries: list[FactsEntry] = Field(default_factory=list)
    source_path: str | None = None

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> FactsDatabase:
        """Ensure every entry id appears exactly once in the database."""

        seen: set[str] = set()
        duplicates: list[str] = []
        for entry in self.entries:
            if entry.id in seen:
                duplicates.append(entry.id)
                continue
            seen.add(entry.id)
        if duplicates:
            raise ValueError(f"duplicate facts ids found: {sorted(set(duplicates))}")
        return self

    @classmethod
    def from_entries(cls, entries: list[FactsEntry], source_path: str | None = None) -> FactsDatabase:
        """Create a database from in-memory entries.

        This keeps the model construction path explicit in tests and future loaders.
        """

        return cls(entries=entries, source_path=source_path)

    def get(self, entry_id: str) -> FactsEntry:
        """Return a single fact by id.

        Raises:
            KeyError: If the id is not present in the database.
        """

        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise KeyError(entry_id)

    def by_category(self, category: FactsCategory) -> list[FactsEntry]:
        """Return all entries in a specific category."""

        return [entry for entry in self.entries if entry.category == category]
