from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.facts.facts_schema import FactsDatabase, FactsEntry


def test_facts_database_accepts_unique_entries() -> None:
    entry = FactsEntry(
        id="exp-1",
        category="experience",
        title="Internship",
        description="Worked on a real product team.",
        technologies=["Python"],
        start_date=date(2024, 1, 1),
    )

    database = FactsDatabase(entries=[entry])

    assert database.get("exp-1") == entry


def test_facts_database_rejects_duplicate_ids() -> None:
    entry = FactsEntry(
        id="dup-1",
        category="project",
        title="Project A",
        description="First project.",
    )

    with pytest.raises(ValidationError):
        FactsDatabase(entries=[entry, entry])
