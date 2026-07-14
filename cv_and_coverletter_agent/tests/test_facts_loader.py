from __future__ import annotations

from pathlib import Path

from src.facts.facts_loader import load_facts_database, save_facts_database
from src.facts.facts_schema import FactsDatabase, FactsEntry


def test_facts_loader_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "facts.yaml"
    database = FactsDatabase(
        entries=[
            FactsEntry(
                id="skill-1",
                category="skill",
                title="Python",
                description="Used Python in production and personal projects.",
                technologies=["Python"],
            )
        ]
    )

    save_facts_database(database, path)
    loaded = load_facts_database(path)

    assert loaded.entries[0].id == "skill-1"
    assert loaded.source_path == str(path)
