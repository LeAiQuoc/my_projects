"""Load and save the verified facts database."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from .facts_schema import FactsDatabase

def bootstrap_facts_database(destination: str | Path, overwrite: bool = False) -> Path:
    """Create a starter facts file from the bundled sample template.

    This gives the project a safe first-run path without requiring the user to
    handcraft YAML from scratch.
    """

    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(destination_path)

    template_path = Path(__file__).with_name("sample_facts.yaml")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, destination_path)
    return destination_path


def load_facts_database(path: str | Path) -> FactsDatabase:
    """Load a facts database from a YAML file on disk."""

    facts_path = Path(path)
    data = yaml.safe_load(facts_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("facts database YAML must contain a mapping at the top level")
    database = FactsDatabase.model_validate(data)
    database.source_path = str(facts_path)
    return database


def save_facts_database(database: FactsDatabase, path: str | Path) -> None:
    """Persist a facts database as YAML."""

    facts_path = Path(path)
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    payload = database.model_dump(mode="json", exclude_none=True)
    facts_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class FactsRepository:
    """Small convenience wrapper around the YAML-backed facts file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> FactsDatabase:
        """Load the database from the configured path."""

        return load_facts_database(self.path)

    def save(self, database: FactsDatabase) -> None:
        """Save the database back to the configured path."""

        save_facts_database(database, self.path)

    def as_mapping(self) -> dict[str, Any]:
        """Return the raw YAML payload for debugging or templating."""

        return self.load().model_dump(mode="json", exclude_none=True)
