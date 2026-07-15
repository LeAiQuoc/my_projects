"""Interactive helper for building a valid facts database file."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.facts.facts_loader import load_facts_database, save_facts_database
from src.facts.facts_schema import FactsCategory, FactsDatabase, FactsEntry


_CATEGORIES: tuple[FactsCategory, ...] = (
    "experience",
    "project",
    "skill",
    "education",
    "certification",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive helper for adding and validating facts entries.",
    )
    parser.add_argument(
        "command",
        choices=("add", "list", "validate"),
        help="Action to perform on the facts database.",
    )
    parser.add_argument(
        "--facts-file",
        default="data/facts.yaml",
        help="Path to the YAML facts file. Defaults to data/facts.yaml.",
    )
    args = parser.parse_args()

    facts_path = Path(args.facts_file)
    database = _load_or_create_database(facts_path)

    if args.command == "add":
        _add_fact_interactively(database, facts_path)
        return
    if args.command == "list":
        _list_facts(database, facts_path)
        return
    _validate_facts(database, facts_path)


def _load_or_create_database(path: Path) -> FactsDatabase:
    if path.exists():
        return load_facts_database(path)
    return FactsDatabase(entries=[], source_path=str(path))


def _add_fact_interactively(database: FactsDatabase, facts_path: Path) -> None:
    print(f"Using facts file: {facts_path}")
    print("Leave optional fields blank if you do not want to include them yet.")
    print()

    category = _prompt_category()
    title = _prompt_required("Title", "Short label, for example 'Plejd Internship'")
    suggested_id = _slugify(f"{category}-{title}")
    entry_id = _prompt_optional(
        "ID",
        f"Unique identifier. Press Enter to use '{suggested_id}'",
    ) or suggested_id
    description = _prompt_required(
        "Description",
        "One factual sentence or two about what you did or what this proves",
    )
    technologies = _prompt_csv("Technologies", "Comma-separated, for example Python, SQL, Docker")
    evidence_url = _prompt_optional("Evidence URL", "Optional GitHub/LinkedIn/portfolio link") or None
    start_date = _prompt_date("Start date", "Optional, format YYYY-MM-DD")
    end_date = _prompt_date("End date", "Optional, format YYYY-MM-DD")

    new_entry = FactsEntry(
        id=entry_id,
        category=category,
        title=title,
        description=description,
        technologies=technologies,
        evidence_url=evidence_url,
        start_date=start_date,
        end_date=end_date,
    )

    updated_entries = list(database.entries)
    updated_entries.append(new_entry)
    updated_database = FactsDatabase.from_entries(updated_entries, source_path=str(facts_path))
    save_facts_database(updated_database, facts_path)

    print()
    print(f"Saved fact '{new_entry.id}' to {facts_path}")
    print(f"The facts file now contains {len(updated_database.entries)} entr{'y' if len(updated_database.entries) == 1 else 'ies'}.")


def _list_facts(database: FactsDatabase, facts_path: Path) -> None:
    print(f"Facts file: {facts_path}")
    if not database.entries:
        print("No entries yet. Run: .\\scripts\\facts.ps1 add")
        return

    for index, entry in enumerate(database.entries, start=1):
        technologies = ", ".join(entry.technologies) if entry.technologies else "none"
        print(f"{index}. [{entry.category}] {entry.id} - {entry.title}")
        print(f"   {entry.description}")
        print(f"   technologies: {technologies}")


def _validate_facts(database: FactsDatabase, facts_path: Path) -> None:
    print(f"Facts file is valid: {facts_path}")
    print(f"Entries: {len(database.entries)}")


def _prompt_category() -> FactsCategory:
    print("Choose a category:")
    for index, category in enumerate(_CATEGORIES, start=1):
        print(f"  {index}. {category}")

    while True:
        raw_value = input("Category number: ").strip()
        if raw_value.isdigit():
            selection = int(raw_value)
            if 1 <= selection <= len(_CATEGORIES):
                return _CATEGORIES[selection - 1]
        print("Please enter one of the numbers shown above.")


def _prompt_required(label: str, help_text: str) -> str:
    print(f"{label}: {help_text}")
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print(f"{label} is required.")


def _prompt_optional(label: str, help_text: str) -> str:
    print(f"{label}: {help_text}")
    return input(f"{label}: ").strip()


def _prompt_csv(label: str, help_text: str) -> list[str]:
    raw_value = _prompt_optional(label, help_text)
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _prompt_date(label: str, help_text: str) -> date | None:
    while True:
        raw_value = _prompt_optional(label, help_text)
        if not raw_value:
            return None
        try:
            return date.fromisoformat(raw_value)
        except ValueError:
            print("Please use YYYY-MM-DD or leave it blank.")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "fact-entry"


if __name__ == "__main__":
    main()