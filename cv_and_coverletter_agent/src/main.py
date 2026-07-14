"""Command-line entrypoint for the CV and cover letter agent."""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

from src.config import DEFAULT_FACTS_FILE, get_env_path
from src.facts.facts_loader import bootstrap_facts_database

load_dotenv()


@click.group()
def cli() -> None:
    """Root CLI group."""
    return None


@cli.command(name="init-facts")
@click.option("--facts-file", type=click.Path(path_type=Path), default=lambda: get_env_path("FACTS_FILE", DEFAULT_FACTS_FILE))
@click.option("--overwrite", is_flag=True, help="Replace an existing facts file if one is already present.")
def init_facts(facts_file: Path, overwrite: bool) -> None:
    """Create the first editable facts file from the sample template."""

    created_path = bootstrap_facts_database(facts_file, overwrite=overwrite)
    click.echo(f"Created starter facts file at {created_path}")


if __name__ == "__main__":
    cli()
