from __future__ import annotations

from src.facts.facts_schema import FactsDatabase, FactsEntry
from src.generation.cv_generator import CVGenerator
from src.job_ads.schema import JobAd


def test_select_relevant_facts_prioritizes_required_skill_matches() -> None:
    facts = FactsDatabase(
        entries=[
            FactsEntry(
                id="f1",
                category="project",
                title="Python ETL Pipeline",
                description="Built ETL jobs in Python and PostgreSQL.",
                technologies=["Python", "PostgreSQL"],
            ),
            FactsEntry(
                id="f2",
                category="project",
                title="Frontend Dashboard",
                description="Built dashboards in React and TypeScript.",
                technologies=["React", "TypeScript"],
            ),
            FactsEntry(
                id="f3",
                category="experience",
                title="Data Engineer Intern",
                description="Worked on SQL tuning and data quality checks.",
                technologies=["SQL"],
            ),
        ]
    )
    job_ad = JobAd(
        company_name="Example Co",
        role_title="Data Engineer",
        required_skills=["Python", "SQL"],
        nice_to_have_skills=["Airflow"],
        tone_signals="neutral professional",
        key_responsibilities=["Build data pipelines", "Improve data quality"],
    )

    selected = CVGenerator().select_relevant_facts(facts, job_ad, limit=2)

    assert len(selected) == 2
    selected_ids = {entry.id for entry in selected}
    assert selected_ids == {"f1", "f3"}


def test_select_relevant_facts_returns_fallback_when_no_match() -> None:
    facts = FactsDatabase(
        entries=[
            FactsEntry(
                id="f1",
                category="project",
                title="Automation Tool",
                description="General scripting project.",
                technologies=["Bash"],
            )
        ]
    )
    job_ad = JobAd(
        company_name="Example Co",
        role_title="ML Engineer",
        required_skills=["PyTorch"],
        nice_to_have_skills=[],
        tone_signals="neutral professional",
        key_responsibilities=[],
    )

    selected = CVGenerator().select_relevant_facts(facts, job_ad, limit=1)

    assert len(selected) == 1
    assert selected[0].id == "f1"
