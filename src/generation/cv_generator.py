"""CV generation template."""

from __future__ import annotations

from typing import Any

from src.facts.facts_schema import FactsDatabase
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


class CVGenerator:
    """Generate a tailored CV section set from the verified facts database."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    async def generate(
        self,
        facts: FactsDatabase,
        job_ad: JobAd,
        style_profile: StyleProfile,
        correction_note: str | None = None,
    ) -> str:
        """Generate a draft CV body for the target job ad."""

        _ = facts, job_ad, style_profile, correction_note
        raise NotImplementedError
