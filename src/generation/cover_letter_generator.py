"""Cover letter generation template."""

from __future__ import annotations

from typing import Any

from src.facts.facts_schema import FactsDatabase
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


class CoverLetterGenerator:
    """Generate a cover letter grounded strictly in verified facts."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    async def generate(
        self,
        facts: FactsDatabase,
        job_ad: JobAd,
        style_profile: StyleProfile,
        correction_note: str | None = None,
    ) -> str:
        """Generate a draft cover letter from grounded inputs.

        The correction note is appended on retries when the evaluator finds issues.
        """

        _ = facts, job_ad, style_profile, correction_note
    raise NotImplementedError
