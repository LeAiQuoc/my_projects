"""Draft evaluation for hallucination, coverage, style, and AI tone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.facts.facts_schema import FactsDatabase
from src.job_ads.schema import JobAd
from src.style.style_profile import StyleProfile


class EvaluationResult(BaseModel):
    """Structured result returned by the evaluator."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    per_check_scores: dict[str, float] = Field(default_factory=dict)

    @field_validator("issues")
    @classmethod
    def _trim_issues(cls, value: list[str]) -> list[str]:
        """Keep evaluator issue text compact and readable."""

        return [issue.strip() for issue in value if issue.strip()]


@dataclass(slots=True)
class Evaluator:
    """Coordinate the checks that decide whether a draft is ready to ship."""

    client: Any | None = None

    async def evaluate(
        self,
        draft: str,
        facts: FactsDatabase,
        job_ad: JobAd,
        style_profile: StyleProfile,
    ) -> EvaluationResult:
        """Run the evaluation checks over a draft.

        The scaffold keeps the control surface explicit while the Deepseek-backed
        hallucination and tone checks are implemented later.
        """

        _ = draft, facts, job_ad, style_profile
        raise NotImplementedError
