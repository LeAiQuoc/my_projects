from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.job_ads.batch import parse_job_ads_batch
from src.job_ads.parser import JobAdParser
from src.job_ads.schema import JobAd


def test_job_ad_model_builds() -> None:
    job_ad = JobAd(
        company_name="Example Co",
        role_title="Software Engineer",
        required_skills=["Python"],
        tone_signals="formal",
        key_responsibilities=["Build product features"],
    )

    assert job_ad.company_name == "Example Co"


@pytest.mark.asyncio
async def test_parse_job_ads_batch_returns_empty_list_for_no_input() -> None:
    parser = JobAdParser()
    assert await parse_job_ads_batch(parser, []) == []


def test_job_ad_model_rejects_blank_required_text() -> None:
    with pytest.raises(ValidationError):
        JobAd(
            company_name=" ",
            role_title="Engineer",
            tone_signals="formal",
        )
