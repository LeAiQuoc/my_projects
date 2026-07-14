from __future__ import annotations

import pytest

from src.job_ads.parser import JobAdParser


@pytest.mark.asyncio
async def test_job_ad_parser_extracts_basic_fields() -> None:
    parser = JobAdParser()
    job_ad = await parser.parse(
        """Example Corp
Role: Software Engineer
Requirements:
- Python
- SQL
Responsibilities:
- Build and maintain tools
- Collaborate with a small team
"""
    )

    assert job_ad.company_name == "Example Corp"
    assert job_ad.role_title == "Software Engineer"
    assert "Python" in job_ad.required_skills or "SQL" in job_ad.required_skills
    assert job_ad.key_responsibilities
