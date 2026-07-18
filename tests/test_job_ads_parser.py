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
    assert job_ad.source_language == "en"
    assert "Python" in job_ad.required_skills or "SQL" in job_ad.required_skills
    assert job_ad.key_responsibilities


@pytest.mark.asyncio
async def test_job_ad_parser_rejects_empty_input_gracefully() -> None:
    parser = JobAdParser()

    with pytest.raises(ValueError, match="job ad text cannot be empty"):
        await parser.parse("   ")


@pytest.mark.asyncio
async def test_job_ad_parser_handles_swedish_nexer_style_ad() -> None:
    parser = JobAdParser()
    job_ad = await parser.parse(
        """## Om jobbet
Nexer Engineering vaxer i Goteborg!

**Om rollen**
Vi soker dig som har flera ars erfarenhet av att arbeta inom fordons- och/eller forsvarsindustrin som mjukvaruutvecklare.
Vi ser garna att du har anvant nagot av foljande programmeringssprak C/C++, Matlab/Simulink och/eller Python.
Vidare vill vi att du har arbetat med nagot av foljande, Autosar eller nagot Vectors verktyg.

For att soka pa den har annonsen vill vi att du beharskar svenska och engelska flytande i bade tal samt skrift.
"""
    )

    assert job_ad.company_name == "Nexer Engineering"
    assert job_ad.role_title.lower() == "mjukvaruutvecklare"
    assert job_ad.source_language == "sv"
    assert "Python" in job_ad.required_skills
    assert "Autosar" in job_ad.required_skills
    assert "C/C++" in job_ad.required_skills
    assert "Matlab/Simulink" in job_ad.required_skills
    assert "Vectors" in job_ad.required_skills
    assert "c/c" not in [skill.lower() for skill in job_ad.required_skills]
    assert "och/eller" not in [skill.lower() for skill in job_ad.required_skills]
    assert "e-post" not in [skill.lower() for skill in job_ad.required_skills]
    assert "Om" not in job_ad.required_skills


@pytest.mark.asyncio
async def test_job_ad_parser_handles_combitech_style_ad() -> None:
    parser = JobAdParser()
    job_ad = await parser.parse(
        """Om jobbet
Combitech i Göteborg söker fler erfarna teammedlemmar till in-house-verksamheten.

Din roll som mjukvaruutvecklare inom Autonomy & Connectivity

Vi använder språk och verktyg som C, C++, Java, Python och Git.
"""
    )

    assert job_ad.company_name == "Combitech"
    assert "mjukvaruutvecklare" in job_ad.role_title.lower()
    assert job_ad.source_language == "sv"
    assert "Python" in job_ad.required_skills


@pytest.mark.asyncio
async def test_job_ad_parser_falls_back_to_filename_prefix_for_generic_heading() -> None:
    parser = JobAdParser()
    job_ad = await parser.parse(
        """Arbetsbeskrivning

Om Lysio Research

Vi behöver nu utöka vårt team med en utvecklare.
""",
        source_url="job_ads/lysio_job_ad.txt",
    )

    assert job_ad.company_name == "Lysio Research"
