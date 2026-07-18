"""Helpers for rendering generated CV and cover-letter outputs."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate

from src.job_ads.schema import JobAd


def default_output_stem(source: str) -> str:
    """Derive a stable filename stem from a job-ad source string."""

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        candidate = Path(parsed.path).stem or parsed.netloc
    else:
        candidate = Path(source).stem or source
    return _slugify(candidate) or "job_ad"


def company_output_stem(prefix: str, company_name: str) -> str:
    """Build a default filename stem from the company name."""

    company_slug = _slugify(company_name) or "company"
    prefix_slug = _slugify(prefix) or "output"
    return f"{prefix_slug}_{company_slug}"


def render_cv_markdown(source_label: str, job_ad: JobAd, cv_text: str) -> str:
    """Render a standalone CV markdown document."""

    body = cv_text.strip()
    return (
        "# Generated CV\n\n"
        f"- Source: {source_label}\n"
        f"- Company: {job_ad.company_name}\n"
        f"- Role: {job_ad.role_title}\n\n"
        f"{body}\n"
    )


def cover_letter_title(language_code: str) -> str:
    """Return the PDF cover-letter title in the target language."""

    if language_code.lower().startswith("sv"):
        return "Personligt brev"
    return "Cover letter"


def write_cover_letter_pdf(
    output_path: Path,
    source_label: str,
    job_ad: JobAd,
    cover_letter_text: str,
    title: str | None = None,
) -> None:
    """Write a simple structured PDF cover letter."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CoverLetterTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#111111"),
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "CoverLetterBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        spaceAfter=8,
    )

    story: list[object] = [
        Paragraph(escape(title or cover_letter_title(job_ad.source_language)), title_style),
    ]

    for paragraph in _split_paragraphs(cover_letter_text):
        story.append(Paragraph(_format_pdf_paragraph(paragraph), body_style))

    document.build(story)


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs while preserving intentional spacing."""

    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]


def _format_pdf_paragraph(text: str) -> str:
    """Convert plain text into reportlab-friendly paragraph markup."""

    return escape(text).replace("\n", "<br/>")


def _slugify(value: str) -> str:
    """Build a safe filename stem from arbitrary text."""

    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()