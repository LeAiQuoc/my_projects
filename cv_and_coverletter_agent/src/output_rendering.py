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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.facts.facts_schema import FactsDatabase
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

    company_slug = company_slugify(company_name)
    prefix_slug = _slugify(prefix) or "output"
    return f"{prefix_slug}_{company_slug}"


def company_slugify(company_name: str) -> str:
    """Return a safe filename slug derived from the company name."""

    return _slugify(company_name) or "company"


def cv_title(language_code: str) -> str:
    """Return the standalone CV title."""

    _ = language_code
    return "CV"


def extract_profile_identity(facts: FactsDatabase) -> dict[str, str]:
    """Read invented contact details from the dedicated profile fact."""

    for entry in facts.entries:
        if entry.id != "profile-identity":
            continue
        identity: dict[str, str] = {}
        for raw_line in entry.description.splitlines():
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            normalized_key = key.strip().lower()
            cleaned_value = value.strip()
            if cleaned_value:
                identity[normalized_key] = cleaned_value
        return identity
    return {}


def write_cv_pdf(
    output_path: Path,
    job_ad: JobAd,
    cv_text: str,
    facts: FactsDatabase,
    title: str | None = None,
) -> None:
    """Write a standalone CV PDF with a normalized contact header."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CvTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111111"),
        spaceAfter=8,
    )
    header_style = ParagraphStyle(
        "CvHeader",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#333333"),
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "CvHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#111111"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "CvBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "CvBullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=0,
    )

    identity = extract_profile_identity(facts)
    contact_parts = [
        identity.get("address", ""),
        identity.get("phone", ""),
        identity.get("email", ""),
        identity.get("linkedin", ""),
    ]
    contact_line = " | ".join(part for part in contact_parts if part)

    story: list[object] = [Paragraph(escape(title or cv_title(job_ad.source_language)), title_style)]
    if identity.get("name"):
        story.append(Paragraph(f"<b>{escape(identity['name'])}</b>", heading_style))
    if contact_line:
        story.append(Paragraph(escape(contact_line), header_style))
    story.append(Spacer(1, 4))

    for kind, text in _normalize_cv_blocks(cv_text):
        if kind == "heading":
            story.append(Paragraph(escape(text), heading_style))
        elif kind == "bullet":
            story.append(Paragraph(f"- {escape(text)}", bullet_style))
        else:
            story.append(Paragraph(_format_pdf_paragraph(text), body_style))

    document.build(story)


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


def _normalize_cv_blocks(cv_text: str) -> list[tuple[str, str]]:
    """Strip wrapper metadata and placeholders from a generated CV draft."""

    lines = [line.rstrip() for line in cv_text.splitlines()]
    normalized: list[tuple[str, str]] = []
    skip_prefixes = (
        "- source:",
        "- company:",
        "- role:",
        "[adress]",
        "[telefon]",
        "[e-post]",
        "[linkedin]",
    )

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if line in {"# Generated CV", "# CV", "# CV - [Namn]", "# CV – [Namn]", "# [Namn]"}:
            continue
        if any(lower.startswith(prefix) for prefix in skip_prefixes):
            continue
        if lower.startswith("## "):
            normalized.append(("heading", line[3:].strip()))
            continue
        if lower.startswith("---"):
            continue
        if line.startswith("- "):
            normalized.append(("bullet", line[2:].strip()))
            continue
        normalized.append(("paragraph", line))

    return normalized


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs while preserving intentional spacing."""

    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]


def _format_pdf_paragraph(text: str) -> str:
    """Convert plain text into reportlab-friendly paragraph markup."""

    return escape(text).replace("\n", "<br/>")


def _slugify(value: str) -> str:
    """Build a safe filename stem from arbitrary text."""

    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()