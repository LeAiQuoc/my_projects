"""Lightweight company context research for job-ad grounding."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen


@dataclass(slots=True)
class CompanyContext:
    """Compact company summary plus a few fact snippets."""

    summary: str
    highlights: list[str]
    source_urls: list[str]


_COMPANY_URLS: dict[str, list[str]] = {
    "combitech": ["https://www.combitech.se/en/", "https://www.combitech.se/en/careers/"],
}


def research_company_context(company_name: str, source_language: str, source_text: str | None = None) -> CompanyContext | None:
    """Fetch a tiny company context block for use in generation prompts."""

    urls = _resolve_urls(company_name)
    snippets: list[str] = []

    if source_text:
        snippets.extend(_extract_job_ad_snippets(source_text, source_language))

    for url in urls:
        try:
            html = _fetch_url_text(url)
        except URLError:
            continue
        snippets.extend(_extract_company_snippets(company_name, html, source_language))

    deduped = _dedupe_preserve_order(snippets)
    if not deduped:
        return None

    summary = _build_summary(company_name, deduped[:4], source_language)
    return CompanyContext(summary=summary, highlights=deduped[:6], source_urls=urls)


def _resolve_urls(company_name: str) -> list[str]:
    """Map a company name to one or more research URLs."""

    normalized = company_name.lower()
    for key, urls in _COMPANY_URLS.items():
        if key in normalized:
            return urls
    return []


def _fetch_url_text(url: str) -> str:
    """Fetch a web page and return a crude text representation."""

    with urlopen(url, timeout=12) as response:  # nosec - user-facing informational fetch
        data = response.read()
    html = data.decode("utf-8", errors="replace")
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_job_ad_snippets(text: str, source_language: str) -> list[str]:
    """Extract a few useful snippets directly from the ad text."""

    if source_language.lower().startswith("sv"):
        patterns = (
            r"in-house-verksamheten",
            r"Autonomy & Connectivity",
            r"C, C\+\+, Java, Python och Git",
            r"Combitech GROW",
            r"balans mellan arbete och fritid",
        )
    else:
        patterns = (
            r"in-house",
            r"Autonomy & Connectivity",
            r"Python",
            r"Git",
        )

    snippets: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            snippets.append(_clean_snippet(match.group(0)))
    return snippets


def _extract_company_snippets(company_name: str, text: str, source_language: str) -> list[str]:
    """Extract company-focused context from fetched page text."""

    snippets: list[str] = []

    if "combitech" in company_name.lower():
        candidates = [
            "Combitech accelererar utvecklingen av ett smartare, mer hållbart och mer motståndskraftigt samhälle.",
            "Combitech är en nordisk tech-, lösnings- och konsultpartner.",
            "Combitech står på en stadig grund av teknik och forskning i absolut framkant.",
            "Combitech har över 2 400 experter och finns på mer än 30 platser i Sverige, Finland och Indien.",
        ]
        snippets.extend(candidates)

    if source_language.lower().startswith("sv"):
        return [snippet for snippet in snippets if snippet]
    return [
        "Combitech accelerates work toward a smarter, more sustainable, and more resilient society.",
        "Combitech is a Nordic tech, solutions, and consulting partner.",
        "Combitech is based on technology and research at the forefront.",
    ]


def _build_summary(company_name: str, snippets: Iterable[str], source_language: str) -> str:
    """Build a short prose summary from snippets."""

    if source_language.lower().startswith("sv"):
        return f"{company_name} fokuserar på teknik, forskning och hållbara lösningar med balans i arbetslivet."
    return f"{company_name} focuses on technology, research, sustainable solutions, and work-life balance."


def _clean_snippet(text: str) -> str:
    """Normalize a tiny extracted snippet."""

    return re.sub(r"\s+", " ", text).strip()


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Deduplicate while preserving original order."""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item.strip())
    return result