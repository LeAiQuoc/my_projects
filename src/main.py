"""Command-line entrypoint for the CV and cover letter agent."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import click
from dotenv import load_dotenv

from src.config import DEFAULT_FACTS_FILE, DEFAULT_STYLE_PROFILE_FILE, get_env_path
from src.evaluation.evaluator import Evaluator
from src.evaluation.humanize_detector import HumanizeDetectorResult, run_humanize_detector
from src.facts.facts_loader import bootstrap_facts_database, load_facts_database
from src.facts.facts_schema import FactsDatabase
from src.generation.cover_letter_generator import CoverLetterGenerator
from src.generation.cv_generator import CVGenerator
from src.job_ads.parser import JobAdParser
from src.job_ads.schema import JobAd
from src.loop.orchestrator import GenerationOrchestrator, OrchestrationResult
from src.pipeline.batch import BatchPipeline, RankedBatchResult
from src.style.style_extractor import StyleExtractor
from src.style.style_profile import StyleProfile

load_dotenv()


def _configure_logging() -> logging.Logger:
    """Configure a simple console logger for CLI commands."""

    logger = logging.getLogger("cv_coverletter_agent")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _load_json_style_profile(path: Path) -> StyleProfile:
    """Load a saved style profile from disk."""

    if not path.exists():
        raise FileNotFoundError(path)
    return StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))


def _save_json_style_profile(profile: StyleProfile, path: Path) -> None:
    """Persist a style profile as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def _read_text_source(source: str) -> tuple[str, str | None]:
    """Read a job-ad source from a file path or HTTP(S) URL.

    Returns the raw text plus an optional source label.
    """

    path = Path(source)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8"), str(path)

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        try:
            with urlopen(source) as response:  # nosec - user-provided URL input
                data = response.read()
        except URLError as exc:
            raise click.ClickException(f"could not fetch job ad URL: {exc}") from exc
        return data.decode("utf-8", errors="replace"), source

    raise click.BadParameter("expected a job-ad file path or an http(s) URL")


def _collect_text_files(directory: Path) -> list[Path]:
    """Collect readable text-like files from a directory."""

    files = [
        item
        for item in sorted(directory.iterdir())
        if item.is_file() and item.suffix.lower() in {".txt", ".md", ".markdown"}
    ]
    return files


def _render_generation_markdown(
    source_label: str,
    job_ad: JobAd,
    result: OrchestrationResult,
    detector_result: HumanizeDetectorResult | None = None,
    rendered_cover_letter: str | None = None,
    detector_executed_at: str | None = None,
) -> str:
    """Render a single generation result as markdown."""

    issues = result.unresolved_issues or result.evaluation.issues
    issues_block = "\n".join(f"- {issue}" for issue in issues) if issues else "- None"

    detector_block = ""
    if detector_result is not None:
        top_issues = detector_result.issues[:5]
        top_issues_block = (
            "\n".join(
                f"- {issue.issue_type} ({issue.severity}): {issue.text}; suggestion: {issue.suggestion}"
                for issue in top_issues
            )
            if top_issues
            else "- None"
        )
        voice_drift = (
            f"{detector_result.voice_drift:.2f}"
            if detector_result.voice_drift is not None
            else "N/A"
        )
        detector_block = (
            "## Humanize Detector\n\n"
            f"- Executed at: {detector_executed_at or 'N/A'}\n"
            f"- Score: {detector_result.score:.1f}\n"
            f"- Label: {detector_result.label}\n"
            f"- Document classification: {detector_result.document_classification}\n"
            f"- Voice drift: {voice_drift}\n\n"
            "- Top issues:\n"
            f"{top_issues_block}\n\n"
        )
    else:
        detector_block = (
            "## Humanize Detector\n\n"
            f"- Executed at: {detector_executed_at or 'N/A'}\n"
            "- Status: unavailable (detector runtime not found or execution failed)\n\n"
        )

    cover_letter_text = rendered_cover_letter or _finalize_cover_letter_for_rendering(result.cover_letter_draft, job_ad)

    return (
        f"# Generated Application Pack\n\n"
        f"- Source: {source_label}\n"
        f"- Company: {job_ad.company_name}\n"
        f"- Role: {job_ad.role_title}\n"
        f"- Attempts: {result.attempts}\n"
        f"- Passed: {result.evaluation.passed}\n\n"
        f"## CV\n\n{result.cv_draft}\n\n"
        f"## Cover Letter\n\n{cover_letter_text}\n\n"
        f"{detector_block}"
        f"## Evaluation Summary\n\n"
        f"- Issues:\n{issues_block}\n"
    )


def _finalize_cover_letter_for_rendering(cover_letter: str, job_ad: JobAd) -> str:
    """Apply final output guardrails for software-role cover letters."""

    role_context = f"{job_ad.role_title} {job_ad.company_name}".lower()
    software_like = any(token in role_context for token in ("software", "mjukvaru", "embedded", "developer"))
    if not software_like:
        return cover_letter

    filtered = [
        line
        for line in cover_letter.splitlines()
        if not any(
            marker in line.lower()
            for marker in ("driver's license", "driving license", "class b", "korkort", "körkort")
        )
    ]
    return "\n".join(filtered).strip()


def _render_batch_markdown(results: list[RankedBatchResult]) -> str:
    """Render ranked batch results as markdown."""

    if not results:
        return "# Ranked Batch Results\n\nNo job ads were found.\n"

    lines = ["# Ranked Batch Results", ""]
    lines.append("| Rank | Company | Role | Fit Score | Passed | Attempts |")
    lines.append("| --- | --- | --- | ---: | --- | ---: |")
    for index, item in enumerate(results, start=1):
        lines.append(
            f"| {index} | {item.job_ad.company_name} | {item.job_ad.role_title} | "
            f"{item.fit_score:.2f} | {item.result.evaluation.passed} | {item.result.attempts} |"
        )

    lines.append("")
    for index, item in enumerate(results, start=1):
        issues = item.result.unresolved_issues or item.result.evaluation.issues
        issues_block = "\n".join(f"- {issue}" for issue in issues) if issues else "- None"
        detector_result = run_humanize_detector(
            item.result.cover_letter_draft,
            scene_mode="public-writing",
            voice_mode="professional",
        )
        detector_lines: list[str] = []
        if detector_result is not None:
            detector_lines.extend(
                [
                    "### Humanize Detector",
                    "",
                    f"- Score: {detector_result.score:.1f}",
                    f"- Label: {detector_result.label}",
                    f"- Document classification: {detector_result.document_classification}",
                    (
                        f"- Voice drift: {detector_result.voice_drift:.2f}"
                        if detector_result.voice_drift is not None
                        else "- Voice drift: N/A"
                    ),
                    "",
                    "- Top issues:",
                ]
            )
            top_issues = detector_result.issues[:5]
            if top_issues:
                detector_lines.extend(
                    [
                        (
                            f"- {issue.issue_type} ({issue.severity}): {issue.text}; "
                            f"suggestion: {issue.suggestion}"
                        )
                        for issue in top_issues
                    ]
                )
            else:
                detector_lines.append("- None")
            detector_lines.append("")

        lines.extend(
            [
                f"## {index}. {item.job_ad.company_name} - {item.job_ad.role_title}",
                f"- Fit score: {item.fit_score:.2f}",
                f"- Passed: {item.result.evaluation.passed}",
                f"- Attempts: {item.result.attempts}",
                "",
                "### CV",
                "",
                item.result.cv_draft,
                "",
                "### Cover Letter",
                "",
                item.result.cover_letter_draft,
                "",
                *detector_lines,
                "### Issues",
                "",
                issues_block,
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _render_style_profile_markdown(profile: StyleProfile) -> str:
    """Render a saved style profile as markdown for review."""

    anchor_block = "\n\n".join(f"- {snippet}" for snippet in profile.anchor_snippets) or "- None"
    return (
        "# Refreshed Style Profile\n\n"
        f"- Tone: {profile.tone_description}\n"
        f"- Avg sentence length: {profile.avg_sentence_length}\n"
        f"- Sentence variance: {profile.sentence_length_variance}\n"
        f"- Characteristic phrases: {', '.join(profile.characteristic_phrases) or 'N/A'}\n"
        f"- Phrases to avoid: {', '.join(profile.phrases_to_avoid) or 'N/A'}\n"
        f"- Structural notes: {profile.structural_notes}\n\n"
        f"## Anchor Snippets\n\n{anchor_block}\n"
    )


def _create_orchestrator(logger: logging.Logger | None = None) -> GenerationOrchestrator:
    """Build the default generation loop from the current environment."""

    return GenerationOrchestrator(
        cv_generator=CVGenerator(),
        cover_letter_generator=CoverLetterGenerator(),
        evaluator=Evaluator(),
        logger=logger,
    )


async def _run_generate(
    job_ad_source: str,
    facts_file: Path,
    style_file: Path,
    logger: logging.Logger,
) -> tuple[JobAd, OrchestrationResult]:
    """Execute the generate command asynchronously."""

    logger.info("Loading facts database from %s", facts_file)
    facts = load_facts_database(facts_file)

    logger.info("Loading style profile from %s", style_file)
    style_profile = _load_json_style_profile(style_file)

    logger.info("Reading job ad source")
    raw_job_ad, source_label = _read_text_source(job_ad_source)

    logger.info("Parsing job ad")
    job_ad = await JobAdParser().parse(raw_job_ad, source_url=source_label)

    logger.info("Running generation loop")
    orchestrator = _create_orchestrator(logger)
    result = await orchestrator.run(facts, job_ad, style_profile)
    return job_ad, result


async def _run_batch(
    job_ads_dir: Path,
    facts_file: Path,
    style_file: Path,
    logger: logging.Logger,
) -> list[RankedBatchResult]:
    """Execute the batch command asynchronously."""

    logger.info("Loading facts database from %s", facts_file)
    facts = load_facts_database(facts_file)

    logger.info("Loading style profile from %s", style_file)
    style_profile = _load_json_style_profile(style_file)

    job_ad_files = _collect_text_files(job_ads_dir)
    if not job_ad_files:
        raise click.ClickException(f"no text job-ad files found in {job_ads_dir}")

    parser = JobAdParser()
    job_ads = []
    for job_ad_file in job_ad_files:
        logger.info("Parsing job ad file %s", job_ad_file.name)
        raw_text = job_ad_file.read_text(encoding="utf-8")
        job_ads.append(await parser.parse(raw_text, source_url=str(job_ad_file)))

    logger.info("Running batch pipeline for %s job ads", len(job_ads))
    orchestrator = _create_orchestrator(logger)
    pipeline = BatchPipeline(orchestrator=orchestrator, max_concurrency=3)
    return await pipeline.run(facts, job_ads, style_profile)


async def _run_refresh_style(samples_dir: Path, output_path: Path, logger: logging.Logger) -> StyleProfile:
    """Rebuild the style profile from a directory of writing samples."""

    sample_files = _collect_text_files(samples_dir)
    if not sample_files:
        raise click.ClickException(f"no text samples found in {samples_dir}")

    samples = []
    for sample_file in sample_files:
        logger.info("Reading sample %s", sample_file.name)
        samples.append(sample_file.read_text(encoding="utf-8"))

    logger.info("Extracting style profile")
    profile = await StyleExtractor(profile_path=output_path).extract(samples)
    _save_json_style_profile(profile, output_path)
    return profile


def _echo_or_write(markdown: str, output: Path | None) -> None:
    """Print markdown to stdout or write it to a file."""

    if output is None:
        click.echo(markdown, nl=False)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    click.echo(f"Wrote output to {output}")


@click.group()
def cli() -> None:
    """Root CLI group."""

    _configure_logging()


@cli.command(name="init-facts")
@click.option(
    "--facts-file",
    type=click.Path(path_type=Path),
    default=lambda: get_env_path("FACTS_FILE", DEFAULT_FACTS_FILE),
)
@click.option("--overwrite", is_flag=True, help="Replace an existing facts file if one is already present.")
def init_facts(facts_file: Path, overwrite: bool) -> None:
    """Create the first editable facts file from the sample template."""

    try:
        created_path = bootstrap_facts_database(facts_file, overwrite=overwrite)
    except FileExistsError as exc:
        raise click.ClickException(
            f"facts file already exists at {exc}. Use --overwrite to replace it."
        ) from exc
    click.echo(f"Created starter facts file at {created_path}")


@cli.command(name="generate")
@click.argument("job_ad_source", type=str)
@click.option(
    "--facts-file",
    type=click.Path(path_type=Path),
    default=lambda: get_env_path("FACTS_FILE", DEFAULT_FACTS_FILE),
    show_default=True,
)
@click.option(
    "--style-file",
    type=click.Path(path_type=Path),
    default=lambda: get_env_path("STYLE_PROFILE_FILE", DEFAULT_STYLE_PROFILE_FILE),
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write the markdown output to a file.")
def generate(job_ad_source: str, facts_file: Path, style_file: Path, output: Path | None) -> None:
    """Generate a CV and cover letter from a job ad source."""

    logger = logging.getLogger("cv_coverletter_agent")
    try:
        job_ad, result = asyncio.run(_run_generate(job_ad_source, facts_file, style_file, logger))
    except (click.ClickException, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    rendered_cover_letter = _finalize_cover_letter_for_rendering(result.cover_letter_draft, job_ad)
    detector_executed_at = datetime.now().isoformat(timespec="seconds")
    detector_feedback = run_humanize_detector(
        rendered_cover_letter,
        scene_mode="public-writing",
        voice_mode="professional",
    )
    markdown = _render_generation_markdown(
        job_ad_source,
        job_ad,
        result,
        detector_result=detector_feedback,
        rendered_cover_letter=rendered_cover_letter,
        detector_executed_at=detector_executed_at,
    )
    _echo_or_write(markdown, output)


@cli.command(name="batch")
@click.argument("job_ads_dir", type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--facts-file",
    type=click.Path(path_type=Path),
    default=lambda: get_env_path("FACTS_FILE", DEFAULT_FACTS_FILE),
    show_default=True,
)
@click.option(
    "--style-file",
    type=click.Path(path_type=Path),
    default=lambda: get_env_path("STYLE_PROFILE_FILE", DEFAULT_STYLE_PROFILE_FILE),
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write the markdown output to a file.")
def batch(job_ads_dir: Path, facts_file: Path, style_file: Path, output: Path | None) -> None:
    """Run the full loop across a directory of job ad files."""

    logger = logging.getLogger("cv_coverletter_agent")
    try:
        results = asyncio.run(_run_batch(job_ads_dir, facts_file, style_file, logger))
    except (click.ClickException, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    markdown = _render_batch_markdown(results)
    _echo_or_write(markdown, output)


@cli.command(name="refresh-style")
@click.argument("samples_dir", type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=lambda: get_env_path("STYLE_PROFILE_FILE", DEFAULT_STYLE_PROFILE_FILE),
    show_default=True,
)
def refresh_style(samples_dir: Path, output: Path) -> None:
    """Re-run style extraction on a directory of writing samples."""

    logger = logging.getLogger("cv_coverletter_agent")
    try:
        profile = asyncio.run(_run_refresh_style(samples_dir, output, logger))
    except (click.ClickException, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_render_style_profile_markdown(profile))
    click.echo(f"Saved style profile to {output}")


if __name__ == "__main__":
    cli()