from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from src.evaluation.evaluator import EvaluationResult
from src.facts.facts_schema import FactsDatabase, FactsEntry
from src.job_ads.schema import JobAd
from src.loop.orchestrator import OrchestrationResult
from src.output_rendering import cover_letter_title
from src.main import cli, _read_text_source
from src.pipeline.batch import RankedBatchResult
from src.style.style_profile import StyleProfile


def _job_ad() -> JobAd:
    return JobAd(
        company_name="Acme",
        role_title="Software Engineer",
        source_language="en",
        required_skills=["python"],
        nice_to_have_skills=[],
        tone_signals="pragmatic",
        key_responsibilities=["ship features"],
        source_text="Role description",
    )


def _style_profile() -> StyleProfile:
    return StyleProfile(
        tone_description="direct and grounded",
        avg_sentence_length=12.0,
        sentence_length_variance=4.0,
        characteristic_phrases=["shipped"],
        phrases_to_avoid=["leverage"],
        structural_notes="brief and evidence-heavy",
        anchor_snippets=["I built it quickly, then refined it in production."],
    )


def _facts_db() -> FactsDatabase:
    return FactsDatabase.from_entries(
        [
            FactsEntry(
                id="fact-1",
                category="experience",
                title="Backend Engineer",
                description="Built async ingestion pipeline.",
                technologies=["python"],
            )
        ]
    )


def _orchestration_result() -> OrchestrationResult:
    return OrchestrationResult(
        cv_draft="# CV\n- Built async ingestion pipeline.",
        cover_letter_draft="# Cover Letter\nI built it quickly.",
        evaluation=EvaluationResult(passed=True, issues=[], per_check_scores={"ai_tone": 1.0}),
        attempts=1,
        unresolved_issues=[],
    )


def test_read_text_source_reads_file(tmp_path: Path) -> None:
    source = tmp_path / "job-ad.txt"
    source.write_text("Company: Acme\nRole: Engineer", encoding="utf-8")

    text, label = _read_text_source(str(source))

    assert "Company: Acme" in text
    assert label == str(source)


def test_generate_command_outputs_markdown(tmp_path: Path, monkeypatch) -> None:
    job_ad_file = tmp_path / "job-ad.txt"
    job_ad_file.write_text("Company: Acme\nRole: Engineer", encoding="utf-8")

    async def _fake_run_generate(job_ad_source: str, facts_file: Path, style_file: Path, logger):
        _ = facts_file, style_file, logger
        assert job_ad_source == str(job_ad_file)
        return _job_ad(), _orchestration_result()

    monkeypatch.setattr("src.main._run_generate", _fake_run_generate)
    monkeypatch.setattr("src.main.run_humanize_detector", lambda *_args, **_kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            str(job_ad_file),
            "--facts-file",
            str(tmp_path / "facts.yaml"),
            "--style-file",
            str(tmp_path / "style.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "# Generated Application Pack" in result.output
    assert "## CV" in result.output
    assert "## Cover Letter" in result.output


def test_generate_cv_command_writes_to_outputs_cv(tmp_path: Path, monkeypatch) -> None:
    job_ad_file = tmp_path / "job-ad.txt"
    job_ad_file.write_text("Company: Acme\nRole: Engineer", encoding="utf-8")

    async def _fake_run_generate_cv(job_ad_source: str, facts_file: Path, style_file: Path, logger):
        _ = facts_file, style_file, logger
        assert job_ad_source == str(job_ad_file)
        return _job_ad(), "job-ad.txt", "# CV\n- Built async ingestion pipeline."

    monkeypatch.setattr("src.main._run_generate_cv", _fake_run_generate_cv)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "generate-cv",
                str(job_ad_file),
                "--facts-file",
                str(tmp_path / "facts.yaml"),
                "--style-file",
                str(tmp_path / "style.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        output_file = Path("outputs_cv") / "cv_acme.md"
        assert output_file.exists()
        assert "# Generated CV" in output_file.read_text(encoding="utf-8")


def test_generate_cl_command_writes_pdf_to_outputs_cl(tmp_path: Path, monkeypatch) -> None:
    job_ad_file = tmp_path / "job-ad.txt"
    job_ad_file.write_text("Company: Acme\nRole: Engineer", encoding="utf-8")

    async def _fake_run_generate_cover_letter(job_ad_source: str, facts_file: Path, style_file: Path, logger):
        _ = facts_file, style_file, logger
        assert job_ad_source == str(job_ad_file)
        return (
            _job_ad(),
            "job-ad.txt",
            "Hej rekryteringsteamet.\n\nMed vänliga hälsningar, John",
            EvaluationResult(passed=True, issues=[], per_check_scores={"ai_tone": 1.0}),
            None,
            "2026-07-18T09:15:27",
        )

    monkeypatch.setattr("src.main._run_generate_cover_letter", _fake_run_generate_cover_letter)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "generate-cl",
                str(job_ad_file),
                "--facts-file",
                str(tmp_path / "facts.yaml"),
                "--style-file",
                str(tmp_path / "style.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        output_file = Path("outputs_cl") / "cover_letter_acme.pdf"
        review_file = Path("outputs_cl") / "acme_scoring_and_review.md"
        assert output_file.exists()
        assert review_file.exists()
        assert output_file.read_bytes().startswith(b"%PDF")
        assert "## Evaluation Summary" in review_file.read_text(encoding="utf-8")


def test_cover_letter_title_is_language_aware() -> None:
    assert cover_letter_title("sv") == "Personligt brev"
    assert cover_letter_title("sv-SE") == "Personligt brev"
    assert cover_letter_title("en") == "Cover letter"


def test_generate_command_renders_detector_block(tmp_path: Path, monkeypatch) -> None:
    job_ad_file = tmp_path / "job-ad.txt"
    job_ad_file.write_text("Company: Acme\nRole: Engineer", encoding="utf-8")

    async def _fake_run_generate(job_ad_source: str, facts_file: Path, style_file: Path, logger):
        _ = facts_file, style_file, logger
        assert job_ad_source == str(job_ad_file)
        return _job_ad(), _orchestration_result()

    monkeypatch.setattr("src.main._run_generate", _fake_run_generate)

    class _Issue:
        issue_type = "template-phrase"
        severity = "high"
        text = "At the end of the day"
        suggestion = "Delete phrase"

    class _DetectorResult:
        score = 42.0
        label = "Some"
        document_classification = "MIXED"
        voice_drift = 18.5
        issues = [_Issue()]

    monkeypatch.setattr("src.main.run_humanize_detector", lambda *_args, **_kwargs: _DetectorResult())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            str(job_ad_file),
            "--facts-file",
            str(tmp_path / "facts.yaml"),
            "--style-file",
            str(tmp_path / "style.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "## Humanize Detector" in result.output
    assert "- Executed at:" in result.output
    assert "- Score: 42.0" in result.output
    assert "- Label: Some" in result.output


def test_init_facts_command_reports_existing_file_cleanly(tmp_path: Path) -> None:
    facts_file = tmp_path / "facts.yaml"
    facts_file.write_text("entries: []\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "init-facts",
            "--facts-file",
            str(facts_file),
        ],
    )

    assert result.exit_code != 0
    assert "facts file already exists" in result.output
    assert "--overwrite" in result.output


def test_batch_command_outputs_ranked_markdown(tmp_path: Path, monkeypatch) -> None:
    job_ads_dir = tmp_path / "job-ads"
    job_ads_dir.mkdir()

    async def _fake_run_batch(job_ads_dir: Path, facts_file: Path, style_file: Path, logger):
        _ = job_ads_dir, facts_file, style_file, logger
        return [
            RankedBatchResult(
                job_ad=_job_ad(),
                result=_orchestration_result(),
                fit_score=3.0,
            )
        ]

    monkeypatch.setattr("src.main._run_batch", _fake_run_batch)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "batch",
            str(job_ads_dir),
            "--facts-file",
            str(tmp_path / "facts.yaml"),
            "--style-file",
            str(tmp_path / "style.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "# Ranked Batch Results" in result.output
    assert "| 1 | Acme | Software Engineer | 3.00 | True | 1 |" in result.output
    assert "## 1. Acme - Software Engineer" in result.output


def test_refresh_style_command_writes_profile(tmp_path: Path, monkeypatch) -> None:
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    output_path = tmp_path / "style.json"

    async def _fake_run_refresh_style(samples_dir: Path, output_path: Path, logger):
        _ = samples_dir, output_path, logger
        return _style_profile()

    monkeypatch.setattr("src.main._run_refresh_style", _fake_run_refresh_style)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "refresh-style",
            str(samples_dir),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "# Refreshed Style Profile" in result.output
    assert f"Saved style profile to {output_path}" in result.output