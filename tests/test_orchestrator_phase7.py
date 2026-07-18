from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.evaluation.evaluator import EvaluationResult
from src.facts.facts_schema import FactsDatabase, FactsEntry
from src.job_ads.schema import JobAd
from src.loop.orchestrator import GenerationOrchestrator
from src.style.style_profile import StyleProfile


@dataclass
class _StaticGenerator:
    text: str

    async def generate(self, *args, **kwargs):
        _ = args, kwargs
        return self.text


class _EvaluatorSequence:
    def __init__(self, results: list[EvaluationResult]) -> None:
        self._results = results
        self.calls = 0

    async def evaluate(self, *args, **kwargs) -> EvaluationResult:
        _ = args, kwargs
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


def _facts() -> FactsDatabase:
    return FactsDatabase.from_entries(
        [
            FactsEntry(
                id="fact-1",
                category="experience",
                title="Backend Engineer",
                description="Built async ingestion pipeline.",
                technologies=["python", "asyncio"],
            )
        ]
    )


def _job() -> JobAd:
    return JobAd(
        company_name="Acme",
        role_title="Software Engineer",
        required_skills=["python"],
        nice_to_have_skills=[],
        tone_signals="pragmatic",
        key_responsibilities=["ship features"],
        source_text="Role description",
    )


def _style() -> StyleProfile:
    return StyleProfile(
        tone_description="Direct and grounded",
        avg_sentence_length=13.0,
        sentence_length_variance=5.0,
        characteristic_phrases=["shipped"],
        phrases_to_avoid=["leverage"],
        structural_notes="Lead with impact",
        anchor_snippets=[
            "I built it quickly, then refined it in production.",
            "Results came from small iterations and clear tradeoffs.",
            "The outcome was measurable and stable.",
        ],
    )


@pytest.mark.asyncio
async def test_orchestrator_runs_humanize_when_ai_tone_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def _fake_rewrite(draft: str, style_profile: StyleProfile, *args, **kwargs) -> str:
        _ = style_profile, args, kwargs
        calls["count"] += 1
        return f"humanized::{draft}"

    monkeypatch.setattr("src.loop.orchestrator.rewrite_for_natural_rhythm", _fake_rewrite)

    evaluator = _EvaluatorSequence(
        [
            EvaluationResult(
                passed=False,
                issues=["ai-tone check failed: sentence rhythm is too uniform"],
                per_check_scores={"ai_tone": 0.4},
            ),
            EvaluationResult(
                passed=True,
                issues=[],
                per_check_scores={"ai_tone": 0.9},
            ),
        ]
    )

    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv draft"),
        cover_letter_generator=_StaticGenerator("cover draft"),
        evaluator=evaluator,
        max_retries=2,
        enable_humanize_pass=True,
    )

    result = await orchestrator.run(_facts(), _job(), _style())

    assert result.evaluation.passed is True
    assert calls["count"] == 2


def test_orchestrator_builds_actionable_template_structure_correction_note() -> None:
    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv draft"),
        cover_letter_generator=_StaticGenerator("cover draft"),
        evaluator=_EvaluatorSequence([]),
    )

    note = orchestrator._build_correction_note(
        ["template structure risk: stock phrases found (aligns with, my technical foundation)"]
    )

    assert note is not None
    assert "Avoid stock fit phrases" in note


@pytest.mark.asyncio
async def test_orchestrator_skips_humanize_without_ai_tone_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def _fake_rewrite(draft: str, style_profile: StyleProfile, *args, **kwargs) -> str:
        _ = draft, style_profile, args, kwargs
        calls["count"] += 1
        return "should-not-run"

    monkeypatch.setattr("src.loop.orchestrator.rewrite_for_natural_rhythm", _fake_rewrite)

    evaluator = _EvaluatorSequence(
        [
            EvaluationResult(
                passed=False,
                issues=["coverage check failed: missing requirement mapping"],
                per_check_scores={"coverage": 0.4},
            ),
            EvaluationResult(
                passed=True,
                issues=[],
                per_check_scores={"coverage": 0.9},
            ),
        ]
    )

    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv draft"),
        cover_letter_generator=_StaticGenerator("cover draft"),
        evaluator=evaluator,
        max_retries=2,
        enable_humanize_pass=True,
    )

    result = await orchestrator.run(_facts(), _job(), _style())

    assert result.evaluation.passed is True
    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_orchestrator_runs_humanize_when_style_mismatch_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def _fake_rewrite(draft: str, style_profile: StyleProfile, *args, **kwargs) -> str:
        _ = style_profile, args, kwargs
        calls["count"] += 1
        return f"humanized::{draft}"

    monkeypatch.setattr("src.loop.orchestrator.rewrite_for_natural_rhythm", _fake_rewrite)

    evaluator = _EvaluatorSequence(
        [
            EvaluationResult(
                passed=False,
                issues=["style mismatch: sentence-length profile differs from style profile"],
                per_check_scores={"style_match": 0.3},
            ),
            EvaluationResult(
                passed=True,
                issues=[],
                per_check_scores={"style_match": 0.9},
            ),
        ]
    )

    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv draft"),
        cover_letter_generator=_StaticGenerator("cover draft"),
        evaluator=evaluator,
        max_retries=2,
        enable_humanize_pass=True,
    )

    result = await orchestrator.run(_facts(), _job(), _style())

    assert result.evaluation.passed is True
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_orchestrator_blocks_humanize_when_protected_span_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_rewrite(draft: str, style_profile: StyleProfile, *args, **kwargs) -> str:
        _ = style_profile, args, kwargs
        return "humanized text without protected keyword"

    monkeypatch.setattr("src.loop.orchestrator.rewrite_for_natural_rhythm", _fake_rewrite)

    evaluator = _EvaluatorSequence(
        [
            EvaluationResult(
                passed=False,
                issues=["ai-tone check failed: sentence rhythm is too uniform"],
                per_check_scores={"ai_tone": 0.4},
            ),
            EvaluationResult(
                passed=True,
                issues=[],
                per_check_scores={"ai_tone": 0.9},
            ),
        ]
    )

    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv draft with python"),
        cover_letter_generator=_StaticGenerator("cover draft with python"),
        evaluator=evaluator,
        max_retries=1,
        enable_humanize_pass=True,
    )

    result = await orchestrator.run(_facts(), _job(), _style())

    assert result.evaluation.passed is True
    assert result.cv_draft == "cv draft with python"
    assert result.cover_letter_draft == "cover draft with python"


@pytest.mark.asyncio
async def test_orchestrator_routes_voice_mode_by_document_type(monkeypatch: pytest.MonkeyPatch) -> None:
    voice_modes: list[str] = []

    async def _fake_rewrite(draft: str, style_profile: StyleProfile, *args, **kwargs) -> str:
        _ = style_profile, args
        voice_modes.append(str(kwargs.get("voice_mode")))
        return f"humanized {draft} with python"

    monkeypatch.setattr("src.loop.orchestrator.rewrite_for_natural_rhythm", _fake_rewrite)

    evaluator = _EvaluatorSequence(
        [
            EvaluationResult(
                passed=False,
                issues=["style mismatch: sentence-length profile differs from style profile"],
                per_check_scores={"style_match": 0.3},
            ),
            EvaluationResult(
                passed=True,
                issues=[],
                per_check_scores={"style_match": 0.9},
            ),
        ]
    )


    @pytest.mark.asyncio
    async def test_orchestrator_dedupes_repeated_cover_letter_sentences() -> None:
        evaluator = _EvaluatorSequence(
            [
                EvaluationResult(
                    passed=True,
                    issues=[],
                    per_check_scores={"coverage": 1.0},
                )
            ]
        )

        duplicated_cover = (
            "Jag arbetar praktiskt med Python och Git. "
            "Jag arbetar praktiskt med Python och Git. "
            "Jag bidrar med tydlig kommunikation i team."
        )

        orchestrator = GenerationOrchestrator(
            cv_generator=_StaticGenerator("cv draft"),
            cover_letter_generator=_StaticGenerator(duplicated_cover),
            evaluator=evaluator,
            max_retries=1,
            enable_humanize_pass=False,
        )

        result = await orchestrator.run(_facts(), _job(), _style())

        assert result.cover_letter_draft.count("Jag arbetar praktiskt med Python och Git") == 1
    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv draft with python"),
        cover_letter_generator=_StaticGenerator("cover draft with python"),
        evaluator=evaluator,
        max_retries=1,
        enable_humanize_pass=True,
    )

    result = await orchestrator.run(_facts(), _job(), _style())

    assert result.evaluation.passed is True
    assert voice_modes == ["technical", "professional"]


@pytest.mark.asyncio
async def test_orchestrator_allows_rewrite_when_only_role_label_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_rewrite(draft: str, style_profile: StyleProfile, *args, **kwargs) -> str:
        _ = style_profile, args, kwargs
        return draft.replace("Software Engineer", "Developer")

    monkeypatch.setattr("src.loop.orchestrator.rewrite_for_natural_rhythm", _fake_rewrite)

    evaluator = _EvaluatorSequence(
        [
            EvaluationResult(
                passed=False,
                issues=["style mismatch: sentence-length profile differs from style profile"],
                per_check_scores={"style_match": 0.3},
            ),
            EvaluationResult(
                passed=True,
                issues=[],
                per_check_scores={"style_match": 0.9},
            ),
        ]
    )

    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv draft with Software Engineer and python"),
        cover_letter_generator=_StaticGenerator("cover draft with Software Engineer and python"),
        evaluator=evaluator,
        max_retries=1,
        enable_humanize_pass=True,
    )

    result = await orchestrator.run(_facts(), _job(), _style())

    assert result.evaluation.passed is True
    assert "Developer" in result.cv_draft


@pytest.mark.asyncio
async def test_orchestrator_voice_mode_can_be_overridden_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    voice_modes: list[str] = []

    async def _fake_rewrite(draft: str, style_profile: StyleProfile, *args, **kwargs) -> str:
        _ = style_profile, args
        voice_modes.append(str(kwargs.get("voice_mode")))
        return f"humanized {draft} with python"

    monkeypatch.setattr("src.loop.orchestrator.rewrite_for_natural_rhythm", _fake_rewrite)
    monkeypatch.setenv("HUMANIZE_VOICE_MODE_CV", "blunt")
    monkeypatch.setenv("HUMANIZE_VOICE_MODE_COVER", "warm")

    evaluator = _EvaluatorSequence(
        [
            EvaluationResult(
                passed=False,
                issues=["style mismatch: sentence-length profile differs from style profile"],
                per_check_scores={"style_match": 0.3},
            ),
            EvaluationResult(
                passed=True,
                issues=[],
                per_check_scores={"style_match": 0.9},
            ),
        ]
    )

    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv draft with python"),
        cover_letter_generator=_StaticGenerator("cover draft with python"),
        evaluator=evaluator,
        max_retries=1,
        enable_humanize_pass=True,
    )

    result = await orchestrator.run(_facts(), _job(), _style())

    assert result.evaluation.passed is True
    assert voice_modes == ["blunt", "warm"]


def test_orchestrator_correction_note_sanitizes_requirement_coverage_issue() -> None:
    orchestrator = GenerationOrchestrator(
        cv_generator=_StaticGenerator("cv"),
        cover_letter_generator=_StaticGenerator("cover"),
        evaluator=_EvaluatorSequence(
            [
                EvaluationResult(
                    passed=True,
                    issues=[],
                    per_check_scores={},
                )
            ]
        ),
    )

    note = orchestrator._build_correction_note(
        [
            "requirement coverage too low; missing: matlab/simulink, autosar, vectors, c/c++",
            "unsupported claim: I know Python and C/C++ already.",
        ]
    )

    assert note is not None
    assert "Do not fabricate missing required skills" in note
    assert "Remove unsupported claims" in note
    assert "c/c++" not in note.lower()