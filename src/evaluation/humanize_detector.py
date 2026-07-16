"""Run the humanize-text-skill detector engine from Python."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any


@dataclass(slots=True)
class DetectorIssue:
    """Compact issue shape returned by the detector."""

    issue_type: str
    text: str
    severity: str
    suggestion: str


@dataclass(slots=True)
class HumanizeDetectorResult:
    """Structured feedback from detector/patterns.js."""

    score: float
    label: str
    document_classification: str
    voice_drift: float | None
    issues: list[DetectorIssue] = field(default_factory=list)


def run_humanize_detector(
    text: str,
    scene_mode: str = "public-writing",
    voice_mode: str = "professional",
) -> HumanizeDetectorResult | None:
    """Execute the humanize-text-skill detector and parse feedback.

    Returns None if detector files or Node are unavailable.
    """

    stripped = text.strip()
    if not stripped:
        return None

    repo_root = Path(__file__).resolve().parents[2]
    detector_path = repo_root / "humanize-text-skill-main" / "detector" / "patterns.js"
    if not detector_path.exists():
        return None

    node_script = (
        "const fs=require('node:fs');"
        "const detector=require(process.argv[1]);"
        "const scene=process.argv[2]||'public-writing';"
        "const voice=process.argv[3]||'professional';"
        "const input=fs.readFileSync(0,'utf8');"
        "const result=detector.analyzeText(input,{contextMode:'general',sceneMode:scene,voiceMode:voice});"
        "process.stdout.write(JSON.stringify(result));"
    )

    try:
        completed = subprocess.run(
            ["node", "-e", node_script, str(detector_path), scene_mode, voice_mode],
            input=stripped,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=True,
            timeout=8,
            cwd=str(repo_root),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    payload = _safe_json_loads(completed.stdout or "")
    if payload is None:
        return None

    issues = [_parse_issue(item) for item in payload.get("issues", [])]
    voice_payload = payload.get("voice")
    drift = None
    if isinstance(voice_payload, dict):
        raw_drift = voice_payload.get("drift")
        if isinstance(raw_drift, (int, float)):
            drift = float(raw_drift)

    score = payload.get("score", 0.0)
    if not isinstance(score, (int, float)):
        score = 0.0

    return HumanizeDetectorResult(
        score=float(score),
        label=str(payload.get("label", "Unknown")),
        document_classification=str(payload.get("document_classification", "UNSCORED")),
        voice_drift=drift,
        issues=issues,
    )


def _safe_json_loads(raw: str) -> dict[str, Any] | None:
    """Parse JSON safely with a dict-only contract."""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_issue(item: Any) -> DetectorIssue:
    """Normalize one detector issue entry."""

    if not isinstance(item, dict):
        return DetectorIssue(issue_type="unknown", text="", severity="unknown", suggestion="")

    return DetectorIssue(
        issue_type=str(item.get("type", "unknown")),
        text=str(item.get("text", "")),
        severity=str(item.get("severity", "unknown")),
        suggestion=str(item.get("suggestion", "")),
    )
