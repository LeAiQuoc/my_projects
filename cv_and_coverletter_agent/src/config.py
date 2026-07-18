"""Shared configuration helpers for file locations and environment defaults."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTS_FILE = PROJECT_ROOT / "data" / "facts.yaml"
DEFAULT_STYLE_PROFILE_FILE = PROJECT_ROOT / "data" / "style_profile.json"
DEFAULT_JOB_ADS_DIR = PROJECT_ROOT / "job_ads"


def get_env_path(name: str, default: Path) -> Path:
    """Read a path from the environment and fall back to a project default."""

    value = os.getenv(name)
    return Path(value) if value else default