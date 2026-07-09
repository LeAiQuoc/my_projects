from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.editing.cutter import ClipCutError, cut_clips
from src.models.schemas import HighlightCandidate


class _FakeProcess:
    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", self._stderr.encode("utf-8")


@pytest.mark.asyncio
async def test_cut_clips_creates_clip_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake")

    async def fake_create_subprocess_exec(*args, **_kwargs):
        output_path = Path(args[-1])
        output_path.write_bytes(b"clip")
        return _FakeProcess(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    highlights = [
        HighlightCandidate(
            start_seconds=1.0,
            end_seconds=4.0,
            hook_title="h",
            rationale="r",
            confidence=0.8,
        )
    ]

    clips = await cut_clips(source_video=source_video, highlights=highlights, output_dir=tmp_path / "out")

    assert len(clips) == 1
    assert clips[0].output_path.exists()
    assert clips[0].output_path.name.startswith("clip_01_")


@pytest.mark.asyncio
async def test_cut_clips_raises_for_missing_source(tmp_path: Path) -> None:
    highlights = [
        HighlightCandidate(
            start_seconds=1.0,
            end_seconds=4.0,
            hook_title="h",
            rationale="r",
            confidence=0.8,
        )
    ]

    with pytest.raises(ClipCutError):
        await cut_clips(source_video=tmp_path / "missing.mp4", highlights=highlights, output_dir=tmp_path / "out")


@pytest.mark.asyncio
async def test_cut_clips_raises_when_ffmpeg_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake")

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return _FakeProcess(returncode=1, stderr="ffmpeg mock failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    highlights = [
        HighlightCandidate(
            start_seconds=1.0,
            end_seconds=4.0,
            hook_title="h",
            rationale="r",
            confidence=0.8,
        )
    ]

    with pytest.raises(ClipCutError):
        await cut_clips(source_video=source_video, highlights=highlights, output_dir=tmp_path / "out")


@pytest.mark.asyncio
async def test_cut_clips_rejects_invalid_window(tmp_path: Path) -> None:
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake")

    highlights = [
        HighlightCandidate(
            start_seconds=4.0,
            end_seconds=4.0,
            hook_title="h",
            rationale="r",
            confidence=0.8,
        )
    ]

    with pytest.raises(ClipCutError):
        await cut_clips(source_video=source_video, highlights=highlights, output_dir=tmp_path / "out")
