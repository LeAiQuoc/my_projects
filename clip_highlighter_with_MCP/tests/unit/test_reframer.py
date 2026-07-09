from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.editing.reframer import ReframeError, reframe_vertical


class _FakeProcess:
    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", self._stderr.encode("utf-8")


@pytest.mark.asyncio
async def test_reframe_vertical_creates_output_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_clip = tmp_path / "clip_01_captioned.mp4"
    input_clip.write_bytes(b"clip")

    captured_args: list[tuple] = []

    async def fake_create_subprocess_exec(*args, **_kwargs):
        captured_args.append(args)
        output_path = Path(args[-1])
        output_path.write_bytes(b"vertical")
        return _FakeProcess(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = await reframe_vertical([input_clip], tmp_path / "out", width=1080, height=1920)

    assert len(result) == 1
    assert result[0].exists()
    assert result[0].name == "clip_01_captioned_vertical.mp4"
    assert any("scale=1080:1920" in str(arg) for arg in captured_args[0])


@pytest.mark.asyncio
async def test_reframe_vertical_raises_for_missing_clip(tmp_path: Path) -> None:
    with pytest.raises(ReframeError):
        await reframe_vertical([tmp_path / "missing.mp4"], tmp_path / "out")


@pytest.mark.asyncio
async def test_reframe_vertical_raises_when_ffmpeg_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_clip = tmp_path / "clip_01_captioned.mp4"
    input_clip.write_bytes(b"clip")

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return _FakeProcess(returncode=1, stderr="ffmpeg mock reframe failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(ReframeError):
        await reframe_vertical([input_clip], tmp_path / "out")


@pytest.mark.asyncio
async def test_reframe_vertical_rejects_invalid_dimensions(tmp_path: Path) -> None:
    input_clip = tmp_path / "clip_01_captioned.mp4"
    input_clip.write_bytes(b"clip")

    with pytest.raises(ReframeError):
        await reframe_vertical([input_clip], tmp_path / "out", width=0, height=1920)
