from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.config.settings import settings
from src.editing.captions import CaptionError, add_captions
from src.models.schemas import ClipAsset, TranscriptSegment


class _FakeProcess:
    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", self._stderr.encode("utf-8")


@pytest.mark.asyncio
async def test_add_captions_generates_srt_and_captioned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_path = tmp_path / "clip_01.mp4"
    clip_path.write_bytes(b"clip")

    clip = ClipAsset(
        clip_index=1,
        source_video=tmp_path / "source.mp4",
        output_path=clip_path,
        start_seconds=10.0,
        end_seconds=20.0,
    )

    transcript = [
        TranscriptSegment(
            start_seconds=9.0,
            end_seconds=12.0,
            text="first part with a much longer caption line that should wrap for vertical video",
        ),
        TranscriptSegment(
            start_seconds=14.0,
            end_seconds=18.0,
            text="second part keeps the caption going and should also wrap cleanly",
        ),
        TranscriptSegment(start_seconds=21.0, end_seconds=22.0, text="outside"),
    ]

    original_font_size = settings.caption_font_size
    original_margin_v = settings.caption_margin_v
    settings.caption_font_size = 16
    settings.caption_margin_v = 64

    captured_args: list[tuple] = []

    async def fake_create_subprocess_exec(*args, **_kwargs):
        captured_args.append(args)
        output_path = Path(args[-1])
        output_path.write_bytes(b"captioned")
        return _FakeProcess(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    try:
        output_dir = tmp_path / "captions"
        result = await add_captions([clip], transcript, output_dir)

        assert len(result) == 1
        assert result[0].exists()

        srt_path = output_dir / "clip_01.srt"
        assert srt_path.exists()
        srt_content = srt_path.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:02,000" in srt_content
        assert "first part" in srt_content
        assert "second part" in srt_content
        assert "\n" in srt_content

        assert captured_args
        assert "FontSize=16" in str(captured_args[0])
        assert "MarginV=64" in str(captured_args[0])
    finally:
        settings.caption_font_size = original_font_size
        settings.caption_margin_v = original_margin_v


@pytest.mark.asyncio
async def test_add_captions_raises_when_no_clip_overlap(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip_01.mp4"
    clip_path.write_bytes(b"clip")

    clip = ClipAsset(
        clip_index=1,
        source_video=tmp_path / "source.mp4",
        output_path=clip_path,
        start_seconds=10.0,
        end_seconds=20.0,
    )

    transcript = [TranscriptSegment(start_seconds=0.0, end_seconds=5.0, text="outside")]

    with pytest.raises(CaptionError):
        await add_captions([clip], transcript, tmp_path / "captions")


@pytest.mark.asyncio
async def test_add_captions_raises_when_ffmpeg_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_path = tmp_path / "clip_01.mp4"
    clip_path.write_bytes(b"clip")

    clip = ClipAsset(
        clip_index=1,
        source_video=tmp_path / "source.mp4",
        output_path=clip_path,
        start_seconds=10.0,
        end_seconds=20.0,
    )

    transcript = [TranscriptSegment(start_seconds=12.0, end_seconds=15.0, text="inside")]

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return _FakeProcess(returncode=1, stderr="ffmpeg subtitle failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(CaptionError):
        await add_captions([clip], transcript, tmp_path / "captions")
