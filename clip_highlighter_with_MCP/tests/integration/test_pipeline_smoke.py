from pathlib import Path

import pytest

from src.models.schemas import ClipAsset, HighlightCandidate, TranscriptSegment, VideoAsset
from src.pipeline import orchestrator
from src.pipeline.orchestrator import run_pipeline


def test_pipeline_symbol_available() -> None:
    assert callable(run_pipeline)


@pytest.mark.asyncio
async def test_run_pipeline_continues_when_one_clip_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ingest_youtube_url(*_args, **_kwargs) -> VideoAsset:
        return VideoAsset(source_url="https://youtu.be/demo", local_path=Path("assets/input/demo.mp4"))

    async def fake_transcribe_video(*_args, **_kwargs) -> list[TranscriptSegment]:
        return [TranscriptSegment(start_seconds=0, end_seconds=40, text="transcript")]

    async def fake_detect_highlights(*_args, **_kwargs) -> list[HighlightCandidate]:
        return [
            HighlightCandidate(
                start_seconds=0,
                end_seconds=20,
                hook_title="One",
                rationale="First",
                confidence=0.9,
            ),
            HighlightCandidate(
                start_seconds=20,
                end_seconds=40,
                hook_title="Two",
                rationale="Second",
                confidence=0.8,
            ),
        ]

    async def fake_cut_clips(*_args, **_kwargs) -> list[ClipAsset]:
        return [
            ClipAsset(
                clip_index=1,
                source_video=Path("assets/input/demo.mp4"),
                output_path=Path("assets/temp/clip_01.mp4"),
                start_seconds=0,
                end_seconds=20,
            ),
            ClipAsset(
                clip_index=2,
                source_video=Path("assets/input/demo.mp4"),
                output_path=Path("assets/temp/clip_02.mp4"),
                start_seconds=20,
                end_seconds=40,
            ),
        ]

    async def fake_add_captions(clips: list[ClipAsset], *_args, **_kwargs) -> list[Path]:
        if clips[0].clip_index == 2:
            raise RuntimeError("caption failure")
        return [Path("assets/temp/clip_01_captioned.mp4")]

    async def fake_reframe_vertical(clips: list[Path], *_args, **_kwargs) -> list[Path]:
        return [Path(str(clips[0]).replace("_captioned", "_captioned_vertical"))]

    monkeypatch.setattr(orchestrator, "ingest_youtube_url", fake_ingest_youtube_url)
    monkeypatch.setattr(orchestrator, "transcribe_video", fake_transcribe_video)
    monkeypatch.setattr(orchestrator, "detect_highlights", fake_detect_highlights)
    monkeypatch.setattr(orchestrator, "cut_clips", fake_cut_clips)
    monkeypatch.setattr(orchestrator, "add_captions", fake_add_captions)
    monkeypatch.setattr(orchestrator, "reframe_vertical", fake_reframe_vertical)

    result = await run_pipeline("https://youtu.be/demo")

    assert len(result.final_clips) == 1
    assert result.final_clips[0].name.endswith("_vertical.mp4")


@pytest.mark.asyncio
async def test_run_pipeline_returns_empty_when_all_clips_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ingest_youtube_url(*_args, **_kwargs) -> VideoAsset:
        return VideoAsset(source_url="https://youtu.be/demo", local_path=Path("assets/input/demo.mp4"))

    async def fake_transcribe_video(*_args, **_kwargs) -> list[TranscriptSegment]:
        return [TranscriptSegment(start_seconds=0, end_seconds=30, text="transcript")]

    async def fake_detect_highlights(*_args, **_kwargs) -> list[HighlightCandidate]:
        return [
            HighlightCandidate(
                start_seconds=0,
                end_seconds=30,
                hook_title="One",
                rationale="Only",
                confidence=0.9,
            )
        ]

    async def fake_cut_clips(*_args, **_kwargs) -> list[ClipAsset]:
        return [
            ClipAsset(
                clip_index=1,
                source_video=Path("assets/input/demo.mp4"),
                output_path=Path("assets/temp/clip_01.mp4"),
                start_seconds=0,
                end_seconds=30,
            )
        ]

    async def fake_add_captions(*_args, **_kwargs) -> list[Path]:
        raise RuntimeError("always fail")

    monkeypatch.setattr(orchestrator, "ingest_youtube_url", fake_ingest_youtube_url)
    monkeypatch.setattr(orchestrator, "transcribe_video", fake_transcribe_video)
    monkeypatch.setattr(orchestrator, "detect_highlights", fake_detect_highlights)
    monkeypatch.setattr(orchestrator, "cut_clips", fake_cut_clips)
    monkeypatch.setattr(orchestrator, "add_captions", fake_add_captions)

    result = await run_pipeline("https://youtu.be/demo")

    assert result.final_clips == []


@pytest.mark.asyncio
async def test_run_pipeline_removes_temp_video_files_after_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    temp_dir.mkdir()
    output_dir.mkdir()
    input_dir.mkdir()

    monkeypatch.setattr(orchestrator.settings, "temp_dir", temp_dir)
    monkeypatch.setattr(orchestrator.settings, "output_dir", output_dir)
    monkeypatch.setattr(orchestrator.settings, "input_dir", input_dir)

    async def fake_ingest_youtube_url(*_args, **_kwargs) -> VideoAsset:
        return VideoAsset(source_url="https://youtu.be/demo", local_path=input_dir / "demo.mp4")

    async def fake_transcribe_video(*_args, **_kwargs) -> list[TranscriptSegment]:
        return [TranscriptSegment(start_seconds=0, end_seconds=30, text="transcript")]

    async def fake_detect_highlights(*_args, **_kwargs) -> list[HighlightCandidate]:
        return [
            HighlightCandidate(
                start_seconds=0,
                end_seconds=30,
                hook_title="One",
                rationale="Only",
                confidence=0.9,
            )
        ]

    async def fake_cut_clips(*_args, **_kwargs) -> list[ClipAsset]:
        clip_path = temp_dir / "clip_01.mp4"
        clip_path.write_bytes(b"raw")
        (temp_dir / "clip_01.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
        return [
            ClipAsset(
                clip_index=1,
                source_video=input_dir / "demo.mp4",
                output_path=clip_path,
                start_seconds=0,
                end_seconds=30,
            )
        ]

    async def fake_add_captions(clips: list[ClipAsset], *_args, **_kwargs) -> list[Path]:
        captioned_path = temp_dir / "clip_01_captioned.mp4"
        captioned_path.write_bytes(b"captioned")
        return [captioned_path]

    async def fake_reframe_vertical(clips: list[Path], *_args, **_kwargs) -> list[Path]:
        output_path = output_dir / "clip_01_captioned_vertical.mp4"
        output_path.write_bytes(b"vertical")
        return [output_path]

    monkeypatch.setattr(orchestrator, "ingest_youtube_url", fake_ingest_youtube_url)
    monkeypatch.setattr(orchestrator, "transcribe_video", fake_transcribe_video)
    monkeypatch.setattr(orchestrator, "detect_highlights", fake_detect_highlights)
    monkeypatch.setattr(orchestrator, "cut_clips", fake_cut_clips)
    monkeypatch.setattr(orchestrator, "add_captions", fake_add_captions)
    monkeypatch.setattr(orchestrator, "reframe_vertical", fake_reframe_vertical)

    result = await run_pipeline("https://youtu.be/demo")

    assert len(result.final_clips) == 1
    assert result.final_clips[0].exists()
    assert not (temp_dir / "clip_01.mp4").exists()
    assert not (temp_dir / "clip_01_captioned.mp4").exists()
    assert not (temp_dir / "clip_01.srt").exists()
