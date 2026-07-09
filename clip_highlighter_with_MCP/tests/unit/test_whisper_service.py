from pathlib import Path
from types import SimpleNamespace

import pytest

from src.transcription import whisper_service
from src.transcription.whisper_service import (
    TranscriptionError,
    WhisperInferenceError,
    transcribe_video,
)


class _FakeWhisperModel:
    def __init__(self, *_args, **_kwargs):
        self.called_with: tuple[str, str | None] | None = None

    def transcribe(self, media_path: str, language: str | None, **_kwargs):
        self.called_with = (media_path, language)
        segments = [
            SimpleNamespace(start=0.0, end=4.2, text=" Hello world "),
            SimpleNamespace(start=4.2, end=4.2, text="ignored because zero duration"),
            SimpleNamespace(start=5.0, end=7.0, text=""),
        ]
        info = SimpleNamespace(language=language or "en")
        return iter(segments), info


class _FailingWhisperModel:
    def __init__(self, *_args, **_kwargs):
        pass

    def transcribe(self, *_args, **_kwargs):
        raise RuntimeError("mock inference error")


@pytest.mark.asyncio
async def test_transcribe_video_returns_normalized_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_file = tmp_path / "sample.mp4"
    media_file.write_bytes(b"fake")

    monkeypatch.setattr(whisper_service, "_load_whisper_model_class", lambda: _FakeWhisperModel)
    monkeypatch.setattr(whisper_service, "_MODEL", None)

    result = await transcribe_video(media_file, language="en")

    assert len(result) == 1
    assert result[0].start_seconds == 0.0
    assert result[0].end_seconds == 4.2
    assert result[0].text == "Hello world"


@pytest.mark.asyncio
async def test_transcribe_video_raises_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"

    with pytest.raises(TranscriptionError):
        await transcribe_video(missing)


@pytest.mark.asyncio
async def test_transcribe_video_wraps_inference_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_file = tmp_path / "sample.mp4"
    media_file.write_bytes(b"fake")

    monkeypatch.setattr(whisper_service, "_load_whisper_model_class", lambda: _FailingWhisperModel)
    monkeypatch.setattr(whisper_service, "_MODEL", None)

    with pytest.raises(WhisperInferenceError):
        await transcribe_video(media_file)
