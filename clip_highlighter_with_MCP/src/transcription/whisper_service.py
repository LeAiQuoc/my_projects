from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.settings import settings
from src.models.schemas import TranscriptSegment

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


class TranscriptionError(RuntimeError):
    """Base error for transcription failures."""

    pass


class WhisperModelLoadError(TranscriptionError):
    """Raised when the faster-whisper model cannot be initialized."""

    pass


class WhisperInferenceError(TranscriptionError):
    """Raised when transcription inference fails."""

    pass


_MODEL: Any = None
_MODEL_NAME = "small"


def _load_whisper_model_class() -> Any:
    """Load WhisperModel class lazily to avoid import-time native library crashes."""
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        raise WhisperModelLoadError(
            "Failed to import faster-whisper. Ensure Python 3.11+ and matching native dependencies.",
        ) from exc

    return WhisperModel


def _get_model() -> Any:
    """Load and cache the Whisper model instance for reuse."""

    global _MODEL

    if _MODEL is not None:
        return _MODEL

    try:
        whisper_model_cls = _load_whisper_model_class()
        # Default to CPU so the app runs on typical Windows installs without CUDA runtime issues.
        _MODEL = whisper_model_cls(
            _MODEL_NAME,
            device=settings.transcription_device,
            compute_type=settings.transcription_compute_type,
        )
    except Exception as exc:  # noqa: BLE001
        if settings.transcription_device != "cpu":
            try:
                whisper_model_cls = _load_whisper_model_class()
                _MODEL = whisper_model_cls(_MODEL_NAME, device="cpu", compute_type="int8")
                return _MODEL
            except Exception as cpu_exc:  # noqa: BLE001
                raise WhisperModelLoadError(
                    f"Failed to load faster-whisper model '{_MODEL_NAME}' on {settings.transcription_device} and CPU fallback: {cpu_exc}",
                ) from cpu_exc

        raise WhisperModelLoadError(f"Failed to load faster-whisper model '{_MODEL_NAME}': {exc}") from exc

    return _MODEL


def _normalize_segments(raw_segments: list[Any]) -> list[TranscriptSegment]:
    """Convert raw model segments into validated TranscriptSegment objects."""

    normalized: list[TranscriptSegment] = []

    for raw in raw_segments:
        start_value = float(getattr(raw, "start", 0.0) or 0.0)
        end_value = float(getattr(raw, "end", 0.0) or 0.0)
        text_value = str(getattr(raw, "text", "") or "").strip()

        if not text_value:
            continue
        if end_value <= start_value:
            continue

        normalized.append(
            TranscriptSegment(
                start_seconds=start_value,
                end_seconds=end_value,
                text=text_value,
            )
        )

    return normalized


def _transcribe_sync(video_path: Path, language: str | None) -> list[TranscriptSegment]:
    """Run blocking transcription and return normalized transcript segments."""

    model = _get_model()

    try:
        segments_iter, _ = model.transcribe(
            str(video_path),
            language=language,
            vad_filter=True,
            beam_size=5,
        )
    except Exception as exc:  # noqa: BLE001
        raise WhisperInferenceError(f"faster-whisper transcription failed: {exc}") from exc

    raw_segments = list(segments_iter)
    return _normalize_segments(raw_segments)


async def transcribe_video(video_path: Path, language: str | None = None) -> list[TranscriptSegment]:
    """Transcribe a video/audio file into timestamped text segments.

    When language is None, faster-whisper auto-detects language.
    """
    if not video_path.exists():
        raise TranscriptionError(f"Video not found: {video_path}")

    return await asyncio.to_thread(_transcribe_sync, video_path, language)
