from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class VideoAsset(BaseModel):
    source_url: str
    local_path: Path
    video_id: str | None = None
    title: str | None = None


class TranscriptSegment(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str


class HighlightCandidate(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    hook_title: str
    rationale: str
    confidence: float = Field(ge=0, le=1)


class ClipAsset(BaseModel):
    clip_index: int
    source_video: Path
    output_path: Path
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)


class PipelineResult(BaseModel):
    source: VideoAsset
    transcript_segments: list[TranscriptSegment]
    highlights: list[HighlightCandidate]
    final_clips: list[Path]
