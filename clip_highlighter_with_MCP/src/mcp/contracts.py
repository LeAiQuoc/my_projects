from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptSegmentInput(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str


class HighlightInput(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    hook_title: str
    rationale: str
    confidence: float = Field(ge=0, le=1)


class ClipInput(BaseModel):
    clip_index: int
    source_video: str
    output_path: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
