from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    highlight_model: str = Field(default="deepseek-chat", alias="HIGHLIGHT_MODEL")

    default_language: str = Field(default="en", alias="DEFAULT_LANGUAGE")
    default_top_clips: int = Field(default=5, alias="DEFAULT_TOP_CLIPS")
    default_clip_min_seconds: int = Field(default=30, alias="DEFAULT_CLIP_MIN_SECONDS")
    default_clip_max_seconds: int = Field(default=60, alias="DEFAULT_CLIP_MAX_SECONDS")
    vertical_output_width: int = Field(default=1080, alias="VERTICAL_OUTPUT_WIDTH")
    vertical_output_height: int = Field(default=1920, alias="VERTICAL_OUTPUT_HEIGHT")
    transcription_device: str = Field(default="cpu", alias="TRANSCRIPTION_DEVICE")
    transcription_compute_type: str = Field(default="int8", alias="TRANSCRIPTION_COMPUTE_TYPE")
    caption_font_size: int = Field(default=18, alias="CAPTION_FONT_SIZE")
    caption_margin_v: int = Field(default=72, alias="CAPTION_MARGIN_V")
    caption_max_line_chars: int = Field(default=22, alias="CAPTION_MAX_LINE_CHARS")
    youtube_download_format: str = Field(
        default="bv*[height<=1080]+ba/b[height<=1080]/b",
        alias="YTDLP_FORMAT",
    )

    assets_dir: Path = Path("assets")
    input_dir: Path = Path("assets/input")
    output_dir: Path = Path("assets/output")
    temp_dir: Path = Path("assets/temp")


settings = AppSettings()
