# Clip Highlighter with AI

A Python project that ingests a long YouTube video, transcribes it, detects highlight moments with an LLM, then cuts, captions, and reframes short clips.

## Current Status
Core pipeline phases 1-7 are implemented (ingest, transcription, highlights, cut, captions, reframe, orchestration).

## Pipeline
1. Ingest YouTube URL
2. Transcribe with Faster-Whisper
3. Select highlights with DeepSeek
4. Cut clips with FFmpeg
5. Add burned-in captions
6. Reframe to vertical format

## Setup (PowerShell)
Run from project root:

```powershell
cd G:\my_projects\clip_highlighter_with_MCP
.\scripts\bootstrap.ps1
```

Then edit `.env` and set your DeepSeek API key:

```text
DEEPSEEK_API_KEY=your_key_here
```

## Switch Transcription from CPU to GPU
The app defaults to CPU transcription so it runs on a normal Windows machine without CUDA.

To use a GPU instead, update these values in `.env`:

```text
TRANSCRIPTION_DEVICE=cuda
TRANSCRIPTION_COMPUTE_TYPE=float16
```

If you want to stay on CPU, keep the defaults:

```text
TRANSCRIPTION_DEVICE=cpu
TRANSCRIPTION_COMPUTE_TYPE=int8
```

GPU mode only works if your NVIDIA drivers and CUDA runtime are installed correctly. If GPU mode fails, switch back to the CPU values above.

## Run the App (PowerShell)
```powershell
cd G:\my_projects\clip_highlighter_with_MCP
.\.venv\Scripts\python.exe -m src.main run --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Generated clips will be written under `assets/output`.

## Run MCP Server (PowerShell)
Use these steps to run the full app through MCP as an end user.

### 1. Open terminal in project root
```powershell
cd G:\my_projects\clip_highlighter_with_MCP
```

### 2. Ensure your .env has DeepSeek key
```text
DEEPSEEK_API_KEY=your_key_here
```

### 3. Start the MCP server (Terminal A)
```powershell
cd G:\my_projects\clip_highlighter_with_MCP
.\.venv\Scripts\python.exe -m src.mcp.server
```

Keep this terminal running.

### 4. Start MCP Inspector (Terminal B)
```powershell
cd G:\my_projects\clip_highlighter_with_MCP
npx @modelcontextprotocol/inspector --command ".\.venv\Scripts\python.exe" --args "-m","src.mcp.server"
```

If prompted by npx on first run, accept installation.

### 5. Call the full app tool
In MCP Inspector:
- Select tool: run_full_pipeline
- Provide input:

```json
{
	"url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### 6. Check outputs
- Final clips are written to assets/output
- Temporary files are cleaned from assets/temp

### Available MCP tools
- ingest_youtube
- transcribe_media
- select_highlights
- cut_video_clips
- caption_video_clips
- reframe_video_clips
- cleanup_temp
- run_full_pipeline

Use run_full_pipeline for one-shot processing, or call tools one-by-one for custom orchestration.

## Useful Commands (PowerShell)
```powershell
# Run tests
.\.venv\Scripts\python.exe -m pytest -q

# Lint and format
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black .
```

## Notes
- MCP adapter layer is implemented as a thin wrapper over the existing pipeline modules.
- Local-first execution is prioritized to keep testing cost low.
