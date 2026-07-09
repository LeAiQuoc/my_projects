# Vietnamese Assistant AI

A local Vietnamese voice assistant project that combines:
- Speech-to-text with Whisper
- Local LLM responses via Ollama
- Text-to-speech via Piper (or gTTS fallback)
- Audio playback and simple on-screen rendering with Pygame
- A Light Speed (`src/research/light-speed/`) TTS training module based on VITS

## Project Structure

- `src/assistant/voiceagent2.py`: main realtime voice loop (recommended)
- `src/assistant/voiceagent.py`: alternative implementation (includes translation + gTTS path)
- `src/assistant/request.py`: Argos Translate and HTTP translation experiments
- `src/assistant/pyaudiotest.py`: ffmpeg conversion test helper
- `src/config/`: configuration and runtime JSON files
- `src/data/`: runtime data and temporary files
- `src/research/light-speed/`: model training/inference code for TTS
- `resources/binaries/`: bundled ffmpeg/mpv binaries and audio USB DLL
- `resources/headers/`: C/C++ header assets
- `resources/wheels/`: local wheel packages

## Requirements

Python dependencies are listed in `requirement.txt`.

Install them in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
```

## External Tools

The assistant scripts also rely on external tools/services:

1. Ollama
- Install Ollama
- Pull the model used in code:

```bash
ollama pull mistral:7b
```

2. Piper TTS (Windows path used by `voiceagent2.py`)
- Code expects:
  - `C:\piper\piper.exe`
  - `C:\piper\vi_VN-vietTTS.onnx`

If your files are elsewhere, update the command in `voiceagent2.py`.

3. FFmpeg
- Needed by `pyaudiotest.py` and any conversion workflow
- Ensure `ffmpeg` is available on PATH, or use the bundled `resources/binaries/ffmpeg/bin` binary directly

## Running the Voice Assistant

Recommended:

```bash
python src/assistant/voiceagent2.py
```

Alternative:

```bash
python src/assistant/voiceagent.py
```

## Running Light Speed Training

From inside `src/research/light-speed/`:

```bash
cd src/research/light-speed
python train.py
```

For multi-GPU:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py
```

## Notes

- `src/assistant/request.py` is experimental and may need cleanup before production use.
- `src/config/constants.py` currently contains API keys in plain text. Move secrets to environment variables (for example with a `.env` file) before sharing or deploying this project.
- On Windows, `PyAudio` installation may fail without build tools. If needed, install a prebuilt wheel compatible with your Python version.
