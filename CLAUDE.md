# whisper-subtitle-cli

## Project Overview
CLI tool to generate subtitles from video/audio files using OpenAI Whisper AI model, with optional translation via local Ollama API.

## Requirements
- Extract audio from video files or YouTube URLs
- Transcribe audio using Whisper AI (speech-to-text)
- Output SRT format (with timestamps for video playback)
- Translate subtitles using local Ollama models
- Support common video formats: MP4, MKV, AVI, MOV, WebM
- Support YouTube and 1000+ other video platforms via URLs

## Technical Stack
- **Python 3.11+**
- **uv** for package management
- **OpenAI Whisper** for AI transcription (CPU/CUDA)
- **mlx-whisper** (optional) for AI transcription on Apple Silicon via Metal GPU
- **stable-ts** (optional) for better timestamp accuracy with any backend
- **ffmpeg** (system dependency) for audio extraction
- **yt-dlp** for downloading videos from URLs
- **Ollama** (local) for subtitle translation

## Output Files
All subtitle files include a date prefix (YYYYMMDD format) and use underscores instead of spaces for easier command line usage.

**Generated Files**:
- `YYYYMMDD_video_id.srt` - SRT subtitle file with timestamps (for URL inputs)
- `YYYYMMDD_filename.srt` - SRT subtitle file (for local files)
- `YYYYMMDD_video_id.{Language}.srt` - Translated subtitle file (when translation is used)
- `YYYYMMDD_video_id.bilingual.srt` - Bilingual subtitle with original + translation (optional)

**Output Location** (priority: custom > default):
- **Custom output**: Use `--output` flag or config `output.directory` to specify directory
- **Local video files**: Saved to same directory as the video file
- **YouTube/URL inputs**: Saved to current working directory (where program is run from)

**Filename Rules**:
- **URL inputs**: Uses video ID (e.g., `dQw4w9WgXcQ`) - short and easy to locate video
- **Local files**: Uses sanitized filename with spaces replaced by underscores

**Date Logic**:
- YouTube URLs: Uses video's original upload date
- Local files: Uses file's modification date
- Fallback: Uses current date if neither is available

## Configuration

Settings are configured in `config.json` at the project root.

### Full Configuration Example
```json
{
  "ollama": {
    "model": "translategemma:4b",
    "base_url": "http://localhost:11434",
    "batch_size": 50,
    "keep_alive": "10m",
    "auto_unload": false,
    "context_lines": 3
  },
  "output": {
    "directory": "/path/to/subtitles"
  }
}
```

### Output Settings
- **output.directory**: Default output directory for all subtitle files
  - If not set, uses default locations (cwd for URLs, video's directory for local files)
  - Can be overridden by `--output` CLI flag
  - Priority: CLI argument > config > default

### Ollama Settings
- **model**: The Ollama model to use for translation (default: `translategemma:4b`)
- **base_url**: Ollama API URL, can point to remote Ollama server (default: `http://localhost:11434`)
- **batch_size**: Number of segments to translate per API call (default: `50`)
- **context_lines**: Number of prior translated segment pairs to include as read-only context in each batch prompt (default: `3`, set `0` to disable). Helps maintain consistency in pronouns, terminology, and tone across batch boundaries.
- **keep_alive**: How long to keep the model loaded in memory after a request (default: `10m`)
  - `"5m"`, `"10m"`, `"1h"` - duration values
  - `"-1"` - keep loaded indefinitely
  - `"0"` - unload immediately after request
- **auto_unload**: Unload Ollama models before Whisper transcription to free VRAM (default: `false`)
  - `false` — do nothing; `--preview` outputs a single combined command
  - `true` — evict all loaded Ollama models before Whisper loads; `--preview` outputs two-phase commands (transcribe first, then translate)
  - Enable if your GPU doesn't have enough VRAM to run both Ollama and Whisper simultaneously

**Note:** Only Ollama API is supported. Other APIs (OpenAI, Claude, etc.) are not compatible.

### CLI Options
- `--model`: Whisper model size: tiny, base, small, medium, large (default: medium)
- `--language`: Source language code for transcription (e.g., en, zh, es). Auto-detect if not specified.
- `--output`, `-o`: Output directory for subtitle files
- `--keep-audio`: Keep the extracted audio file (WAV)
- `--yes`, `-y`: Auto-accept translation prompts with defaults
- `--check-system`: Display system diagnostics (GPU, CUDA, ffmpeg, Ollama)
- `--stable`: Use stable-ts for better timestamp accuracy (requires: `uv sync --extra stable`)
- `--vad`: Enable VAD to reduce hallucinations in silence (requires: `--stable`)
- `--prompt-file`: Path to a text file with extra instructions for the translation model (e.g., glossary, style guide)
- `--preview-opt`: Non-interactive preview selection (`L`=list JSON, `S`=skip, `0`=transcribe, `N`=subtitle index). Implies `--preview`.
- `--action`: Action to perform: `transcribe` (skip translation entirely) or `translate` (SRT input only, skip transcription). Default: both (prompt for translation after transcribing)
  - `--action transcribe` — transcribe only, no translation prompt
  - `--action translate` — requires SRT file as input, goes directly to translation

## Project Structure
```
whisper-subtitle-cli/
├── src/
│   ├── transcriber.py      # Whisper transcription logic
│   ├── subtitle_writer.py  # SRT file generation
│   ├── audio_extractor.py  # Audio extraction from video
│   ├── video_downloader.py # YouTube/URL video downloading
│   └── translator.py       # Ollama translation logic
├── main.py                 # CLI entry point
├── config.json             # Ollama configuration
├── tests/
│   ├── test_transcriber.py
│   ├── test_subtitle_writer.py
│   ├── test_subtitle_download.py
│   ├── test_audio_extractor.py
│   ├── test_video_downloader.py
│   ├── test_translator.py
│   ├── test_preview_opt.py
│   └── test_main_integration.py
├── pyproject.toml
├── uv.lock                 # uv lock file
├── .python-version         # Python version (used by uv)
├── README.md               # User documentation
├── INSTALL.md              # Installation guide
└── CLAUDE.md
```

## Dependencies
- openai-whisper
- mlx-whisper (optional, Apple Silicon)
- stable-ts (optional, better timestamps)
- ffmpeg-python (Python wrapper)
- click (for CLI interface)
- yt-dlp (for downloading videos from URLs)
- requests (for Ollama API calls)
- pytest (for testing)

## Current Status
✅ **Implementation Complete**

All core features implemented and tested:
- ✅ Audio extraction from video files
- ✅ YouTube URL support (and 1000+ other platforms via yt-dlp)
- ✅ Subtitle download from YouTube (human-made subtitles only)
- ✅ AI transcription using OpenAI Whisper
- ✅ SRT subtitle file generation
- ✅ Subtitle translation via local Ollama API (batch processing with recursive retry)
- ✅ CLI interface with options
- ✅ System diagnostics (`--check-system`)

## Usage

See `README.md` for full usage examples:
- Basic usage and advanced options → `README.md` § Basic Usage, Advanced Options
- Subtitle download (YouTube) → `README.md` § Subtitle Download (YouTube)
- Subtitle translation (Ollama) → `README.md` § Subtitle Translation (Ollama)
- Batch scripting (`--preview`, `--subtitle`, `--action`) → `README.md` § Batch Scripting, The `--action` Flag
- Whisper model sizes and language codes → `README.md` § Whisper Model Options, Language Codes

## Setup
```bash
# Install dependencies (creates .venv and installs all packages)
uv sync

# Apple Silicon: install with mlx-whisper for Metal GPU acceleration
uv sync --extra mlx

# Optional: install stable-ts for better timestamp accuracy
uv sync --extra stable

# Combine extras (Apple Silicon with stable-ts)
uv sync --extra mlx --extra stable
```

## Testing
```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_transcriber.py -v

# Run translator tests
uv run pytest tests/test_translator.py -v
```

## Windows CUDA Support (Maintenance Guide)

### Background

PyTorch on Windows from PyPI is **CPU-only**. To get CUDA support on Windows, we must:
1. Use PyTorch's wheel index (`https://download.pytorch.org/whl`)
2. Specify exact CUDA version (e.g., `torch==2.5.1+cu121`)

Linux doesn't have this problem - PyPI torch includes CUDA automatically.

### How It Works

1. **Default behavior** (`uv sync`):
   - Windows: Installs `torch==2.5.1+cu121` from PyTorch wheel index
   - Linux/macOS: Installs torch from PyPI (CUDA included on Linux)

2. **If CUDA version mismatch**:
   - User runs `python main.py --check-system`
   - Shows driver version and compatible CUDA versions
   - Suggests override command: `uv pip install torch==X.X.X+cuXXX --index-url ...`

### Configuration Location

- **CUDA_VERSIONS dict** in `main.py`: Maps CUDA versions to driver requirements
- **DEFAULT_CUDA** in `main.py`: Current default (cu121)
- **pyproject.toml**: Windows torch version and PyTorch wheel index

### Adding a New CUDA Version

When PyTorch releases support for a new CUDA version (e.g., cu126):

1. **Find minimum driver version** at:
   https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

2. **Update `main.py`** - add to CUDA_VERSIONS dict:
   ```python
   CUDA_VERSIONS = {
       "cu118": {"min_driver": 520, "cuda": "11.8", "torch": "2.5.1+cu118"},
       "cu121": {"min_driver": 525, "cuda": "12.1", "torch": "2.5.1+cu121"},
       # Add new version here if needed
   }
   ```

   **Note:** torch 2.5.1 is pinned because newer versions have compatibility issues with Whisper.
   Only add new CUDA versions if they offer torch 2.5.1 builds AND benefit users (e.g., older driver support).

3. **Update `pyproject.toml`** if changing default:
   ```toml
   # Update the Windows torch version
   "torch==2.6.0+cu126; sys_platform == 'win32'",
   ```

4. **Update `INSTALL.md`** with new override commands

5. **Update DEFAULT_CUDA** in `main.py` if changing default

### Updating PyTorch Version

When updating to a new PyTorch version (e.g., 2.6.0):

1. Update all torch versions in `CUDA_VERSIONS` dict
2. Update `pyproject.toml` Windows torch version
3. Test on Windows if possible, or document for contributors to test

### Why cu121 as Default?

- cu121 (CUDA 12.1) requires driver >= 525
- Driver 525+ was released in late 2022
- Most users with reasonably recent NVIDIA drivers should have this
- Provides good balance between compatibility and performance

## Next Steps

### Completed
- `--preview` and `--subtitle` flags for batch script automation (see `plan/finished/PLAN-preview-subtitle-flags.md`)
- Fix subtitle language code handling — dirty yt-dlp codes now display cleanly (see `plan/finished/PLAN-fix-subtitle-lang-codes.md`)
- Two-phase preview commands for VRAM-constrained machines (see `plan/finished/PLAN-two-phase-preview.md`)
- Auto-unload Ollama models before Whisper, configurable via `ollama.auto_unload` in `config.json` (see `plan/finished/PLAN-configurable-auto-unload.md`)
- `--action` flag to separate transcribe/translate steps (see `plan/finished/PLAN-action-flag.md`)
- `S. Skip this video` option in the subtitle selection menu — preview mode emits nothing (video absent from batch script), interactive mode exits cleanly
- Channel name + video title shown in subtitle menu header and as `# comment` in `--preview` output
- Sliding context window for translation — `ollama.context_lines` (default 3) passes last N translated pairs as read-only context between batches (see `plan/PLAN-sliding-context-window.md`)
- `--preview` with no subtitles now prompts to transcribe or skip (instead of auto-transcribing), so users can skip videos that don't need transcription (e.g., Chinese videos)
- `--prompt-file` flag for custom translation instructions (glossary, style guide, terminology) — injected as `[Additional instructions:]` block in all translation prompts
- `--prompt-file` now passed through to `--preview` output commands (translate and single-phase only, not transcribe-only)
- `--preview-opt` flag for non-interactive preview: `L` (JSON list), `S` (skip), `0` (transcribe), `N` (subtitle index). Implies `--preview`. Enables external automation (Telegram bot, API).

### Future (Optional Enhancements)
- Add support for batch processing multiple videos/URLs
- Add progress bars for long videos
- Support for additional subtitle formats (VTT, ASS)
- Playlist support (download and process all videos from a playlist)
- Web interface

### Next Session
- Run `uv run pytest -v` to verify clean state (242 passed, 7 skipped as of last session)
- The 7 skipped tests are `stable-ts` related — they skip automatically when `stable-ts` is not installed (optional dep). Run `uv sync --extra stable` to enable them.
- `config.json` has uncommitted local changes (user's personal model/output settings) — leave as-is
- `scripts/` directory moved to `~/claude/automation-scripts/` (separate repo, shared env.sh pattern for cron-friendly absolute paths). The old `scripts/` is gitignored and removed from this project.
- `.gitignore` has an uncommitted change adding `scripts/` — can be committed or left as-is (the directory no longer exists here)
- Consider adding `ollama.prompt_file` config option as an alternative to CLI flag (not yet implemented)
- `--preview-opt` is implemented — next logical step is building the Telegram bot that uses it (query with `L`, present options, select with `0`/`N`/`S`)
- Click limitation: `--preview` cannot take an optional argument natively. That's why `--preview-opt` is a separate option (see `plan/PLAN-preview-opt.md` for design rationale)
- `tests/test_preview_opt.py` added with 19 tests; file has minor Pyright warnings for unused mock variables (cosmetic only)
