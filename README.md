# whisper-subtitle-cli

Extract subtitles from video files or YouTube URLs using AI transcription (OpenAI Whisper). Generates SRT subtitle files for video playback.

> **Note**: This project was built with AI assistance.

## Quick Start

> **First time?** See [INSTALL.md](INSTALL.md) for detailed setup (uv, ffmpeg, CUDA, Ollama).

```bash
# Install dependencies
uv sync                   # CPU/CUDA
uv sync --extra mlx       # Apple Silicon (Metal GPU)
uv sync --extra stable    # Better timestamp accuracy (optional)

# Check your system (GPU, CUDA, ffmpeg, Ollama)
uv run python main.py --check-system

# Local video file
uv run python main.py video.mp4

# YouTube or URL
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Translate an existing SRT file
uv run python main.py existing.srt
```

---

## Features

- **YouTube & URL Support**: Process videos from YouTube, Vimeo, Twitch, and [1000+ platforms](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)
- **Subtitle Download**: Automatically download existing YouTube subtitles (much faster than transcription)
- **AI-Powered Transcription**: Uses OpenAI Whisper for accurate speech-to-text
- **Better Timestamps**: Optional stable-ts backend for improved timing (`--stable`, `--vad`)
- **Subtitle Translation**: Translate subtitles to any language using local Ollama models (no cloud API needed)
- **Unattended Mode**: Use `--yes` flag to auto-translate after transcription completes
- **Batch Scripting**: Use `--preview` + `--subtitle N` to automate subtitle selection across many URLs
- **Multiple Languages**: Auto-detects language or accepts manual specification
- **Flexible Models**: Choose from 5 Whisper model sizes balancing speed vs accuracy

## Installation

See [INSTALL.md](INSTALL.md) for detailed step-by-step instructions, including:
- Prerequisites (uv, ffmpeg, CUDA, Ollama)
- GPU acceleration setup (NVIDIA CUDA / Apple Silicon Metal)
- Troubleshooting common issues

## Usage

### Basic Usage

Extract subtitles from a local video file:

```bash
uv run python main.py video.mp4
```

Extract subtitles from a YouTube URL:

```bash
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

This creates an SRT file with date prefix (YYYYMMDD format):
- `YYYYMMDD_video.srt` or `YYYYMMDD_Video_Title.srt` - Subtitle file with timestamps (for video players)
- `YYYYMMDD_video.Chinese.srt` - Translated subtitle (if translation is used)

**Note**:
- Filenames use underscores instead of spaces for easier command line usage (e.g., `My Video.mp4` → `YYYYMMDD_My_Video.srt`)
- For local files, subtitles are saved in the same directory as the video
- For URLs, subtitles are saved in the current working directory (videos are downloaded to a temp directory and cleaned up by your OS)

### Advanced Options

```bash
# Use a different model size
uv run python main.py video.mp4 --model small

# Specify the language (faster than auto-detect)
uv run python main.py video.mp4 --language en

# Save output to a specific directory
uv run python main.py video.mp4 --output ./subtitles

# Keep the extracted audio file
uv run python main.py video.mp4 --keep-audio

# Use stable-ts for better timestamps
uv run python main.py video.mp4 --stable

# Add VAD to reduce hallucinations in silence (optional)
uv run python main.py video.mp4 --stable --vad

# Skip subtitle prompt: pre-select by index (0=transcribe, 1+=download that subtitle)
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --subtitle 1
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --subtitle 0  # force transcribe

# Transcribe only — skip translation entirely
uv run python main.py video.mp4 --action transcribe

# Translate an existing SRT directly — skip download/transcription
uv run python main.py existing.srt --action translate -y

# Use a custom prompt file for translation (glossary, style guide, etc.)
uv run python main.py video.mp4 --prompt-file glossary.txt

# Preview mode: check subtitles, ask user, print the real command, then exit
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --preview

# YouTube URLs work the same way
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python main.py "https://youtu.be/VIDEO_ID"  # Short format

# Other platforms (Vimeo, Twitch, etc.)
uv run python main.py "https://vimeo.com/123456"
```

### Whisper Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | ~39MB | Fastest | Basic | Quick tests, simple audio |
| base | ~140MB | Fast | Good | Faster alternative |
| small | ~470MB | Moderate | Better | Clear speech, important content |
| medium | ~1.5GB | Slow | High | **Default - Professional use** |
| large | ~2.9GB | Slowest | Best | Maximum accuracy needed |

### Language Codes

Common language codes (or use auto-detect by omitting):
- `en` - English
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `ko` - Korean

[Full list of supported languages](https://github.com/openai/whisper#available-models-and-languages)

### Subtitle Download (YouTube)

When you provide a YouTube URL, the tool automatically checks for existing subtitles:

```bash
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# The tool will show available subtitles:
Checking for available subtitles...

Available subtitles:
  1. English (en)
  2. Spanish (es)
  3. German (de)
  0. Transcribe video instead
  S. Skip this video

Which subtitle would you like? (0-N, or S to skip) [0]: 1

Downloading English subtitle...
✓ Subtitle downloaded: YYYYMMDD_Video_Title.srt
✅ Done!
```

**Benefits**:
- ⚡ **Much faster**: ~1 second vs 30-120 seconds for transcription
- ✅ **More accurate**: Human-made subtitles are often better than AI
- 🎯 **Smart filtering**: Only shows manual subtitles (auto-generated are hidden)
- 🔄 **Automatic fallback**: If no subtitles exist, automatically transcribes
- 🎛️ **User choice**: Select option 0 to transcribe even when subtitles available
- ⏭️ **Skip option**: Enter `S` to skip a video entirely (useful in batch preview runs)
- 🤖 **Automation-friendly**: Use `--subtitle N` to pre-select, `--preview` for batch scripts (see [Batch Scripting](#batch-scripting---preview-and---subtitle))

### Subtitle Translation (Ollama)

After creating or downloading subtitles, you can translate them to another language using a local Ollama model.

**Translate during transcription:**

```bash
uv run python main.py video.mp4

# After transcription completes:
✓ SRT file created: 20250119_video.srt

Would you like to translate the subtitles? [Y/n]:
Source language [English]:
Target language [Chinese]:
Create bilingual subtitle (original + translation)? [Y/n]:

Using Ollama model: translategemma:4b

Translating 150 segments...
  Translating segment 150/150...
✓ Translated SRT created: 20250119_video.Chinese.srt

✅ Done!
```

**Translate an existing SRT file:**

If you already have an SRT file (from a previous run or another source), you can translate it directly without re-downloading or re-transcribing:

```bash
uv run python main.py existing_subtitle.srt

# Output:
SRT file detected: existing_subtitle.srt
Skipping download/transcription, going directly to translation.

✓ Parsed 150 segments from SRT file

Would you like to translate the subtitles? [Y/n]:
Source language [English]:
Target language [Chinese]: Japanese

Using Ollama model: translategemma:4b

Translating 150 segments...
✓ Translated SRT created: existing_subtitle.Japanese.srt

✅ Done!
```

This is useful when:
- Translation failed midway and you want to retry
- You want to translate to a different language
- You have SRT files from another source

**Requirements:**
- Ollama must be installed and running (`ollama serve`)
- A model must be downloaded (`ollama pull translategemma:4b`)
- See [INSTALL.md](INSTALL.md) for Ollama setup instructions

**Configuration:**
Edit `config.json` to change settings:
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

- `ollama.model`: Ollama model for translation
- `ollama.batch_size`: Segments per API call (higher = better context, more memory)
- `ollama.keep_alive`: How long model stays loaded (`"10m"`, `"1h"`, `"-1"` for indefinitely)
- `ollama.auto_unload`: Set to `true` if your GPU doesn't have enough VRAM to run Ollama and Whisper simultaneously. When enabled, Ollama models are evicted before Whisper loads, and `--preview` outputs two separate commands (transcribe first, then translate). Default: `false`.
- `ollama.context_lines`: Number of prior translated segment pairs passed as read-only context to each batch (default: `3`, set `0` to disable). Keeps pronouns, names, and tone consistent across batch boundaries.
- `output.directory`: Default output directory (overrides default, can be overridden by `--output` flag)

**Note:** Translation uses Ollama's local API only. The `base_url` can point to a remote Ollama server, but other APIs (OpenAI, Claude, etc.) are not supported.

### The `--yes` Flag

Use `--yes` (or `-y`) to auto-translate after transcription without prompts. Useful when transcription takes a long time and you want to walk away.

```bash
uv run python main.py video.mp4 --yes
```

Translation defaults with `--yes`:
- Source language: value of `--language` if provided, otherwise "English"
- Target language: "Chinese"
- Bilingual subtitle: Yes

**Example: URL with no subtitles available**
```bash
uv run python main.py "https://youtube.com/..." --language en --yes

# No subtitles found, transcribing...
# (transcription runs, you can walk away)
# ✓ SRT file created: 20260123_video_id.srt
# Auto-translates: English → Chinese, bilingual
# ✓ Translated SRT created: 20260123_video_id.Chinese.srt
# ✓ Bilingual SRT created: 20260123_video_id.bilingual.srt
# ✅ Done!
```

**Example: URL with subtitles available**
```bash
uv run python main.py "https://youtube.com/..." --language en --yes

# Available subtitles:
#   1. English (en)
#   2. Spanish (es)
#   0. Transcribe video instead
#   S. Skip this video
# Which subtitle would you like? (0-N, or S to skip) [0]:    ← still prompts here
#
# (after you choose, auto-translates without further prompts)
```

### The `--action` Flag

Use `--action` to explicitly control what step to perform:

| Flag | Behavior |
|------|----------|
| *(none)* | Transcribe, then prompt for translation (default) |
| `--action transcribe` | Transcribe only — no translation prompt |
| `--action translate` | Translate an existing SRT — skip download/transcription |

```bash
# Transcribe only — no translation prompt
uv run python main.py video.mp4 --action transcribe

# Translate an existing SRT directly
uv run python main.py 20260123_video.srt --action translate -y

# Error: --action translate requires an SRT file
uv run python main.py video.mp4 --action translate  # exits with error
```

### The `--prompt-file` Flag

Use `--prompt-file` to provide extra instructions to the translation model. The file can contain anything: a glossary, style guide, tone preferences, or domain-specific terminology.

```bash
# Create a prompt file
cat > glossary.txt << 'EOF'
Translate technical terms as follows:
- "container" → 容器
- "pod" → Pod (keep English)
- "node" → 節點
Use formal tone throughout.
EOF

# Use it during translation
uv run python main.py video.mp4 --prompt-file glossary.txt
uv run python main.py existing.srt --prompt-file glossary.txt -y
```

The file contents are injected into every translation prompt sent to Ollama, so the model follows your instructions for every segment.

### Batch Scripting (`--preview` and `--subtitle`)

When processing many URLs in a script, the subtitle selection prompt blocks automation. Use `--preview` and `--subtitle` to handle this in two passes.

**`--subtitle N`** — Pre-select subtitle by index, skipping the interactive prompt:

```bash
# Download the first subtitle (index 1) without any prompt
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --subtitle 1 -y

# Force transcription, skipping the subtitle check entirely
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --subtitle 0 -y
```

**`--preview`** — Check subtitles interactively, then output the ready-to-run command(s) to stdout and exit:

```bash
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --preview
```

When the user **picks a subtitle to download** (no GPU needed), one command is output:
```
uv run python main.py 'https://youtube.com/watch?v=VIDEO_ID' --subtitle 1 -y
```

When the video needs **transcription** (GPU needed for Whisper), two commands are output so you can free VRAM between phases:
```
uv run python main.py 'https://youtube.com/watch?v=VIDEO_ID' --subtitle 0 --action transcribe
uv run python main.py '/path/to/20200101_VIDEO_ID.srt' -y --action translate
```

When **no subtitles are found**, you're prompted to transcribe or skip (useful for skipping videos that don't need transcription, e.g., Chinese videos you already understand).

Informational messages (subtitle list, prompt) go to stderr, so only the command(s) reach stdout.

**Two-pass workflow** for a batch of URLs:

```bash
# preview_run.sh — one URL per line
uv run python main.py "https://youtube.com/watch?v=AAA" --preview --model large
uv run python main.py "https://youtube.com/watch?v=BBB" --preview
uv run python main.py "https://youtube.com/watch?v=CCC" --preview --output ./subs
```

```bash
# Pass 1: interactive — user picks subtitle for each URL, real commands are saved
bash preview_run.sh > real_run.sh

# Pass 2: unattended — run all commands without prompts
bash real_run.sh
```

Notes:
- Transcription paths output **two commands**: Phase 1 uses `--action transcribe` (no translation), Phase 2 uses `--action translate` with the SRT file (with `-y`)
- Subtitle download paths (choice > 0) output **one command** with `-y` (no GPU used)
- Enter **`S` to skip** a video — no command is emitted, so that URL is absent from `real_run.sh`
- `--preview` never includes `--preview` in the output command(s)
- Non-default flags (`--model`, `--language`, `--output`, `--keep-audio`, `--stable`, `--vad`, `--prompt-file`) are preserved in output commands (`--prompt-file` is included in translate commands only)
- Informational output goes to **stderr**; only the command(s) go to **stdout** (enables clean piping)

#### Non-Interactive Preview (`--preview-opt`)

Use `--preview-opt` to drive preview mode without interactive prompts, enabling automation via external systems (e.g., a Telegram bot or API).

`--preview-opt` implies `--preview` automatically. Values:

| Value | Behavior |
|-------|----------|
| `L`   | Output a JSON object with available subtitles, video title, and channel |
| `S`   | Skip the video (emit nothing) |
| `0`   | Emit the transcribe command |
| `N`   | Emit the command for subtitle index N |

**List available subtitles (JSON):**
```bash
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --preview-opt L
```
```json
{"url": "https://youtube.com/watch?v=VIDEO_ID", "video_title": "Some Video", "channel": "SomeChannel", "subtitles": [{"index": 1, "lang": "en", "name": "English"}], "can_transcribe": true}
```

**Select a subtitle non-interactively:**
```bash
# Get command for subtitle index 1
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --preview-opt 1

# Get transcribe command
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --preview-opt 0

# Skip the video
uv run python main.py "https://youtube.com/watch?v=VIDEO_ID" --preview-opt S
```

## Output Format

### SRT Format (video.srt)
```
1
00:00:00,000 --> 00:00:02,500
Hello, world!

2
00:00:02,500 --> 00:00:05,000
This is a test.
```

## Supported Formats

- **Local Video Files**: Any format supported by ffmpeg (MP4, MKV, AVI, MOV, WebM, FLV, etc.)
- **URLs**: YouTube, Vimeo, Twitch, and [1000+ platforms via yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

## Development

### Running Tests

```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_transcriber.py -v
```

### Project Structure

```
whisper-subtitle-cli/
├── src/
│   ├── audio_extractor.py     # Extract audio from video
│   ├── transcriber.py         # AI transcription (Whisper)
│   ├── subtitle_writer.py     # Write SRT files
│   ├── translator.py          # Subtitle translation (Ollama)
│   └── video_downloader.py    # Download videos from URLs
├── tests/                     # Unit and integration tests
├── main.py                    # CLI entry point
├── config.json                # Ollama configuration
├── pyproject.toml             # uv dependencies
├── .python-version            # Python version (used by uv)
└── README.md                  # Documentation
```

## Troubleshooting

Run system diagnostics to check GPU, CUDA, ffmpeg, and Ollama:
```bash
uv run python main.py --check-system
```

**Common issues:**
- **Slow transcription**: Use `--model tiny` or `--model base`, or specify `--language` to skip auto-detection
- **Out of memory**: Use a smaller model (see Whisper Model Options above)
- **Subtitles out of sync**: Use `--stable` flag for better timestamps (requires `uv sync --extra stable`)
- **Hallucinations in silence**: Add `--vad` flag with `--stable` to enable Voice Activity Detection
- **Translation fails**: Ensure Ollama is running (`ollama serve`) and model is downloaded (`ollama pull translategemma:4b`)

See [INSTALL.md](INSTALL.md#common-issues) for detailed troubleshooting.

## License

This project uses:
- openai-whisper (MIT License)
- ffmpeg (LGPL/GPL)

## Acknowledgments

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) - AI speech recognition
- [stable-ts](https://github.com/jianfch/stable-ts) - Improved timestamp accuracy (optional)
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing
