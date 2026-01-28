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

Which subtitle would you like to download? [0]: 1

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
    "keep_alive": "10m"
  },
  "output": {
    "directory": "/path/to/subtitles"
  }
}
```

- `ollama.model`: Ollama model for translation
- `ollama.batch_size`: Segments per API call (higher = better context, more memory)
- `ollama.keep_alive`: How long model stays loaded (`"10m"`, `"1h"`, `"-1"` for indefinitely)
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
# Which subtitle would you like to download? [0]:    ← still prompts here
#
# (after you choose, auto-translates without further prompts)
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
