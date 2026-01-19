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
- **Poetry** for package management
- **Faster Whisper** for AI transcription
- **ffmpeg** (system dependency) for audio extraction
- **yt-dlp** for downloading videos from URLs
- **Ollama** (local) for subtitle translation

## Output Files
All subtitle files include a date prefix (YYYYMMDD format) and use underscores instead of spaces for easier command line usage.

**Generated Files**:
- `YYYYMMDD_video_title.srt` - SRT subtitle file with timestamps (for video playback)
- `YYYYMMDD_video_title.{Language}.srt` - Translated subtitle file (when translation is used)

**Output Location**:
- **YouTube/URL inputs**: Saved to system temp directory (platform-specific)
  - macOS/Linux: `/tmp/`
  - Windows: `%TEMP%`
  - OS automatically cleans up temp files
- **Local video files**: Saved to same directory as the video file
- **Custom output**: Use `--output` flag to specify custom directory

**Filename Rules**:
- Spaces are replaced with underscores (`_`)
- Invalid characters (`/ \ : * ? " < > |`) are replaced with underscores
- Leading and trailing underscores are removed
- Makes files easier to work with in command line

**Date Logic**:
- YouTube URLs: Uses video's original upload date
- Local files: Uses file's modification date
- Fallback: Uses current date if neither is available

## Configuration

### Ollama Settings (`config.json`)
Configure the Ollama model and API URL in `config.json` at the project root:

```json
{
  "ollama": {
    "model": "qwen2.5:7b",
    "base_url": "http://localhost:11434",
    "batch_size": 50,
    "keep_alive": "10m"
  }
}
```

- **model**: The Ollama model to use for translation (default: `qwen2.5:7b`)
- **base_url**: Ollama API URL, can point to remote Ollama server (default: `http://localhost:11434`)
- **batch_size**: Number of segments to translate per API call (default: `50`)
- **keep_alive**: How long to keep the model loaded in memory after a request (default: `10m`)
  - `"5m"`, `"10m"`, `"1h"` - duration values
  - `"-1"` - keep loaded indefinitely
  - `"0"` - unload immediately after request

**Note:** Only Ollama API is supported. Other APIs (OpenAI, Claude, etc.) are not compatible.

### CLI Options
- Whisper model size: tiny, base, small, medium, large (default: medium)
- Language: auto-detect or specify (e.g., en, zh, es)
- Output directory

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
│   ├── test_audio_extractor.py
│   ├── test_video_downloader.py
│   ├── test_translator.py
│   └── test_main_integration.py
├── pyproject.toml
├── .python-version         # Python version for pyenv
└── CLAUDE.md
```

## Dependencies
- faster-whisper
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
- ✅ AI transcription using Faster Whisper
- ✅ SRT subtitle file generation
- ✅ Subtitle translation via local Ollama API (batch processing with recursive retry)
- ✅ CLI interface with options
- ✅ Temp directory output for URL downloads

## Usage

### Basic Usage
```bash
# Extract subtitles from a local video file
python main.py video.mp4
# Creates: YYYYMMDD_video.srt (subtitle file with timestamps)

# Extract subtitles from a YouTube URL
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
# Creates files in system temp directory (/tmp/ on macOS/Linux)

# Short YouTube URL format
python main.py "https://youtu.be/VIDEO_ID"

# Translate an existing SRT file (skip download/transcription)
python main.py existing_subtitle.srt
# Goes directly to translation prompt
```

### Advanced Options
```bash
# Use a larger model for better accuracy (slower)
python main.py video.mp4 --model medium

# Specify language (faster than auto-detect)
python main.py "https://youtube.com/watch?v=VIDEO_ID" --language en

# Save to a different directory
python main.py video.mp4 --output ./subtitles

# Keep the extracted audio file
python main.py video.mp4 --keep-audio
```

### YouTube URL Support
```bash
# Process any YouTube video
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Works with playlists, shorts, and other platforms
python main.py "https://vimeo.com/123456"

# Both downloaded videos and subtitle files are saved to system temp directory
# OS cleans up temp files automatically
# Subtitles are named after the video title with date prefix
```

### Subtitle Download (YouTube)
When processing a YouTube URL, the tool automatically checks for existing subtitles:

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Output:
Checking for available subtitles...

Available subtitles:
  1. English (en)
  2. Spanish (es)
  3. French (fr)
  0. Transcribe video instead

Which subtitle would you like to download? [0]: 1

Downloading English subtitle...
✓ Subtitle downloaded: Video_Title.srt
✅ Done!
```

**Features**:
- Only shows human-made subtitles (auto-generated are filtered out)
- Much faster than transcription (~1 second vs 30-120 seconds)
- If no subtitles exist, automatically falls back to transcription
- Option 0 allows transcription even when subtitles are available

### Subtitle Translation (Ollama)
After creating or downloading subtitles, you can translate them using a local Ollama model:

```bash
python main.py video.mp4

# Output:
✓ SRT file created: 20260119_video.srt

Would you like to translate the subtitles? [y/N]: y
Source language [English]: English
Target language: Chinese

Using Ollama model: qwen2.5:7b

Translating 150 segments...
  Translating segment 150/150...
✓ Translated SRT created: 20260119_video.Chinese.srt

✅ Done! Subtitle extraction complete.
```

**Requirements**:
- Ollama must be running locally (`ollama serve`)
- Model must be pulled (`ollama pull qwen2.5:7b`)

**How Translation Works**:
- Segments are translated in **batches** (default: 50 segments per batch) for better context
- Batch translation provides better quality than one-by-one translation
- If a batch fails, it's automatically **split in half** and retried
- This continues recursively until the problematic segment is isolated
- Timestamps are preserved by our code - the LLM never sees them

**Configuration**:
Edit `config.json` to change the model, API URL, or batch size:
```json
{
  "ollama": {
    "model": "llama3:8b",
    "base_url": "http://localhost:11434",
    "batch_size": 50
  }
}
```

### Available Models
- **tiny**: Fastest, least accurate (~39MB)
- **base**: Good balance (~140MB)
- **small**: Better accuracy (~470MB)
- **medium**: High accuracy (default, ~1.5GB)
- **large**: Best accuracy (~2.9GB)

## Testing
```bash
# Run all tests
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_transcriber.py -v

# Run translator tests
poetry run pytest tests/test_translator.py -v
```

## Next Steps (Optional Enhancements)
- Add support for batch processing multiple videos/URLs
- Add progress bars for long videos
- Support for additional subtitle formats (VTT, ASS)
- GPU acceleration support (CUDA)
- Playlist support (download and process all videos from a playlist)
- Web interface
