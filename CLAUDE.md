# Video Subtitle Extractor

## Project Overview
CLI tool to generate subtitles from video/audio files using OpenAI Whisper AI model.

## Requirements
- Extract audio from video files or YouTube URLs
- Transcribe audio using Whisper AI (speech-to-text)
- Output SRT format (with timestamps)
- Output plain text format (for easy reading)
- Support common video formats: MP4, MKV, AVI, MOV, WebM
- Support YouTube and 1000+ other video platforms via URLs

## Technical Stack
- **Python 3.11+**
- **Poetry** for package management
- **Faster Whisper** for AI transcription
- **ffmpeg** (system dependency) for audio extraction
- **yt-dlp** for downloading videos from URLs

## Output Files
For input `video.mp4`:
- `video.srt` - SRT subtitle file with timestamps
- `video.txt` - Plain text transcript without timestamps

## Configuration Options
- Whisper model size: tiny, base, small, medium, large (default: base)
- Language: auto-detect or specify (e.g., en, zh, es)
- Output directory

## Project Structure
```
video-subtitle/
├── src/
│   ├── transcriber.py      # Whisper transcription logic
│   ├── subtitle_writer.py  # SRT and TXT file generation
│   ├── audio_extractor.py  # Audio extraction from video
│   └── video_downloader.py # YouTube/URL video downloading
├── main.py                 # CLI entry point
├── tests/
│   ├── test_transcriber.py
│   ├── test_subtitle_writer.py
│   ├── test_audio_extractor.py
│   ├── test_video_downloader.py
│   └── test_main_integration.py
├── pyproject.toml
└── CLAUDE.md
```

## Dependencies
- faster-whisper
- ffmpeg-python (Python wrapper)
- click (for CLI interface)
- yt-dlp (for downloading videos from URLs)
- pytest (for testing)

## Current Status
✅ **Implementation Complete**

All core features implemented and tested:
- ✅ Audio extraction from video files
- ✅ YouTube URL support (and 1000+ other platforms via yt-dlp)
- ✅ Subtitle download from YouTube (human-made subtitles only)
- ✅ AI transcription using Faster Whisper
- ✅ SRT subtitle file generation
- ✅ Plain text file generation
- ✅ CLI interface with options
- ✅ 51/51 unit tests passing

## Usage

### Basic Usage
```bash
# Extract subtitles from a local video file
python main.py video.mp4

# Extract subtitles from a YouTube URL
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Short YouTube URL format
python main.py "https://youtu.be/VIDEO_ID"

# This creates:
# - video.srt (subtitle file with timestamps)
# - video.txt (plain text for reading)
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

# Downloaded videos are saved to /tmp (OS cleans up automatically)
# Subtitles are named after the video title
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
- Only creates SRT file when downloading subtitles (no TXT file)

### Available Models
- **tiny**: Fastest, least accurate (~39MB)
- **base**: Good balance (default, ~140MB)
- **small**: Better accuracy (~470MB)
- **medium**: High accuracy (~1.5GB)
- **large**: Best accuracy (~2.9GB)

## Testing
```bash
# Run all tests
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_transcriber.py -v
```

## Next Steps (Optional Enhancements)
- Add support for batch processing multiple videos/URLs
- Add progress bars for long videos
- Support for additional subtitle formats (VTT, ASS)
- GPU acceleration support (CUDA)
- Playlist support (download and process all videos from a playlist)
- Web interface
