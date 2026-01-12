# whisper-subtitle-cli

Extract subtitles from video files or YouTube URLs using AI transcription (OpenAI Whisper). Generates both SRT files for video playback and plain text files for easy reading.

> **Note**: This project was built with AI assistance.

## Features

- **YouTube & URL Support**: Process videos from YouTube, Vimeo, Twitch, and [1000+ platforms](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)
- **Subtitle Download**: Automatically download existing YouTube subtitles (much faster than transcription)
- **AI-Powered Transcription**: Uses Faster Whisper for accurate speech-to-text
- **Dual Output**: Creates both SRT (with timestamps) and TXT (plain text) files
- **Multiple Languages**: Auto-detects language or accepts manual specification
- **Flexible Models**: Choose from 5 model sizes balancing speed vs accuracy

## Installation

**Prerequisites:**
- Python 3.11 or 3.12
- Poetry
- ffmpeg

```bash
# Install ffmpeg
brew install ffmpeg  # macOS
# or: sudo apt install ffmpeg  # Linux

# Install project dependencies
poetry install --no-root
```

**New to Python or Poetry?** See [INSTALL.md](INSTALL.md) for detailed step-by-step instructions and troubleshooting.

## Usage

### Basic Usage

Extract subtitles from a local video file:

```bash
python main.py video.mp4
```

Extract subtitles from a YouTube URL:

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

This creates three files with date prefix (YYYYMMDD format):
- `YYYYMMDD_video.srt` or `YYYYMMDD_Video_Title.srt` - Subtitle file with timestamps (for video players)
- `YYYYMMDD_video.txt` or `YYYYMMDD_Video_Title.txt` - Plain text transcript (for reading)
- `YYYYMMDD_video.timestamped.txt` or `YYYYMMDD_Video_Title.timestamped.txt` - Timestamped text (for translation)

**Note**:
- Filenames use underscores instead of spaces for easier command line usage (e.g., `My Video.mp4` → `YYYYMMDD_My_Video.srt`)
- When processing URLs, videos are downloaded to your system's temporary directory and cleaned up automatically by your OS

### Advanced Options

```bash
# Use a different model size
python main.py video.mp4 --model small

# Specify the language (faster than auto-detect)
python main.py video.mp4 --language en

# Save output to a specific directory
python main.py video.mp4 --output ./subtitles

# Keep the extracted audio file
python main.py video.mp4 --keep-audio

# YouTube URLs work the same way
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
python main.py "https://youtu.be/VIDEO_ID"  # Short format

# Other platforms (Vimeo, Twitch, etc.)
python main.py "https://vimeo.com/123456"
```

### Subtitle Download (YouTube)

When you provide a YouTube URL, the tool automatically checks for existing subtitles:

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

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

### Model Options

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

### Plain Text Format (video.txt)
```
Hello, world!

This is a test.
```

## Supported Formats

- **Local Video Files**: Any format supported by ffmpeg (MP4, MKV, AVI, MOV, WebM, FLV, etc.)
- **URLs**: YouTube, Vimeo, Twitch, and [1000+ platforms via yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_transcriber.py -v
```

### Project Structure

```
whisper-subtitle-cli/
├── src/
│   ├── audio_extractor.py     # Extract audio from video
│   ├── transcriber.py          # AI transcription
│   ├── subtitle_writer.py      # Write SRT and TXT files
│   └── video_downloader.py     # Download videos from URLs
├── tests/                      # Unit and integration tests
│   ├── test_audio_extractor.py
│   ├── test_transcriber.py
│   ├── test_subtitle_writer.py
│   ├── test_video_downloader.py
│   └── test_main_integration.py
├── main.py                     # CLI entry point
├── pyproject.toml              # Poetry dependencies
├── README.md                   # User documentation
└── CLAUDE.md                   # Project documentation
```

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg using your package manager (see Requirements section).

### "No module named 'src'"
Make sure you're running the script from the project root directory.

### Slow transcription
Use a smaller model (`--model tiny` or `--model base`) or specify the language (`--language en`) to skip auto-detection.

### Out of memory
Use a smaller model. Try `--model base` or `--model tiny` for lower memory usage (see Model Options table above for sizes).

## License

This project uses:
- faster-whisper (MIT License)
- OpenAI Whisper models (MIT License)
- ffmpeg (LGPL/GPL)

## Acknowledgments

Built with:
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper models
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing
