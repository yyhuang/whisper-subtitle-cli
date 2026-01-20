#!/usr/bin/env python
"""
Video Subtitle Extractor

Extract subtitles from video files using AI transcription (Whisper).
Outputs SRT format for video players.
"""

import click
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

from src.audio_extractor import AudioExtractor
from src.transcriber import Transcriber
from src.subtitle_writer import SubtitleWriter
from src.video_downloader import VideoDownloader, is_url
from src.translator import OllamaTranslator, load_config


class DataInput(click.ParamType):
    """Custom Click parameter type that accepts file paths, URLs, or SRT files."""
    name = "data_input"

    def convert(self, value, param, ctx):
        # If it's a URL, just return it (we'll validate it during download)
        if is_url(value):
            return value

        # If it's a file path, check it exists
        path = Path(value)
        if not path.exists():
            self.fail(f"File not found: {value}", param, ctx)
        return str(path.resolve())


def is_srt_file(path: str) -> bool:
    """Check if the input is an SRT subtitle file."""
    return path.lower().endswith('.srt')


def get_date_prefix(upload_date: str = None, file_path: Path = None) -> str:
    """
    Get date prefix for filename in YYYYMMDD format.

    Args:
        upload_date: Video upload date in YYYYMMDD format (from yt-dlp)
        file_path: Local file path (used to get modification date if upload_date is None)

    Returns:
        Date prefix string in YYYYMMDD format
    """
    if upload_date:
        # Use upload date from video metadata (already in YYYYMMDD format)
        return upload_date
    elif file_path and file_path.exists():
        # Fall back to file modification date for local files
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime('%Y%m%d')
    else:
        # Fall back to current date
        return datetime.now().strftime('%Y%m%d')


def create_bilingual_segments(original_segments, translated_segments):
    """
    Create bilingual segments with original and translated text.

    Each segment has 2 lines:
    - Line 1: Original text (multiple lines joined with " / ")
    - Line 2: Translated text (multiple lines joined with " / ")

    Args:
        original_segments: List of original subtitle segments
        translated_segments: List of translated subtitle segments

    Returns:
        List of bilingual segments with both texts
    """
    bilingual = []
    for orig, trans in zip(original_segments, translated_segments):
        # Join multiple lines with " / " for compact display
        orig_text = orig['text'].replace('\n', ' / ')
        trans_text = trans['text'].replace('\n', ' / ')
        bilingual.append({
            'start': orig['start'],
            'end': orig['end'],
            'text': f"{orig_text}\n{trans_text}"
        })
    return bilingual


def translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name):
    """
    Handle subtitle translation workflow.

    Args:
        segments: List of subtitle segments
        srt_path: Path to the original SRT file
        output_dir: Directory for output files
        date_prefix: Date prefix for filenames
        base_name: Base name for output files
    """
    if not click.confirm('\nWould you like to translate the subtitles?', default=False):
        return

    source_lang = click.prompt('Source language', default='English')
    target_lang = click.prompt('Target language')
    want_bilingual = click.confirm('Create bilingual subtitle (original + translation)?', default=False)

    # Load config and show model info
    config = load_config()
    model_name = config['ollama']['model']
    click.echo(f"\nUsing Ollama model: {model_name}")

    # Check Ollama connection
    translator = OllamaTranslator()
    if not translator.check_connection():
        click.echo(
            f"\n❌ Cannot connect to Ollama at {config['ollama']['base_url']}. "
            "Make sure Ollama is running (ollama serve).",
            err=True
        )
        return

    # Translate with progress indicator
    click.echo(f"\nTranslating {len(segments)} segments...")

    def progress_callback(current, total):
        click.echo(f"  Translating segment {current}/{total}...", nl=False)
        click.echo('\r', nl=False)

    try:
        translated_segments = translator.translate_segments(
            segments,
            source_lang,
            target_lang,
            progress_callback=progress_callback
        )
        click.echo()  # New line after progress

        # Write translated SRT
        translated_srt_path = output_dir / f"{date_prefix}_{base_name}.{target_lang}.srt"
        writer = SubtitleWriter()
        writer.write_srt(translated_segments, str(translated_srt_path))

        click.echo(f"✓ Translated SRT created: {translated_srt_path.name}")

        # Create bilingual output if requested
        if want_bilingual:
            bilingual_segments = create_bilingual_segments(segments, translated_segments)
            bilingual_srt_path = output_dir / f"{date_prefix}_{base_name}.bilingual.srt"
            writer.write_srt(bilingual_segments, str(bilingual_srt_path))
            click.echo(f"✓ Bilingual SRT created: {bilingual_srt_path.name}")

    except ConnectionError as e:
        click.echo(f"\n❌ Connection error: {e}", err=True)
    except RuntimeError as e:
        click.echo(f"\n❌ Translation error: {e}", err=True)


def handle_srt_translation(srt_path: str, output: str = None):
    """
    Handle translation of an existing SRT file.

    Args:
        srt_path: Path to the SRT file
        output: Optional output directory
    """
    srt_file = Path(srt_path)
    click.echo(f"SRT file detected: {srt_file.name}")
    click.echo("Skipping download/transcription, going directly to translation.\n")

    # Parse SRT file
    writer = SubtitleWriter()
    try:
        segments = writer.parse_srt(srt_path)
    except Exception as e:
        click.echo(f"❌ Failed to parse SRT file: {e}", err=True)
        sys.exit(1)

    click.echo(f"✓ Parsed {len(segments)} segments from SRT file")

    # Determine output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = srt_file.parent

    # Extract base name and date prefix from filename
    # Expected format: YYYYMMDD_name.srt or YYYYMMDD_name.Language.srt
    filename = srt_file.stem  # Remove .srt extension

    # Check if filename starts with date prefix
    if len(filename) >= 8 and filename[:8].isdigit():
        date_prefix = filename[:8]
        rest = filename[9:] if len(filename) > 9 else filename[8:]  # Skip underscore
    else:
        date_prefix = get_date_prefix(file_path=srt_file)
        rest = filename

    # Remove language suffix if present (e.g., "video.Chinese" -> "video")
    if '.' in rest:
        base_name = rest.rsplit('.', 1)[0]
    else:
        base_name = rest

    # Go directly to translation
    translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name)

    click.echo("\n✅ Done!")


@click.command()
@click.argument('data_input', type=DataInput())
@click.option(
    '--model',
    default='medium',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large'], case_sensitive=False),
    help='Whisper model size (larger = more accurate but slower)'
)
@click.option(
    '--language',
    default=None,
    help='Language code (e.g., en, zh, es). Auto-detect if not specified.'
)
@click.option(
    '--output',
    '-o',
    default=None,
    type=click.Path(),
    help='Output directory (default: same as video file)'
)
@click.option(
    '--keep-audio',
    is_flag=True,
    default=False,
    help='Keep the extracted audio file (WAV)'
)
def main(data_input, model, language, output, keep_audio):
    """
    Extract subtitles from DATA_INPUT (file path, URL, or SRT file) using AI transcription.

    Generates .srt file for video players with timestamps.

    \b
    Examples:
      python main.py video.mp4
      python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
      python main.py video.mp4 --model medium --language en
      python main.py existing.srt
    """
    try:
        # Handle SRT file input - skip to translation
        if is_srt_file(data_input):
            handle_srt_translation(data_input, output)
            return

        # Step 0: Handle URL vs file path
        is_url_input = False
        temp_dir_path = None

        if is_url(data_input):
            is_url_input = True
            click.echo(f"Detected URL: {data_input}")

            # Check for available subtitles first
            click.echo("\nChecking for available subtitles...")
            downloader = VideoDownloader()  # Uses system temp directory by default
            temp_dir_path = downloader.download_dir

            subtitles = downloader.get_available_subtitles(data_input)

            if subtitles:
                # List available subtitles
                click.echo("\nAvailable subtitles:")
                subtitle_list = list(subtitles.items())
                for idx, (lang_code, info) in enumerate(subtitle_list, 1):
                    click.echo(f"  {idx}. {info['name']} ({lang_code})")
                click.echo(f"  0. Transcribe video instead")

                # User selects subtitle
                choice = click.prompt(
                    "\nWhich subtitle would you like to download?",
                    type=click.IntRange(0, len(subtitle_list)),
                    default=0
                )

                if choice > 0:
                    # Download selected subtitle
                    selected_lang = subtitle_list[choice - 1][0]
                    selected_name = subtitle_list[choice - 1][1]['name']

                    click.echo(f"\nDownloading {selected_name} subtitle...")

                    # Get video ID for output naming
                    video_info = downloader.get_video_info(data_input)
                    base_name = video_info['video_id']

                    # Get date prefix from video upload date
                    date_prefix = get_date_prefix(upload_date=video_info.get('upload_date'))

                    # Determine output directory (use temp directory for URL downloads)
                    if output:
                        output_dir = Path(output)
                        output_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        output_dir = downloader.download_dir  # Use temp directory

                    srt_path = output_dir / f"{date_prefix}_{base_name}.srt"

                    # Download subtitle
                    downloader.download_subtitle(data_input, selected_lang, str(srt_path))

                    # Parse the downloaded SRT for translation
                    writer = SubtitleWriter()
                    segments = writer.parse_srt(str(srt_path))

                    click.echo(f"✓ Subtitle downloaded: {srt_path}")

                    # Offer translation
                    translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name)

                    click.echo("\n✅ Done! Subtitle download complete.")
                    click.echo(f"\nOutput files saved to:")
                    click.echo(f"  {output_dir}")
                    return  # Exit early, skip transcription

            # No subtitles or user chose to transcribe
            if not subtitles:
                click.echo("No manual subtitles available. Transcribing video...")
            else:
                click.echo("\nProceeding with video transcription...")

            click.echo("\n[0/4] Downloading video...")

            video_info = downloader.download(data_input, quiet=False)

            video_path = Path(video_info['file_path'])
            video_title = video_info['title']
            base_name = video_info['video_id']

            # Get date prefix from video upload date
            date_prefix = get_date_prefix(upload_date=video_info.get('upload_date'))

            click.echo(f"✓ Downloaded: {video_title}")
            click.echo(f"  Duration: {video_info['duration']:.1f}s")
            click.echo(f"✓ Saved to temp directory (OS will clean up automatically)")
        else:
            # Existing file path behavior
            video_path = Path(data_input).resolve()
            # Sanitize local filename to replace spaces with underscores
            base_name = VideoDownloader.sanitize_filename(video_path.stem)

            # Get date prefix from file modification date
            date_prefix = get_date_prefix(file_path=video_path)

            click.echo(f"Processing: {video_path.name}")

        # Determine output directory
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # For local files, use file's directory; for URLs, use temp directory
            output_dir = video_path.parent if not is_url_input else video_path.parent

        # Generate output file paths with date prefix
        audio_path = output_dir / f"{date_prefix}_{base_name}.wav"
        srt_path = output_dir / f"{date_prefix}_{base_name}.srt"

        # Step 1: Extract audio
        step_num = "[1/4]" if is_url(data_input) else "[1/3]"
        click.echo(f"\n{step_num} Extracting audio from video...")
        extractor = AudioExtractor()
        extractor.extract_audio(str(video_path), str(audio_path))
        click.echo(f"✓ Audio extracted to: {audio_path.name}")

        # Step 2: Transcribe audio
        step_num = "[2/4]" if is_url(data_input) else "[2/3]"
        click.echo(f"\n{step_num} Transcribing audio (model: {model})...")
        if language:
            click.echo(f"      Language: {language}")
        else:
            click.echo("      Language: auto-detect")

        transcriber = Transcriber(model_size=model)
        segments = transcriber.transcribe(str(audio_path), language=language)
        click.echo(f"✓ Transcription complete ({len(segments)} segments)")

        # Step 3: Write subtitle files
        step_num = "[3/4]" if is_url(data_input) else "[3/3]"
        click.echo(f"\n{step_num} Writing subtitle file...")
        writer = SubtitleWriter()

        writer.write_srt(segments, str(srt_path))
        click.echo(f"✓ SRT file created: {srt_path}")

        # Step 4: Offer translation (for URL inputs this becomes [4/4])
        translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name)

        # Clean up audio file if not keeping it
        if not keep_audio and audio_path.exists():
            audio_path.unlink()
            click.echo(f"\n✓ Cleaned up temporary audio file")
        elif keep_audio:
            click.echo(f"\n✓ Audio file kept: {audio_path}")

        click.echo("\n✅ Done! Subtitle extraction complete.")
        click.echo(f"\nOutput files saved to:")
        click.echo(f"  {output_dir}")

    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
