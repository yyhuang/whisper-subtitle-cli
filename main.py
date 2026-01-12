#!/usr/bin/env python
"""
Video Subtitle Extractor

Extract subtitles from video files using AI transcription (Whisper).
Outputs both SRT format (for video players) and plain text (for reading).
"""

import click
import os
import sys
from pathlib import Path
from datetime import datetime

from src.audio_extractor import AudioExtractor
from src.transcriber import Transcriber
from src.subtitle_writer import SubtitleWriter
from src.video_downloader import VideoDownloader, is_url


class VideoInput(click.ParamType):
    """Custom Click parameter type that accepts both file paths and URLs."""
    name = "video_input"

    def convert(self, value, param, ctx):
        # If it's a URL, just return it (we'll validate it during download)
        if is_url(value):
            return value

        # If it's a file path, check it exists
        path = Path(value)
        if not path.exists():
            self.fail(f"File not found: {value}", param, ctx)
        return str(path.resolve())


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


@click.command()
@click.argument('video_input', type=VideoInput())
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
def main(video_input, model, language, output, keep_audio):
    """
    Extract subtitles from VIDEO_INPUT (file path or URL) using AI transcription.

    Generates two files:
    - .srt file (for video players with timestamps)
    - .txt file (plain text for easy reading)

    Example:
        python main.py video.mp4
        python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
        python main.py video.mp4 --model medium --language en
    """
    try:
        # Step 0: Handle URL vs file path
        if is_url(video_input):
            click.echo(f"Detected URL: {video_input}")

            # Check for available subtitles first
            click.echo("\nChecking for available subtitles...")
            downloader = VideoDownloader(download_dir="/tmp")

            subtitles = downloader.get_available_subtitles(video_input)

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

                    # Get video title for output naming
                    video_info = downloader.get_video_info(video_input)
                    base_name = VideoDownloader.sanitize_filename(video_info['title'])

                    # Get date prefix from video upload date
                    date_prefix = get_date_prefix(upload_date=video_info.get('upload_date'))

                    # Determine output directory
                    if output:
                        output_dir = Path(output)
                        output_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        output_dir = Path.cwd()

                    srt_path = output_dir / f"{date_prefix}_{base_name}.srt"
                    timestamped_txt_path = output_dir / f"{date_prefix}_{base_name}.timestamped.txt"

                    # Download subtitle
                    downloader.download_subtitle(video_input, selected_lang, str(srt_path))

                    # Parse the downloaded SRT and create timestamped TXT
                    writer = SubtitleWriter()
                    segments = writer.parse_srt(str(srt_path))
                    writer.write_timestamped_txt(segments, str(timestamped_txt_path))

                    click.echo(f"✓ Subtitle downloaded: {srt_path}")
                    click.echo(f"✓ Timestamped text created: {timestamped_txt_path}")
                    click.echo("\n✅ Done! Subtitle download complete.")
                    click.echo(f"\nOutput files:")
                    click.echo(f"  • {srt_path.name} (for video playback)")
                    click.echo(f"  • {timestamped_txt_path.name} (easy to copy/translate)")
                    return  # Exit early, skip transcription

            # No subtitles or user chose to transcribe
            if not subtitles:
                click.echo("No manual subtitles available. Transcribing video...")
            else:
                click.echo("\nProceeding with video transcription...")

            click.echo("\n[0/4] Downloading video...")

            video_info = downloader.download(video_input, quiet=False)

            video_path = Path(video_info['file_path'])
            video_title = video_info['title']
            base_name = VideoDownloader.sanitize_filename(video_title)

            # Get date prefix from video upload date
            date_prefix = get_date_prefix(upload_date=video_info.get('upload_date'))

            click.echo(f"✓ Downloaded: {video_title}")
            click.echo(f"  Duration: {video_info['duration']:.1f}s")
            click.echo(f"✓ Saved to /tmp (OS will clean up automatically)")
        else:
            # Existing file path behavior
            video_path = Path(video_input).resolve()
            base_name = video_path.stem

            # Get date prefix from file modification date
            date_prefix = get_date_prefix(file_path=video_path)

            click.echo(f"Processing: {video_path.name}")

        # Determine output directory
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = video_path.parent

        # Generate output file paths with date prefix
        audio_path = output_dir / f"{date_prefix}_{base_name}.wav"
        srt_path = output_dir / f"{date_prefix}_{base_name}.srt"
        txt_path = output_dir / f"{date_prefix}_{base_name}.txt"
        timestamped_txt_path = output_dir / f"{date_prefix}_{base_name}.timestamped.txt"

        # Step 1: Extract audio
        step_num = "[1/4]" if is_url(video_input) else "[1/3]"
        click.echo(f"\n{step_num} Extracting audio from video...")
        extractor = AudioExtractor()
        extractor.extract_audio(str(video_path), str(audio_path))
        click.echo(f"✓ Audio extracted to: {audio_path.name}")

        # Step 2: Transcribe audio
        step_num = "[2/4]" if is_url(video_input) else "[2/3]"
        click.echo(f"\n{step_num} Transcribing audio (model: {model})...")
        if language:
            click.echo(f"      Language: {language}")
        else:
            click.echo("      Language: auto-detect")

        transcriber = Transcriber(model_size=model)
        segments = transcriber.transcribe(str(audio_path), language=language)
        click.echo(f"✓ Transcription complete ({len(segments)} segments)")

        # Step 3: Write subtitle files
        step_num = "[3/4]" if is_url(video_input) else "[3/3]"
        click.echo(f"\n{step_num} Writing subtitle files...")
        writer = SubtitleWriter()

        writer.write_srt(segments, str(srt_path))
        click.echo(f"✓ SRT file created: {srt_path}")

        writer.write_txt(segments, str(txt_path))
        click.echo(f"✓ Text file created: {txt_path}")

        writer.write_timestamped_txt(segments, str(timestamped_txt_path))
        click.echo(f"✓ Timestamped text created: {timestamped_txt_path}")

        # Clean up audio file if not keeping it
        if not keep_audio and audio_path.exists():
            audio_path.unlink()
            click.echo(f"\n✓ Cleaned up temporary audio file")
        elif keep_audio:
            click.echo(f"\n✓ Audio file kept: {audio_path}")

        click.echo("\n✅ Done! Subtitle extraction complete.")
        click.echo(f"\nOutput files:")
        click.echo(f"  • {srt_path.name} (for video playback)")
        click.echo(f"  • {txt_path.name} (for reading)")
        click.echo(f"  • {timestamped_txt_path.name} (easy to copy/translate)")

    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
