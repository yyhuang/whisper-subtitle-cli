#!/usr/bin/env python
"""
Video Subtitle Extractor

Extract subtitles from video files using AI transcription (Whisper).
Outputs SRT format for video players.
"""

import click
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime

from src.audio_extractor import AudioExtractor
from src.transcriber import Transcriber
from src.subtitle_writer import SubtitleWriter
from src.video_downloader import VideoDownloader, is_url
from src.translator import OllamaTranslator, load_config, parse_language

# =============================================================================
# CUDA Version Configuration
# =============================================================================
# This mapping defines supported PyTorch CUDA versions and their requirements.
# torch 2.5.1 is pinned because newer versions (2.6.0+) have compatibility issues with Whisper.
#
# How to add a new CUDA version:
# 1. Check minimum driver version at:
#    https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
# 2. Add entry to CUDA_VERSIONS dict below
# 3. Update INSTALL.md with the new override command
#
# Note: cu124+ not included because torch 2.5.1 is the max stable version for Whisper,
# and it's available on cu121. No benefit to using cu124 with the same torch version.
CUDA_VERSIONS = {
    "cu118": {"min_driver": 520, "cuda": "11.8", "torch": "2.5.1+cu118"},
    "cu121": {"min_driver": 525, "cuda": "12.1", "torch": "2.5.1+cu121"},
}

# Default CUDA version for Windows (used in pyproject.toml)
# Change this when updating the default, and update pyproject.toml to match
DEFAULT_CUDA = "cu121"


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


def get_output_directory(cli_output: str, config: dict, default_path: Path) -> Path:
    """
    Determine output directory with priority: CLI argument > config > default.

    Args:
        cli_output: Output directory from CLI --output flag (or None)
        config: Configuration dictionary from load_config()
        default_path: Default path to use if neither CLI nor config specifies

    Returns:
        Path object for the output directory
    """
    # Priority 1: CLI argument
    if cli_output:
        output_dir = Path(cli_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Priority 2: Config file
    config_output = config.get('output', {}).get('directory')
    if config_output:
        output_dir = Path(config_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Priority 3: Default
    return default_path


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


def translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name, config, yes=False, language_name=None):
    """
    Handle subtitle translation workflow.

    Args:
        segments: List of subtitle segments
        srt_path: Path to the original SRT file
        output_dir: Directory for output files
        date_prefix: Date prefix for filenames
        base_name: Base name for output files
        config: Configuration dictionary from load_config()
        yes: If True, skip prompts and use defaults
        language_name: Source language name from --language flag (used with --yes)

    Returns:
        Translation time in seconds, or None if translation was skipped
    """
    if yes:
        # --yes: auto-accept translation with defaults.
        # Source language uses --language name if provided, otherwise "English".
        # Target is always "Chinese", bilingual is always enabled.
        source_lang = language_name or 'English'
        target_lang = 'Chinese'
        want_bilingual = True
    else:
        if not click.confirm('\nWould you like to translate the subtitles?', default=True):
            return None

        source_lang = click.prompt('Source language', default='English')
        target_lang = click.prompt('Target language', default='Chinese')

        # Validate user input for source and target languages
        for lang_input, label in [(source_lang, 'Source'), (target_lang, 'Target')]:
            parsed = parse_language(lang_input)
            if parsed is None:
                click.echo(f"⚠ Warning: '{lang_input}' not recognized, using as-is.", err=True)
        # Convert to canonical names if recognized
        source_parsed = parse_language(source_lang)
        if source_parsed:
            source_lang = source_parsed[0]
        target_parsed = parse_language(target_lang)
        if target_parsed:
            target_lang = target_parsed[0]

        want_bilingual = click.confirm('Create bilingual subtitle (original + translation)?', default=True)

    # Show model info
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
        translation_start = time.time()
        translated_segments = translator.translate_segments(
            segments,
            source_lang,
            target_lang,
            progress_callback=progress_callback
        )
        translation_time = time.time() - translation_start
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

        return translation_time

    except ConnectionError as e:
        click.echo(f"\n❌ Connection error: {e}", err=True)
        return None
    except RuntimeError as e:
        click.echo(f"\n❌ Translation error: {e}", err=True)
        return None


def handle_srt_translation(srt_path: str, output: str, config: dict, yes: bool = False, language_name: str = None):
    """
    Handle translation of an existing SRT file.

    Args:
        srt_path: Path to the SRT file
        output: Output directory from CLI argument (or None)
        config: Configuration dictionary from load_config()
        yes: If True, skip prompts and use defaults
        language_name: Source language name from --language flag (used with --yes)
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

    # Determine output directory (priority: CLI > config > default)
    output_dir = get_output_directory(output, config, srt_file.parent)

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
    translation_time = translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name, config, yes=yes, language_name=language_name)

    click.echo("\n✅ Done!")

    # Display timing summary
    if translation_time is not None:
        click.echo(f"\n⏱ Time spent:")
        click.echo(f"  Translation: {translation_time:.1f}s")


def _get_nvidia_info() -> dict:
    """
    Get NVIDIA GPU information from nvidia-smi.

    Returns:
        dict with keys:
        - available: bool - whether NVIDIA GPU is available
        - driver_version: str or None - driver version (e.g., "535.104.05")
        - cuda_version: str or None - max supported CUDA version (e.g., "12.2")
        - gpu_name: str or None - GPU name (e.g., "NVIDIA GeForce RTX 3080")
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output: "535.104.05, NVIDIA GeForce RTX 3080"
        parts = result.stdout.strip().split(', ', 1)
        driver_version = parts[0] if parts else None
        gpu_name = parts[1] if len(parts) > 1 else None

        # Get CUDA version from nvidia-smi header
        header_result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        cuda_version = None
        for line in header_result.stdout.split('\n'):
            if 'CUDA Version:' in line:
                # Parse "CUDA Version: 12.2" from the line
                import re
                match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                if match:
                    cuda_version = match.group(1)
                break

        return {
            'available': True,
            'driver_version': driver_version,
            'cuda_version': cuda_version,
            'gpu_name': gpu_name
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'available': False, 'driver_version': None, 'cuda_version': None, 'gpu_name': None}


def _get_cuda_compatibility(driver_version: str) -> list:
    """
    Get compatible PyTorch CUDA versions based on NVIDIA driver version.

    Args:
        driver_version: NVIDIA driver version string (e.g., "535.104.05")

    Returns:
        List of compatible CUDA versions (e.g., ["cu118", "cu121"])
    """
    if not driver_version:
        return []

    try:
        # Parse major version from driver version
        major = int(driver_version.split('.')[0])
    except (ValueError, IndexError):
        return []

    # Check compatibility against CUDA_VERSIONS config
    compatible = []
    for cuda_ver, info in CUDA_VERSIONS.items():
        if major >= info["min_driver"]:
            compatible.append(cuda_ver)

    return compatible


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _check_ollama() -> bool:
    """Check if Ollama is running by pinging the API."""
    import requests
    config = load_config()
    base_url = config.get('ollama', {}).get('base_url', 'http://localhost:11434')
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def run_system_check():
    """Run system diagnostics and display results."""
    click.echo("System Check:")
    click.echo(f"  Platform: {platform.system()} {platform.machine()}")

    # Check Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        click.echo("  Apple Silicon: Yes")
        try:
            import mlx_whisper  # noqa: F401
            click.echo("  mlx-whisper: Installed (Metal GPU acceleration available)")
        except ImportError:
            click.echo("  mlx-whisper: Not installed")
            click.echo("    → Run 'uv sync --extra mlx' for Metal GPU acceleration")

        # Check stable-ts on Apple Silicon
        try:
            import stable_whisper  # noqa: F401
            click.echo("  stable-ts: Installed (use --stable for better timestamps)")
        except ImportError:
            click.echo("  stable-ts: Not installed")
            click.echo("    → Run 'uv sync --extra stable' for better timestamp accuracy")
    else:
        # Check NVIDIA/CUDA
        nvidia_info = _get_nvidia_info()
        click.echo(f"  NVIDIA GPU: {'Found' if nvidia_info['available'] else 'Not found'}")

        if nvidia_info['available']:
            if nvidia_info['gpu_name']:
                click.echo(f"  GPU Model: {nvidia_info['gpu_name']}")
            if nvidia_info['driver_version']:
                click.echo(f"  Driver Version: {nvidia_info['driver_version']}")
            if nvidia_info['cuda_version']:
                click.echo(f"  Max CUDA Version: {nvidia_info['cuda_version']}")

            # Show compatible PyTorch CUDA versions
            compatible = _get_cuda_compatibility(nvidia_info['driver_version'])
            if compatible:
                click.echo(f"  Compatible PyTorch: {', '.join(compatible)}")

                # Check if default CUDA version is compatible
                if DEFAULT_CUDA not in compatible:
                    click.echo("")
                    click.echo(f"  ⚠ Warning: This project defaults to {DEFAULT_CUDA}, but your driver only supports:")
                    click.echo(f"    {', '.join(compatible)}")
                    # Suggest the best compatible version
                    if compatible:
                        best_compatible = compatible[-1]  # Highest compatible version
                        torch_ver = CUDA_VERSIONS[best_compatible]["torch"]
                        click.echo("")
                        click.echo("  → To fix, run:")
                        click.echo(f"    uv pip install torch=={torch_ver} --index-url https://download.pytorch.org/whl/{best_compatible}")
                    else:
                        default_info = CUDA_VERSIONS[DEFAULT_CUDA]
                        click.echo(f"    You may need to upgrade your NVIDIA driver to version {default_info['min_driver']}+")

        import torch
        cuda_available = torch.cuda.is_available()
        click.echo(f"  PyTorch CUDA: {'Available' if cuda_available else 'Not available'}")

        if nvidia_info['available'] and not cuda_available:
            click.echo("")
            click.echo("  ⚠ NVIDIA GPU detected but PyTorch CUDA is not available.")
            click.echo("    This usually means PyTorch was installed without CUDA support,")
            click.echo("    or there's a CUDA version mismatch with your driver.")
            if platform.system() == "Windows":
                compatible = _get_cuda_compatibility(nvidia_info['driver_version'])
                if compatible and DEFAULT_CUDA not in compatible:
                    best_compatible = compatible[-1]
                    torch_ver = CUDA_VERSIONS[best_compatible]["torch"]
                    click.echo("")
                    click.echo(f"  → Your driver doesn't support {DEFAULT_CUDA} (default). Try:")
                    click.echo(f"    uv pip install torch=={torch_ver} --index-url https://download.pytorch.org/whl/{best_compatible}")
                else:
                    click.echo("")
                    click.echo("  → Try reinstalling: uv sync --reinstall-package torch")
            else:
                click.echo("    On Linux, PyTorch from PyPI should include CUDA automatically.")
                click.echo("    Try: uv sync --reinstall-package torch")
        elif cuda_available:
            click.echo(f"  PyTorch CUDA Device: {torch.cuda.get_device_name(0)}")

        # Check stable-ts on non-Apple Silicon
        try:
            import stable_whisper  # noqa: F401
            click.echo("  stable-ts: Installed (use --stable for better timestamps)")
        except ImportError:
            click.echo("  stable-ts: Not installed")
            click.echo("    → Run 'uv sync --extra stable' for better timestamp accuracy")

    # Check ffmpeg
    ffmpeg_ok = _check_ffmpeg()
    click.echo(f"  ffmpeg: {'Installed' if ffmpeg_ok else 'Not found'}")
    if not ffmpeg_ok:
        click.echo("    → Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")

    # Check Ollama
    ollama_ok = _check_ollama()
    click.echo(f"  Ollama: {'Running' if ollama_ok else 'Not running or not installed'}")
    if not ollama_ok:
        click.echo("    → Optional: Install from https://ollama.ai for subtitle translation")


@click.command()
@click.argument('data_input', type=DataInput(), required=False)
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
@click.option(
    '--yes', '-y',
    is_flag=True,
    default=False,
    help='Answer yes to all prompts (use defaults for translation)'
)
@click.option(
    '--check-system',
    is_flag=True,
    default=False,
    help='Check system capabilities (GPU, CUDA, ffmpeg, Ollama)'
)
@click.option(
    '--stable',
    is_flag=True,
    default=False,
    help='Use stable-ts for better timestamp accuracy (requires: uv sync --extra stable)'
)
@click.option(
    '--vad',
    is_flag=True,
    default=False,
    help='Use Voice Activity Detection to reduce hallucinations (requires: --stable)'
)
def main(data_input, model, language, output, keep_audio, yes, check_system, stable, vad):
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
    # Handle --check-system flag (runs without requiring data_input)
    if check_system:
        run_system_check()
        return

    # data_input is required for normal operation
    if data_input is None:
        raise click.UsageError("Missing argument 'DATA_INPUT'.")

    try:
        # Load config for output directory settings
        config = load_config()

        # Parse language input: accept name ("Korean") or code ("ko")
        if language:
            parsed = parse_language(language)
            if parsed is None:
                click.echo(
                    f"❌ Error: '{language}' is not a recognized language.\n"
                    f"Use a language code (e.g., ko, en, zh) or name (e.g., Korean, English, Chinese).",
                    err=True
                )
                sys.exit(1)
            language_name, language_code = parsed
        else:
            language_name = None
            language_code = None

        # Handle SRT file input - skip to translation
        if is_srt_file(data_input):
            handle_srt_translation(data_input, output, config, yes=yes, language_name=language_name)
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

                # Always prompt for subtitle selection regardless of --yes flag.
                # This is a fast operation (~1 sec) and the user is still at the terminal.
                # --yes only affects the translation prompts after transcription.
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

                    # Determine output directory (priority: CLI > config > cwd)
                    output_dir = get_output_directory(output, config, Path.cwd())

                    srt_path = output_dir / f"{date_prefix}_{base_name}.srt"

                    # Download subtitle
                    downloader.download_subtitle(data_input, selected_lang, str(srt_path))

                    # Parse the downloaded SRT for translation
                    writer = SubtitleWriter()
                    segments = writer.parse_srt(str(srt_path))

                    click.echo(f"✓ Subtitle downloaded: {srt_path}")

                    # Offer translation
                    translation_time = translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name, config, yes=yes, language_name=language_name)

                    click.echo("\n✅ Done! Subtitle download complete.")
                    click.echo(f"\nOutput files saved to:")
                    click.echo(f"  {output_dir}")

                    # Display timing summary
                    if translation_time is not None:
                        click.echo(f"\n⏱ Time spent:")
                        click.echo(f"  Translation: {translation_time:.1f}s")

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

        # Determine output directory (priority: CLI > config > default)
        # Default: video's directory for local files, cwd for URL downloads
        default_output = Path.cwd() if is_url_input else video_path.parent
        output_dir = get_output_directory(output, config, default_output)

        # Generate output file paths with date prefix
        # Audio is temporary - save to video's directory (temp dir for URLs)
        audio_path = video_path.parent / f"{date_prefix}_{base_name}.wav"
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
        if language_code:
            click.echo(f"      Language: {language_name} ({language_code})")
        else:
            click.echo("      Language: auto-detect")

        transcriber = Transcriber(model_size=model, use_stable=stable, use_vad=vad)
        click.echo(f"      Device: {transcriber.device} ({transcriber.compute_type})")
        click.echo(f"      Backend: {transcriber.backend}")
        transcribe_start = time.time()
        segments = transcriber.transcribe(str(audio_path), language=language_code)
        transcribe_time = time.time() - transcribe_start
        click.echo(f"✓ Transcription complete ({len(segments)} segments)")

        # Step 3: Write subtitle files
        step_num = "[3/4]" if is_url(data_input) else "[3/3]"
        click.echo(f"\n{step_num} Writing subtitle file...")
        writer = SubtitleWriter()

        writer.write_srt(segments, str(srt_path))
        click.echo(f"✓ SRT file created: {srt_path}")

        # Step 4: Offer translation (for URL inputs this becomes [4/4])
        translation_time = translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name, config, yes=yes, language_name=language_name)

        # Clean up audio file if not keeping it
        if not keep_audio and audio_path.exists():
            audio_path.unlink()
            click.echo(f"\n✓ Cleaned up temporary audio file")
        elif keep_audio:
            click.echo(f"\n✓ Audio file kept: {audio_path}")

        click.echo("\n✅ Done! Subtitle extraction complete.")
        click.echo(f"\nOutput files saved to:")
        click.echo(f"  {output_dir}")

        # Display timing summary
        click.echo(f"\n⏱ Time spent:")
        click.echo(f"  Whisper transcription: {transcribe_time:.1f}s")
        if translation_time is not None:
            click.echo(f"  Translation: {translation_time:.1f}s")
            click.echo(f"  Total: {transcribe_time + translation_time:.1f}s")

    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
