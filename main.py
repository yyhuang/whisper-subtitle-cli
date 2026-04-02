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
from src.translator import OllamaTranslator, load_config, parse_language, unload_all_models

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


class SubtitleChoice(click.ParamType):
    """Accepts an integer in [0, max_val] or 's'/'S' to skip the video."""
    name = "choice"

    def __init__(self, max_val: int):
        self.max_val = max_val

    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lower() == 's':
            return 's'
        try:
            int_val = int(value)
            if 0 <= int_val <= self.max_val:
                return int_val
            self.fail(f"Expected 0-{self.max_val} or 'S' to skip", param, ctx)
        except (ValueError, TypeError):
            self.fail(f"Expected 0-{self.max_val} or 'S' to skip", param, ctx)


def is_srt_file(path: str) -> bool:
    """Check if the input is an SRT subtitle file."""
    return path.lower().endswith('.srt')


def _build_preview_command(
    data_input: str,
    subtitle_choice: int,
    model: str,
    language: str | None,
    output: str | None,
    keep_audio: bool,
    stable: bool,
    vad: bool,
    prompt_file: str | None = None,
) -> str:
    """Build the real command for --preview mode output (subtitle download paths)."""
    import shlex

    parts = ['uv run python main.py', shlex.quote(data_input)]
    parts.append(f'--subtitle {subtitle_choice}')
    parts.append('-y')

    if model != 'medium':
        parts.append(f'--model {model}')
    if language is not None:
        parts.append(f'--language {language}')
    if output is not None:
        parts.append(f'--output {shlex.quote(output)}')
    if keep_audio:
        parts.append('--keep-audio')
    if stable:
        parts.append('--stable')
    if vad:
        parts.append('--vad')
    if prompt_file is not None:
        parts.append(f'--prompt-file {shlex.quote(prompt_file)}')

    return ' '.join(parts)


def _build_transcribe_command(
    data_input: str,
    model: str,
    language: str | None,
    output: str | None,
    keep_audio: bool,
    stable: bool,
    vad: bool,
) -> str:
    """Build Phase 1 transcription command (transcribe only, no translation)."""
    import shlex

    parts = ['uv run python main.py', shlex.quote(data_input)]
    parts.append('--subtitle 0')
    parts.append('--action transcribe')

    if model != 'medium':
        parts.append(f'--model {model}')
    if language is not None:
        parts.append(f'--language {language}')
    if output is not None:
        parts.append(f'--output {shlex.quote(output)}')
    if keep_audio:
        parts.append('--keep-audio')
    if stable:
        parts.append('--stable')
    if vad:
        parts.append('--vad')

    return ' '.join(parts)


def _build_translate_command(
    srt_path: str,
    output: str | None,
    language: str | None,
    prompt_file: str | None = None,
) -> str:
    """Build Phase 2 translation command (SRT file + -y + --action translate)."""
    import shlex

    parts = ['uv run python main.py', shlex.quote(srt_path), '-y', '--action translate']

    if language is not None:
        parts.append(f'--language {language}')
    if output is not None:
        parts.append(f'--output {shlex.quote(output)}')
    if prompt_file is not None:
        parts.append(f'--prompt-file {shlex.quote(prompt_file)}')

    return ' '.join(parts)


def format_video_label(video_meta: dict, url: str = None) -> str:
    """
    Format video label from metadata for display.

    Args:
        video_meta: Dict with 'title' and 'channel' keys
        url: Optional URL to append in parentheses

    Returns:
        Formatted string like "Channel - Title (URL)" or "Title (URL)"
    """
    title = video_meta.get('title', 'Unknown')
    channel = video_meta.get('channel')

    if channel:
        label = f"{channel} - {title}"
    else:
        label = title

    if url:
        label = f"{label} ({url})"

    return label


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


def translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name, config, yes=False, language_name=None, custom_prompt=None):
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
    translator = OllamaTranslator(custom_prompt=custom_prompt)
    if translator.prompt_file_source:
        click.echo(f"Using config prompt file: {translator.prompt_file_source}")
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


def handle_srt_translation(srt_path: str, output: str, config: dict, yes: bool = False, language_name: str = None, custom_prompt: str = None):
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
    translation_time = translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name, config, yes=yes, language_name=language_name, custom_prompt=custom_prompt)

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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
    help='Enable VAD to reduce hallucinations in silence (requires: --stable)'
)
@click.option(
    '--subtitle',
    type=int,
    default=None,
    help='Pre-select subtitle by index (0=transcribe, 1+=download that subtitle). Skips interactive prompt.'
)
@click.option(
    '--preview',
    is_flag=True,
    default=False,
    help='Check subtitles, prompt user, output the real command with --subtitle N to stdout, then exit.'
)
@click.option(
    '--action',
    type=click.Choice(['transcribe', 'translate'], case_sensitive=False),
    default=None,
    help='Action to perform: transcribe (no translation), translate (SRT input only). Default: both.'
)
@click.option(
    '--prompt-file',
    type=click.Path(exists=True),
    default=None,
    help='Text file with extra instructions for the translation model (e.g., glossary, style guide).'
)
@click.option(
    '--preview-opt',
    'preview_opt',
    type=str,
    default=None,
    help='Non-interactive preview selection: L=list subtitles (JSON), S=skip, 0=transcribe, N=subtitle index. Implies --preview.'
)
def main(data_input, model, language, output, keep_audio, yes, check_system, stable, vad, subtitle, preview, action, prompt_file, preview_opt):
    """
    Extract subtitles from DATA_INPUT (file path, URL, or SRT file) using AI transcription.

    Generates .srt file for video players with timestamps.

    \b
    Examples:
      python main.py video.mp4
      python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
      python main.py video.mp4 --model medium --language en
      python main.py existing.srt
      python main.py "https://youtube.com/watch?v=ID" --preview
      python main.py "https://youtube.com/watch?v=ID" --subtitle 1 -y

    \b
    Batch scripting (two-pass workflow):
      # Pass 1: interactively pick subtitles, save real commands
      bash preview_run.sh > real_run.sh
      # Pass 2: run unattended
      bash real_run.sh
    """
    # Handle --check-system flag (runs without requiring data_input)
    if check_system:
        run_system_check()
        return

    # data_input is required for normal operation
    if data_input is None:
        raise click.UsageError("Missing argument 'DATA_INPUT'.")

    # --preview-opt implies --preview and validates the value
    if preview_opt is not None:
        preview_opt = preview_opt.lower()
        if preview_opt not in ('l', 's') and not preview_opt.isdigit():
            raise click.UsageError(
                f"Invalid --preview-opt value: '{preview_opt}'. "
                "Must be L (list), S (skip), 0 (transcribe), or a subtitle index number."
            )
        if preview_opt.isdigit():
            preview_opt = int(preview_opt)
        preview = True

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

        # Read custom prompt file if provided
        custom_prompt = None
        if prompt_file:
            custom_prompt = Path(prompt_file).read_text().strip()
            if custom_prompt:
                click.echo(f"Using prompt file: {prompt_file}", err=preview)

        # --action translate requires SRT input
        if action == 'translate' and not is_srt_file(data_input):
            click.echo(
                "❌ Error: --action translate requires an SRT file as input. "
                "Provide an existing .srt file.",
                err=True
            )
            sys.exit(1)

        # Handle SRT file input - skip to translation
        if is_srt_file(data_input):
            if preview:
                cmd = _build_preview_command(data_input, 0, model, language, output, keep_audio, stable, vad, prompt_file=prompt_file)
                click.echo(cmd)
                return
            handle_srt_translation(data_input, output, config, yes=yes, language_name=language_name, custom_prompt=custom_prompt)
            return

        # Step 0: Handle URL vs file path
        is_url_input = False
        temp_dir_path = None

        if is_url(data_input):
            is_url_input = True
            click.echo(f"Detected URL: {data_input}", err=preview)

            downloader = VideoDownloader()  # Uses system temp directory by default
            temp_dir_path = downloader.download_dir

            # When --subtitle 0 is given, skip the subtitle check entirely
            skip_subtitle_check = (subtitle is not None and subtitle == 0)
            subtitles_checked = False
            subtitles: dict = {}

            if not skip_subtitle_check:
                # Check for available subtitles first
                click.echo("\nChecking for available subtitles...", err=preview)
                subtitles, video_meta = downloader.get_available_subtitles(data_input)
                subtitles_checked = True
                subtitle_list = list(subtitles.items())

                # --preview-opt L: output JSON list and exit early (before any menu output)
                if preview_opt == 'l':
                    import json
                    data = {
                        'url': data_input,
                        'video_title': video_meta.get('title', ''),
                        'channel': video_meta.get('channel', ''),
                        'subtitles': [
                            {'index': idx, 'lang': lang_code, 'name': info['name']}
                            for idx, (lang_code, info) in enumerate(subtitle_list, 1)
                        ],
                        'can_transcribe': True,
                    }
                    click.echo(json.dumps(data))
                    return

                if subtitles:
                    video_label = format_video_label(video_meta, data_input)
                    click.echo(f"\nVideo: {video_label}", err=preview)

                    click.echo("\nAvailable subtitles:", err=preview)
                    for idx, (lang_code, info) in enumerate(subtitle_list, 1):
                        click.echo(f"  {idx}. {info['name']} ({lang_code})", err=preview)
                    click.echo(f"  0. Transcribe video instead", err=preview)
                    click.echo(f"  S. Skip this video", err=preview)

                    if preview:
                        if preview_opt is not None:
                            # Non-interactive: --preview-opt provides the choice
                            if preview_opt == 's':
                                return  # Skip: emit nothing to stdout
                            else:
                                choice = preview_opt  # int: 0 or subtitle index
                                if isinstance(choice, int) and choice > len(subtitle_list):
                                    click.echo(
                                        f"Error: --preview-opt {choice} is out of range. "
                                        f"Only {len(subtitle_list)} subtitle(s) available.",
                                        err=True,
                                    )
                                    sys.exit(1)
                        else:
                            # Interactive: prompt goes to stderr so only the command(s) reach stdout
                            choice = click.prompt(
                                "\nWhich subtitle would you like? (0-N, or S to skip)",
                                type=SubtitleChoice(len(subtitle_list)),
                                default='0',
                                err=True,
                            )
                        if choice == 's':
                            return  # Skip: emit nothing to stdout
                        comment = f"# {format_video_label(video_meta, data_input)}"
                        click.echo(comment)
                        if choice == 0:
                            if config['ollama'].get('auto_unload', False):
                                # Two-phase: transcribe first, translate separately (VRAM constraint)
                                video_info = downloader.get_video_info(data_input)
                                video_id = video_info['video_id']
                                date_prefix = get_date_prefix(upload_date=video_info.get('upload_date'))
                                output_dir = get_output_directory(output, config, Path.cwd())
                                srt_path = str(output_dir / f"{date_prefix}_{video_id}.srt")
                                click.echo(_build_transcribe_command(data_input, model, language, output, keep_audio, stable, vad))
                                click.echo(_build_translate_command(srt_path, output, language, prompt_file=prompt_file))
                            else:
                                click.echo(_build_preview_command(data_input, 0, model, language, output, keep_audio, stable, vad))
                        else:
                            cmd = _build_preview_command(data_input, choice, model, language, output, keep_audio, stable, vad, prompt_file=prompt_file)
                            click.echo(cmd)
                        return
                    elif subtitle is not None:
                        # --subtitle N: validate then select without prompting
                        if subtitle > len(subtitle_list):
                            click.echo(
                                f"❌ Error: --subtitle {subtitle} is out of range. "
                                f"Only {len(subtitle_list)} subtitle(s) available (or use 0 to transcribe).",
                                err=True,
                            )
                            sys.exit(1)
                        choice = subtitle
                    else:
                        # Normal interactive prompt
                        # Always prompt regardless of --yes (this is fast, user is at terminal)
                        # --yes only affects translation prompts after transcription.
                        choice = click.prompt(
                            "\nWhich subtitle would you like? (0-N, or S to skip)",
                            type=SubtitleChoice(len(subtitle_list)),
                            default='0',
                        )
                        if choice == 's':
                            return  # Skip this video

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

                        # Derive translation source from the downloaded subtitle's language,
                        # not from --language (which is for Whisper transcription).
                        parsed_sub_lang = parse_language(selected_name)
                        download_language_name = parsed_sub_lang[0] if parsed_sub_lang else selected_name

                        # Offer translation
                        translation_time = translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name, config, yes=yes, language_name=download_language_name, custom_prompt=custom_prompt)

                        click.echo("\n✅ Done! Subtitle download complete.")
                        click.echo(f"\nOutput files saved to:")
                        click.echo(f"  {output_dir}")

                        # Display timing summary
                        if translation_time is not None:
                            click.echo(f"\n⏱ Time spent:")
                            click.echo(f"  Translation: {translation_time:.1f}s")

                        return  # Exit early, skip transcription

                    # choice == 0: fall through to transcription

                else:
                    # No subtitles available
                    if preview:
                        if preview_opt is not None:
                            # Non-interactive: --preview-opt provides the choice
                            if preview_opt == 's':
                                return  # Skip: emit nothing to stdout
                            elif isinstance(preview_opt, int) and preview_opt > 0:
                                click.echo(
                                    f"Error: --preview-opt {preview_opt} is invalid. "
                                    f"No subtitles available for this URL.",
                                    err=True,
                                )
                                sys.exit(1)
                            # preview_opt == 0: fall through to transcribe
                        else:
                            video_label = format_video_label(video_meta, data_input)
                            click.echo(f"\nVideo: {video_label}", err=True)
                            click.echo(f"\nNo subtitles available.", err=True)
                            click.echo(f"  0. Transcribe video", err=True)
                            click.echo(f"  S. Skip this video", err=True)

                            choice = click.prompt(
                                "\nTranscribe or skip? (0 or S)",
                                type=SubtitleChoice(0),
                                default='0',
                                err=True,
                            )
                            if choice == 's':
                                return  # Skip: emit nothing to stdout

                        comment = f"# {format_video_label(video_meta, data_input)}"
                        click.echo(comment)
                        if config['ollama'].get('auto_unload', False):
                            # Two-phase: transcribe first, translate separately (VRAM constraint)
                            video_info = downloader.get_video_info(data_input)
                            video_id = video_info['video_id']
                            date_prefix = get_date_prefix(upload_date=video_info.get('upload_date'))
                            output_dir = get_output_directory(output, config, Path.cwd())
                            srt_path = str(output_dir / f"{date_prefix}_{video_id}.srt")
                            click.echo(_build_transcribe_command(data_input, model, language, output, keep_audio, stable, vad))
                            click.echo(_build_translate_command(srt_path, output, language, prompt_file=prompt_file))
                        else:
                            click.echo(_build_preview_command(data_input, 0, model, language, output, keep_audio, stable, vad))
                        return
                    elif subtitle is not None and subtitle > 0:
                        click.echo(
                            f"❌ Error: No subtitles available for this URL. "
                            f"Use --subtitle 0 to transcribe instead.",
                            err=True,
                        )
                        sys.exit(1)

            # No subtitles or user chose to transcribe — proceed to video download
            if subtitles_checked and not subtitles:
                click.echo("No manual subtitles available. Transcribing video...")
            elif subtitles_checked and subtitles:
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

            if preview:
                if config['ollama'].get('auto_unload', False):
                    # Two-phase: transcribe first, translate separately (VRAM constraint)
                    output_dir = get_output_directory(output, config, video_path.parent)
                    srt_path = str(output_dir / f"{date_prefix}_{base_name}.srt")
                    click.echo(_build_transcribe_command(data_input, model, language, output, keep_audio, stable, vad))
                    click.echo(_build_translate_command(srt_path, output, language, prompt_file=prompt_file))
                else:
                    click.echo(_build_preview_command(data_input, 0, model, language, output, keep_audio, stable, vad))
                return

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

        if config['ollama'].get('auto_unload', False):
            n_unloaded = unload_all_models(config['ollama']['base_url'])
            if n_unloaded:
                click.echo(f"  Unloading {n_unloaded} Ollama model(s) to free VRAM...")

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
        # Skip translation entirely when --action transcribe is specified
        if action == 'transcribe':
            translation_time = None
        else:
            translation_time = translate_subtitles(segments, srt_path, output_dir, date_prefix, base_name, config, yes=yes, language_name=language_name, custom_prompt=custom_prompt)

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
