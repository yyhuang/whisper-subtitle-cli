# Installation Guide

This guide will walk you through installing `whisper-subtitle-cli` step by step, even if you're new to Python or command-line tools.

## Prerequisites

You'll need to install these tools before using this project:
1. **uv** (Python package manager - also handles Python installation)
2. **ffmpeg** (audio/video processing tool)
3. **CUDA Toolkit** (for NVIDIA GPU acceleration - highly recommended)
4. **Ollama** (optional - only needed for subtitle translation)

---

## Step 1: Install uv

**uv** is a fast Python package manager that also handles Python version management. It will automatically download the correct Python version when needed.

### macOS/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:
```bash
source $HOME/.local/bin/env
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Verify uv installation

```bash
uv --version
```

You should see something like `uv 0.5.x`.

---

## Step 2: Install ffmpeg

ffmpeg is a powerful tool for processing audio and video files.

### macOS

```bash
brew install ffmpeg
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

### Linux (Fedora)

```bash
sudo dnf install ffmpeg
```

### Windows

**Option A: Using Chocolatey (recommended)**
```bash
# Install Chocolatey first (if you don't have it)
# Visit https://chocolatey.org/install for instructions

# Then install ffmpeg
choco install ffmpeg
```

**Option B: Manual installation**
1. Visit https://ffmpeg.org/download.html
2. Download the Windows build
3. Extract to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your system PATH

### Verify ffmpeg installation

```bash
ffmpeg -version
```

You should see version information for ffmpeg.

---

## Step 3: Install CUDA Toolkit (for NVIDIA GPU)

**Skip this step if:**
- You're on Apple Silicon (M1/M2/M3/M4) - use `uv sync --extra mlx` instead in Step 6
- You don't have an NVIDIA GPU
- You're okay with slower CPU-only transcription

GPU acceleration makes transcription **5-10x faster**. Without CUDA, a 10-minute video might take 5+ minutes to transcribe; with CUDA, it takes under a minute.

### Check if you have an NVIDIA GPU

```bash
# Linux
nvidia-smi

# Windows (PowerShell)
nvidia-smi
```

If you see GPU information, you have an NVIDIA GPU and should install CUDA.

### Install CUDA Toolkit

1. Download from https://developer.nvidia.com/cuda-downloads
2. Select your operating system and follow the installer
3. Restart your terminal after installation

### Verify CUDA installation

```bash
nvcc --version
```

You should see CUDA version information (e.g., `release 11.8` or `release 12.1`).

**Note:** After installing CUDA, you'll also need to install PyTorch with CUDA support. This is covered in Step 6.

---

## Step 4: Install Ollama (Optional - for Translation)

Ollama is required only if you want to **translate subtitles** to another language. Skip this step if you only need transcription.

Ollama runs AI models **locally on your computer** - no API keys or cloud services needed. All translation processing happens on your machine, keeping your data private.

**Note:** This tool only supports Ollama for translation. Other APIs (OpenAI, Claude, etc.) are not supported.

### macOS

```bash
# Download and install from the official website
# Visit: https://ollama.com/download

# Or use Homebrew
brew install ollama
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download from https://ollama.com/download

### Start Ollama and Download a Model

After installation, start the Ollama service and download a translation model:

```bash
# Start Ollama service (runs in background)
ollama serve

# In a new terminal, download the default translation model
ollama pull translategemma:4b
```

**Model options for translation:**

TranslateGemma is Google's specialized translation model, built on Gemma 3 and fine-tuned for 55 languages. It's the recommended choice for subtitle translation.

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `translategemma:4b` | ~3.3GB | Good (default) | Fast |
| `translategemma:12b` | ~8.1GB | Better | Medium |
| `translategemma:27b` | ~17GB | Best | Slow |

Other models (qwen, llama, gemma) also work but TranslateGemma is optimized for translation tasks.

Choose based on your hardware. Larger models need more RAM/VRAM.

### Verify Ollama installation

```bash
# Check if Ollama is running
ollama list
```

You should see the model you downloaded (e.g., `translategemma:4b`).

---

## Step 5: Download the Project

### Option A: Using Git (if you have it)

```bash
# Clone the repository
git clone https://github.com/yourusername/whisper-subtitle-cli.git

# Navigate into the project directory
cd whisper-subtitle-cli
```

### Option B: Download ZIP

1. Go to the GitHub repository page
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file
5. Open terminal and navigate to the extracted folder:
   ```bash
   cd path/to/whisper-subtitle-cli
   ```

---

## Step 6: Install Project Dependencies

From inside the project directory, run:

```bash
uv sync
```

This will:
- Automatically download and install the correct Python version (if needed)
- Create a virtual environment (`.venv/`)
- Download and install all required packages

### GPU Acceleration Setup

**Apple Silicon (M1/M2/M3/M4):**
```bash
# Install with Metal GPU support
uv sync --extra mlx
```

**NVIDIA GPU (Linux):**

On Linux, `uv sync` automatically installs PyTorch with CUDA support. No extra steps needed.

**NVIDIA GPU (Windows):**

On Windows, `uv sync` installs PyTorch with CUDA 12.1 support by default (requires driver >= 525).

To check if CUDA is working:
```bash
uv run python main.py --check-system
```

If you see "PyTorch CUDA: Not available", your driver may not support CUDA 12.1. The `--check-system` output will show your compatible CUDA versions and the command to fix it.

**Manual CUDA version override (Windows only):**
```bash
# For older drivers (>= 520) - use CUDA 11.8
uv pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU detection:**
```bash
uv run python main.py --check-system
```

This shows your system capabilities and confirms GPU acceleration is working.

### Optional: stable-ts for Better Timestamps

stable-ts is an optional enhancement that improves timestamp accuracy in subtitles. It works with all backends (CPU, CUDA, and MLX).

```bash
# Install stable-ts
uv sync --extra stable

# Or combine with mlx (Apple Silicon)
uv sync --extra mlx --extra stable

# Use with --stable flag
uv run python main.py video.mp4 --stable

# Add VAD to reduce hallucinations (optional)
uv run python main.py video.mp4 --stable --vad
```

When to use `--stable`:
- Subtitles are slightly out of sync with audio
- You need precise timestamps for professional use

When to add `--vad`:
- Whisper generates text during silent sections (hallucinations)
- Audio has long pauses or background noise
- Uses Silero VAD (neural network based)

### About Whisper Models (for Transcription)

The **first time you run** the tool, it will automatically download the Whisper AI model. This happens once and is cached for future use.

**Whisper model options (for transcription):**
| Model | Download Size | RAM Needed | Quality | Speed |
|-------|---------------|------------|---------|-------|
| `tiny` | ~39MB | ~1GB | Basic | Fastest |
| `base` | ~140MB | ~1GB | Good | Fast |
| `small` | ~470MB | ~2GB | Better | Medium |
| `medium` | ~1.5GB | ~5GB | High (default) | Slow |
| `large` | ~2.9GB | ~10GB | Best | Slowest |

The default `medium` model offers a good balance. Use `--model tiny` or `--model base` for faster processing or if you have limited RAM.

---

## Step 7: Verify Installation

Let's make sure everything works!

```bash
uv run python main.py --help
```

You should see the help message with usage instructions.

---

## Quick Start

Now you're ready to use the tool!

### Test with a video file

```bash
uv run python main.py your-video.mp4
```

### Test with a YouTube URL

```bash
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Test with an existing subtitle file

```bash
uv run python main.py existing_subtitle.srt
```

This skips transcription and goes directly to translation.

---

## Common Issues

### "uv: command not found"

uv is not in your PATH. Try:
- Restart your terminal
- Run `source $HOME/.local/bin/env`
- Reinstall uv following Step 1

### "ffmpeg: command not found"

ffmpeg is not installed or not in your PATH. See Step 2.

### "ModuleNotFoundError: No module named 'src'"

Make sure you're running the command from the project root directory (the folder containing `main.py`).

### Python version issues

uv automatically manages Python versions. If you encounter issues:

```bash
# Check which Python uv is using
uv run python --version

# Force uv to use a specific Python version
uv python install 3.11
uv sync
```

### Slow download speeds

If uv is downloading packages very slowly, check your internet connection or try again later.

### GPU not being used / Slow transcription on NVIDIA GPU

Run the system check to diagnose:
```bash
uv run python main.py --check-system
```

The output will show:
- Your NVIDIA driver version
- Compatible CUDA versions for your driver
- Whether PyTorch CUDA is available

**Common issues:**

1. **"NVIDIA GPU: Found" but "PyTorch CUDA: Not available"**
   - On Windows: Your driver may not support the default CUDA 12.1
   - Run `--check-system` to see which CUDA versions your driver supports
   - Follow the suggested `uv pip install` command to install a compatible version

2. **"Compatible PyTorch: cu118" (but project defaults to cu121)**
   - Your driver is older and only supports CUDA 11.8
   - Run: `uv pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118`

3. **No NVIDIA GPU detected**
   - NVIDIA drivers may not be installed
   - Install drivers from https://www.nvidia.com/drivers

---

## Updating the Project

If you want to update to the latest version:

```bash
# If using Git
git pull

# Reinstall dependencies
uv sync
```

---

## Getting Help

If you run into issues:

1. Check the [README.md](README.md) for usage examples
2. Check the [Troubleshooting section](#common-issues) above
3. Open an issue on GitHub with:
   - Your operating system
   - Python version (`uv run python --version`)
   - uv version (`uv --version`)
   - The full error message

---

## Next Steps

Once installed, check out the [README.md](README.md) for:
- Usage examples
- Advanced options
- Model selection guide
- Language codes
