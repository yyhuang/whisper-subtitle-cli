# Installation Guide

This guide will walk you through installing `whisper-subtitle-cli` step by step, even if you're new to Python or command-line tools.

## Prerequisites

You'll need to install three things before using this tool:
1. **Python** (version 3.11 or 3.12)
2. **Poetry** (Python package manager)
3. **ffmpeg** (audio/video processing tool)

---

## Step 1: Install Python

This project requires **Python 3.11 or 3.12** (not 3.13+, due to dependency compatibility).

### Check if Python is already installed

Open your terminal (macOS/Linux) or Command Prompt (Windows) and run:

```bash
python3 --version
```

If you see `Python 3.11.x` or `Python 3.12.x`, you're good! Skip to Step 2.

### Install Python if needed

#### Option A: Using pyenv (Recommended)

**pyenv** is the recommended way to install Python because it lets you easily install and switch between specific Python versions. This is especially useful when:
- Your system Python is too new (3.13+) or too old
- You work on multiple projects requiring different Python versions
- You want to avoid conflicts with system Python

##### Install pyenv

**macOS:**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install pyenv
brew install pyenv

# Add pyenv to your shell (for zsh, the default on macOS)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc

# Reload your shell
source ~/.zshrc
```

**Linux (Ubuntu/Debian):**
```bash
# Install dependencies
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl git \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add to your shell (for bash)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

# Reload your shell
source ~/.bashrc
```

##### Install Python with pyenv

This project includes a `.python-version` file that tells pyenv which Python version to use. Simply install the required version:

```bash
# Navigate to the project directory
cd /path/to/whisper-subtitle-cli

# Check which Python version is required
cat .python-version  # Shows the required version (e.g., 3.11.11)

# Install that version with pyenv
pyenv install 3.11.11  # Use the version from .python-version

# Verify - pyenv automatically uses the version from .python-version
python --version  # Should match the version in .python-version
```

**Important notes:**
- The `.python-version` file is already included in this project - no need to run `pyenv local`
- pyenv automatically detects this file and uses the correct Python version
- This only affects this project directory - won't change your system Python or other projects
- With pyenv active, use `python` (not `python3`) - pyenv handles the version
- **Never use `pyenv global`** - it changes system-wide settings and can cause issues

#### Option B: Using Homebrew (macOS only)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.12
```

#### Option C: Download from python.org

1. Visit https://www.python.org/downloads/
2. Download Python 3.12 (not the latest if it's 3.13+)
3. Run the installer
4. **Windows users**: Check "Add Python to PATH" during installation

#### Option D: Linux Package Manager

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

**Fedora:**
```bash
sudo dnf install python3.12
```

---

## Step 2: Install Poetry

Poetry is a tool that manages Python dependencies. Think of it like a package manager for your Python projects.

### Install Poetry

#### macOS/Linux

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

After installation, you may need to add Poetry to your PATH. The installer will show you instructions like:

```bash
export PATH="/Users/yourusername/.local/bin:$PATH"
```

Add this line to your shell configuration file (`~/.zshrc` for macOS or `~/.bashrc` for Linux).

#### Windows (PowerShell)

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Verify Poetry installation

```bash
poetry --version
```

You should see something like `Poetry (version 1.7.0)`.

**Troubleshooting**: If `poetry` command is not found:
- Close and reopen your terminal
- Make sure you added Poetry to your PATH (see above)
- On Windows, you may need to restart your computer

### Configure Poetry to use pyenv (if applicable)

If you installed Python with pyenv, make sure Poetry uses the correct version:

```bash
# Navigate to the project directory first
cd /path/to/whisper-subtitle-cli

# Verify pyenv is using the right Python
pyenv version  # Should show 3.11.x or 3.12.x

# Tell Poetry to use the current Python
poetry env use python
```

This ensures Poetry creates its virtual environment with the pyenv-managed Python.

---

## Step 3: Install ffmpeg

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
ollama pull gemma3:4b
```

**Model options for translation:**
| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `qwen2.5:3b` | ~2GB | Good | Fast |
| `gemma3:4b` | ~4.5GB | Better (default) | Medium |
| `llama3:8b` | ~4.7GB | Good | Medium |
| `qwen2.5:14b` | ~9GB | Best | Slow |

Choose based on your hardware. Larger models need more RAM/VRAM.

### Verify Ollama installation

```bash
# Check if Ollama is running
ollama list
```

You should see the model you downloaded (e.g., `gemma3:4b`).

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

Now we'll use Poetry to install all the required Python packages.

### Install dependencies

From inside the project directory, run:

```bash
poetry install --no-root
```

This will:
- Create a virtual environment (isolated Python environment for this project)
- Download and install all required packages

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
python main.py --help
```

You should see the help message with usage instructions. If you see this, congratulations! 🎉

---

## Quick Start

Now you're ready to use the tool!

### Test with a video file

```bash
python main.py your-video.mp4
```

### Test with a YouTube URL

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

---

## Common Issues

### "python: command not found"

Try using `python3` instead:
```bash
python3 main.py video.mp4
```

Or if you're using pyenv, make sure it's properly initialized:
```bash
# Check if pyenv is set up
pyenv version

# If not found, reinitialize
eval "$(pyenv init -)"
```

### pyenv: "python" still points to system Python

If `python --version` shows a different version than expected after setting `pyenv local`:

```bash
# Make sure pyenv shims are in PATH
echo $PATH | grep pyenv

# If not, add to your shell config and restart terminal
echo 'eval "$(pyenv init -)"' >> ~/.zshrc  # or ~/.bashrc

# Or use pyenv exec explicitly
pyenv exec python main.py video.mp4
```

### Poetry uses wrong Python version

If Poetry creates a venv with the wrong Python:

```bash
# Tell Poetry to use pyenv's Python
poetry env use $(pyenv which python)

# Or specify the version explicitly
poetry env use python3.12

# Reinstall dependencies
poetry install --no-root
```

### "poetry: command not found"

Poetry is not in your PATH. See Step 2 for instructions on adding it.

### "ffmpeg: command not found"

ffmpeg is not installed or not in your PATH. See Step 3.

### "ModuleNotFoundError: No module named 'src'"

Make sure you're running the command from the project root directory (the folder containing `main.py`).

### Virtual environment issues

If you have issues with Poetry's virtual environment, you can create one manually:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies without Poetry
pip install faster-whisper ffmpeg-python click yt-dlp
```

### Slow download speeds

If Poetry is downloading packages very slowly, you can try using a mirror or check your internet connection.

---

## Updating the Project

If you want to update to the latest version:

```bash
# If using Git
git pull

# Reinstall dependencies
poetry install --no-root
```

---

## Getting Help

If you run into issues:

1. Check the [README.md](README.md) for usage examples
2. Check the [Troubleshooting section](#common-issues) above
3. Open an issue on GitHub with:
   - Your operating system
   - Python version (`python3 --version`)
   - Poetry version (`poetry --version`)
   - The full error message

---

## Next Steps

Once installed, check out the [README.md](README.md) for:
- Usage examples
- Advanced options
- Model selection guide
- Language codes
