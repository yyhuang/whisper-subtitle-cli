# PLAN: Two-phase preview commands (transcribe + translate)

## Context

The user's machine cannot run Whisper (transcription) and Ollama (translation) at the same time due to limited VRAM. When using `--preview` mode and a video needs transcription, the tool currently outputs **one** command that does both transcription and translation (`-y`). The user needs **two separate commands** so they can run transcription first, free the GPU, then run translation.

## Current Behavior

When `--preview` detects a video needs transcription (no subtitles or user picks "transcribe"):

```
uv run python main.py "URL" --subtitle 0 -y
```

## New Behavior

Output **two** commands:

```bash
# Phase 1: Transcribe only (NO -y, so no translation prompt)
uv run python main.py "URL" --subtitle 0 --language en
# Phase 2: Translate the resulting SRT file (--language passed through for source language)
uv run python main.py /path/to/20200101_abc123.srt -y --language en
```

The `--language` flag serves double duty:
- In Phase 1 (transcribe): tells Whisper the source language
- In Phase 2 (translate): tells Ollama the source language for translation (via `parse_language()` → `language_name` in `handle_srt_translation`)

**Note**: When the user picks a subtitle to **download** (choice > 0), keep the current single-command behavior since subtitle download doesn't use Whisper/GPU.

## Files to Modify

### 1. `main.py` — Add helpers + modify 3 preview paths

**Add two new helper functions** (near `_build_preview_command` at line ~70):

- `_build_transcribe_command(data_input, model, language, output, keep_audio, stable, vad)` → builds transcription command **without** `-y`
- `_build_translate_command(srt_path, output, language)` → builds `main.py SRT_PATH -y [--language] [--output]`

**Modify 3 preview code paths:**

| Location | Current | New |
|---|---|---|
| **URL, no subtitles** (line ~780) | Single cmd with `--subtitle 0 -y` | Call `get_video_info()` to compute SRT path, output 2 commands |
| **URL, subtitles exist, user picks 0** (line ~698-708) | Single cmd with `--subtitle 0 -y` | Same: call `get_video_info()`, output 2 commands |
| **Local file** (line ~823) | Single cmd with `--subtitle 0 -y` | Compute SRT path from file path, output 2 commands |

**SRT path computation:**
- For URLs: call `downloader.get_video_info(data_input)` to get `video_id` + `upload_date`, then `get_output_directory()` for the dir → `{output_dir}/{date_prefix}_{video_id}.srt`
- For local files: use `Path(data_input).resolve()` to get filename + modification date → `{output_dir}/{date_prefix}_{sanitized_name}.srt`

**Note**: `get_video_info()` makes a second yt-dlp call (after `get_available_subtitles()`). This is acceptable in interactive preview mode. No API change to `VideoDownloader`.

### 2. `tests/test_preview_subtitle_flags.py` — Add new test class

Add `TestPreviewTwoPhaseWorkflow` class with tests:

- URL no subtitles → outputs exactly 2 command lines
- First command has `--subtitle 0`, no `-y`
- Second command has `.srt` path + `-y`, no `--subtitle`
- SRT filename includes `video_id` and `date_prefix`
- URL subtitles exist, user picks 0 → also 2 commands
- Local file → 2 commands
- `--output` flag preserved in both commands
- `--language` flag passed through to translate command

**Existing tests**: All continue to pass without modification (they only assert `'--subtitle 0' in result.output` which remains true in the first command). Will add `get_video_info` mock to existing tests that exercise the choice=0 path for robustness.

### 3. `CLAUDE.md` — Update project status

Add completed feature note.

## Verification

1. `uv run pytest tests/test_preview_subtitle_flags.py -v` — all tests pass
2. `uv run pytest -v` — full suite passes
3. Manual test with a real YouTube URL:
   - `uv run python main.py "https://youtube.com/watch?v=SOME_ID" --preview` → pick 0 → see 2 commands
   - `uv run python main.py "https://youtube.com/watch?v=SOME_ID" --preview` → pick a subtitle → see 1 command
   - Redirect: `uv run python main.py "URL" --preview 2>/dev/null` → only commands on stdout
