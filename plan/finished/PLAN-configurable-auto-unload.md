# PLAN: Make auto_unload a configurable option

## Context
This plan builds on two previously completed features:
- **`plan/finished/PLAN-two-phase-preview.md`**: Added `--preview` flag that outputs two commands for VRAM-constrained machines — Phase 1 transcribes only, Phase 2 translates the SRT.
- **Auto-unload** (commit `c298745`): Added `unload_all_models()` to evict Ollama models from VRAM just before Whisper loads.

Both of these features exist purely to work around VRAM constraints. Users with enough VRAM don't need either behavior. Both should be gated behind a single config flag, defaulting to `false` (opt-in).

## Config
Add `ollama.auto_unload` (boolean, default `false`) to `config.json`.

## Changes

### 1. `src/translator.py` — `load_config()` (line 160)
Add `"auto_unload": False` to default ollama config.

### 2. `main.py` — unload call (line 921)
Gate the `unload_all_models()` call behind `config['ollama']['auto_unload']`.

### 3. `main.py` — preview paths (3 locations)
When `auto_unload` is `false`, preview for transcription paths outputs a **single command** (`_build_preview_command` with `--subtitle 0` and `-y`) instead of two-phase split.

Locations:
- **Line 755-763**: URL with subtitles, user picks 0 → single command
- **Line 839-848**: URL with no subtitles → single command
- **Line 888-894**: Local file → single command

When `auto_unload` is `true`, keep current two-phase behavior.

To pass `auto_unload` into the preview code paths, read it from `config` which is already available in `main()`.

### 4. Tests

#### `tests/test_translator.py`
- Add test: `load_config` returns `auto_unload: False` by default
- Add test: `load_config` reads `auto_unload: true` from config file

#### `tests/test_preview_subtitle_flags.py`
- Update `TestPreviewTwoPhaseWorkflow` tests to patch config with `auto_unload: True` (since they test two-phase behavior)
- Add new tests for `auto_unload: False` → single command output

#### `tests/test_main_integration.py` (if relevant)
- Update any test that hits the unload path to account for the config flag

### 5. `config.json`
User will set `"auto_unload": true` in their own config. No change to the committed config needed (default is false).

### 6. `CLAUDE.md`
- Update config example to show `auto_unload` option
- Update the "Auto-unload Ollama models" section to note it's configurable

## Verification
```bash
# Run all tests
uv run pytest -v

# Manual: preview with default config (auto_unload=false) should output single command
uv run python main.py "https://youtube.com/watch?v=TEST" --preview

# Manual: preview with auto_unload=true should output two commands
# (set "auto_unload": true in config.json first)
```
