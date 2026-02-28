# PLAN: Add --action flag to separate transcribe/translate steps

## Context

When `auto_unload=True`, `--preview` outputs two-phase commands. Phase 1 (transcribe-only) intentionally omits `-y`, but the code still calls `translate_subtitles()` which prompts "Would you like to translate the subtitles?" interactively. There's no way to say "skip translation entirely."

Rather than adding a `--no-translate` hack, we add `--action` to make the intent explicit:
- `--action transcribe` — transcribe only, skip translation entirely
- `--action translate` — translate an existing SRT only
- No `--action` (default) — do both (current behavior, prompts or `-y`)

## Example usage

```bash
# Default: do both (current behavior, with prompts or -y)
uv run python main.py video.mp4
uv run python main.py video.mp4 -y

# Transcribe only — no translation prompt at all
uv run python main.py video.mp4 --action transcribe

# Translate only — expects SRT input
uv run python main.py existing.srt --action translate
uv run python main.py existing.srt --action translate -y
```

`--preview` two-phase output becomes:
```bash
# Phase 1:
uv run python main.py URL --subtitle 0 --action transcribe
# Phase 2:
uv run python main.py 20260228_abc.srt --action translate -y
```

## Changes

### 1. Add `--action` CLI option to `main.py`

Add a Click option:
```python
@click.option(
    '--action',
    type=click.Choice(['transcribe', 'translate'], case_sensitive=False),
    default=None,
    help='Action to perform: transcribe (no translation), translate (SRT input only). Default: both.'
)
```

### 2. Wire `--action transcribe` — skip translation step

In `main()`, after transcription completes (line ~952), check action before calling `translate_subtitles`:
```python
# Only offer translation if action is not 'transcribe'
if action != 'transcribe':
    translation_time = translate_subtitles(...)
```

Also skip translation for the subtitle download path (line ~825).

### 3. Wire `--action translate` — go straight to SRT translation

At the top of `main()`, if `action == 'translate'`:
- Validate that input is an SRT file (error if not)
- Call `handle_srt_translation()` directly
- This reuses the existing SRT translation path

### 4. Update `_build_transcribe_command()` — add `--action transcribe`

File: `main.py`, lines 103-131

Add `--action transcribe` to the Phase 1 command. Remove the misleading comment "no -y so translation is not triggered":
```python
parts.append('--action transcribe')
```

### 5. Update `_build_translate_command()` — add `--action translate`

File: `main.py`, lines 134-149

Add `--action translate` to the Phase 2 command (makes intent explicit, though SRT input already routes to translation):
```python
parts.append('--action translate')
```

### 6. Update `_build_preview_command()` — no change needed

This builds commands for subtitle download paths and single-command mode. These already use `-y` for the "do both" behavior, which is correct.

### 7. Update tests

File: `tests/test_preview_subtitle_flags.py`
- Two-phase tests: verify Phase 1 command includes `--action transcribe`
- Two-phase tests: verify Phase 2 command includes `--action translate`

File: `tests/test_main_integration.py`
- Add test: `--action transcribe` with video file — no translation prompt
- Add test: `--action translate` with SRT file — goes to translation
- Add test: `--action translate` with video file — shows error
- Existing tests (no `--action`) should pass unchanged (backward compat)

### 8. Update CLAUDE.md

- Add `--action` to CLI Options section
- Update `--preview` documentation

## Files to modify
- `main.py` — add `--action` option, wire logic, update command builders
- `tests/test_main_integration.py` — new integration tests
- `tests/test_preview_subtitle_flags.py` — update two-phase preview tests
- `CLAUDE.md` — document new flag

## Verification
1. `uv run pytest -v` — all existing tests pass (backward compat)
2. Manual test: `uv run python main.py video.mp4 --action transcribe` — no translation prompt
3. Manual test: `uv run python main.py existing.srt --action translate -y` — translates directly
4. Manual test: `uv run python main.py video.mp4` — current behavior unchanged (prompts)
5. Preview two-phase: verify generated commands include `--action transcribe` / `--action translate`
