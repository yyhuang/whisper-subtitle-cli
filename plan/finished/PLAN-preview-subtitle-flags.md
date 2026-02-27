# PLAN: Add `--preview` and `--subtitle` flags for script automation

## Context
When running multiple subtitle commands in a batch script, the interactive subtitle selection prompt blocks automation. We need:
1. `--subtitle N` - pre-select a subtitle by index (0=transcribe, 1/2/3...=subtitle), skipping the interactive prompt
2. `--preview` - check available subtitles, ask user interactively, then output the real command (with `--subtitle N` baked in) to stdout

Workflow:
```bash
# Pass 1: preview script asks user, outputs real commands
bash preview_run.sh > real_run.sh
# Pass 2: real script runs unattended
bash real_run.sh
```

## Files to modify

### 1. `main.py` — Add two CLI options and modify subtitle selection logic

**Add CLI options:**
- `--subtitle` (`type=int, default=None`): Pre-select subtitle by index. `0` = transcribe, `1`+ = download that subtitle.
- `--preview` (`is_flag=True`): Preview mode — check subtitles, ask user, print the real command to stdout, then exit.

**Modify subtitle selection logic (lines 619-689):**
- When `--subtitle N` is provided and N > 0: skip the interactive prompt, directly select the Nth subtitle. If N exceeds available count, error out.
- When `--subtitle 0` is provided: skip subtitle check entirely, go straight to transcription.
- When `--preview` is provided:
  - Check available subtitles (existing logic)
  - Show the list and prompt user for choice (existing interactive logic)
  - Print the real command to **stdout** (all other output goes to **stderr** so piping works cleanly)
  - Exit without processing

**Important detail for `--preview`:** All informational output (like "Checking for available subtitles...") should go to stderr so only the final command goes to stdout. We can use `click.echo(..., err=True)` for this.

### 2. `tests/` — Add tests for new flags

**Test `--subtitle`:**
- `--subtitle 0` skips subtitle check, goes to transcription
- `--subtitle 1` selects first subtitle without prompting
- `--subtitle` with out-of-range index shows error

**Test `--preview`:**
- Outputs a valid command to stdout
- Command includes `--subtitle N` matching user's choice
- Preserves other flags from the original command (e.g., `--model`, `--language`, `-y`)

## Implementation details

### Command output format for `--preview`
```
uv run python main.py "URL" --subtitle 1 -y
```
- Always includes `--subtitle N`
- Preserves `--model`, `--language`, `--output`, `--keep-audio`, `--stable`, `--vad` if they were specified
- Always appends `-y` for unattended translation
- Does NOT include `--preview` (obviously)

### Edge cases
- `--preview` with a local file (not URL): no subtitle check needed, just output the command as-is with `--subtitle 0`
- `--preview` with an SRT file: no subtitle check, output the command as-is
- `--subtitle` with no subtitles available and N > 0: error message, exit 1
- `--preview` without a URL: just output the command with `--subtitle 0` (no subtitles to check for local files)

## Verification
1. `uv run pytest -v` — all existing tests pass
2. Manual test `--preview` with a YouTube URL that has subtitles
3. Manual test `--subtitle 0` skips prompt
4. Manual test `--subtitle 1` downloads first subtitle without prompt
5. Test piping: `uv run python main.py --preview "URL" 2>/dev/null` should output only the command
