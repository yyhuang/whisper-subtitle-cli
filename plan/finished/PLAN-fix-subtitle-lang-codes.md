# Plan: Fix subtitle language code handling

## Context

Two related bugs when downloading subtitles from YouTube:

1. **Dirty lang codes from yt-dlp**: Some videos return codes like `en-nP7-2PuUl7o` (language + hash suffix). `_get_language_name()` doesn't recognize these, so the display shows garbage like `EN-NP7-2PUUL7O (en-nP7-2PuUl7o)` instead of `English (en)`.

2. **Wrong translation source language**: When using `--subtitle N --language zh --yes`, the `--language` value (meant for Whisper transcription) is incorrectly used as the translation source language. If you download an English subtitle but passed `--language zh`, translation thinks the source is Chinese.

## Fix

### 1. Clean dirty lang codes in `_get_language_name()` (`src/video_downloader.py`)

Add a helper to extract the base language code before lookup:

- `en-nP7-2PuUl7o` → `en` → `English`
- `es-419-XTK0TJgvC-M` → `es-419` → try `es-419`, fallback to `es` → `Spanish`
- `ja-p4xb9ptA1GQ` → `ja` → `Japanese`
- `zh` → `zh` → `Chinese`

**Strategy**: Split on `-`. yt-dlp hashes are long alphanumeric strings (8+ chars). Standard subtags are short (`419`, `Hans`, `Hant`). Walk the parts and stop when we hit a hash-like segment.

Keep the raw code as dict key in `get_available_subtitles()` (yt-dlp needs the original for download). Only clean for display and language identification.

### 2. Use downloaded subtitle's language for translation (`main.py`)

In the `choice > 0` block (line ~729-758), derive the translation source language from the selected subtitle's lang code instead of from `--language`:

```python
selected_lang = subtitle_list[choice - 1][0]       # raw code, e.g. 'en-nP7-2PuUl7o'
selected_name = subtitle_list[choice - 1][1]['name'] # cleaned name, e.g. 'English'

# Use the subtitle's actual language for translation, not --language
parsed = parse_language(selected_name)
download_language_name = parsed[0] if parsed else selected_name

translation_time = translate_subtitles(
    ..., language_name=download_language_name
)
```

This way `--language zh` + `--subtitle 1` (English) correctly uses "English" as translation source.

## Files to modify

| File | Change |
|------|--------|
| `src/video_downloader.py` | Add `_clean_language_code()` helper; update `_get_language_name()` to use it |
| `main.py` | Use subtitle's language name (not `--language`) for translation when downloading |
| `tests/test_video_downloader.py` | Tests for dirty code cleaning |
| `tests/test_main_integration.py` or `tests/test_preview_subtitle_flags.py` | Test that `--subtitle N --language zh` uses correct translation source |

## Verification

1. `uv run pytest -v` — all existing tests pass
2. Manual test with a YouTube URL that has dirty codes — verify clean display
3. Verify `--subtitle 1 --language zh --yes` uses the downloaded subtitle's language as translation source, not "Chinese"
