"""Ollama-based subtitle translator with batch processing."""

import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# Language name to ISO 639-1 code mapping for TranslateGemma
LANGUAGE_CODES = {
    'afrikaans': 'af', 'amharic': 'am', 'arabic': 'ar', 'assamese': 'as',
    'azerbaijani': 'az', 'bashkir': 'ba', 'belarusian': 'be', 'bulgarian': 'bg',
    'bengali': 'bn', 'tibetan': 'bo', 'breton': 'br', 'bosnian': 'bs',
    'catalan': 'ca', 'czech': 'cs', 'welsh': 'cy', 'danish': 'da',
    'german': 'de', 'greek': 'el', 'english': 'en', 'spanish': 'es',
    'estonian': 'et', 'basque': 'eu', 'persian': 'fa', 'finnish': 'fi',
    'faroese': 'fo', 'french': 'fr', 'galician': 'gl', 'gujarati': 'gu',
    'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'he', 'hindi': 'hi',
    'croatian': 'hr', 'haitian': 'ht', 'hungarian': 'hu', 'armenian': 'hy',
    'indonesian': 'id', 'icelandic': 'is', 'italian': 'it', 'japanese': 'ja',
    'javanese': 'jw', 'georgian': 'ka', 'kazakh': 'kk', 'khmer': 'km',
    'kannada': 'kn', 'korean': 'ko', 'latin': 'la', 'luxembourgish': 'lb',
    'lingala': 'ln', 'lao': 'lo', 'lithuanian': 'lt', 'latvian': 'lv',
    'malagasy': 'mg', 'maori': 'mi', 'macedonian': 'mk', 'malayalam': 'ml',
    'mongolian': 'mn', 'marathi': 'mr', 'malay': 'ms', 'maltese': 'mt',
    'burmese': 'my', 'nepali': 'ne', 'dutch': 'nl', 'nynorsk': 'nn',
    'norwegian': 'no', 'occitan': 'oc', 'punjabi': 'pa', 'polish': 'pl',
    'pashto': 'ps', 'portuguese': 'pt', 'romanian': 'ro', 'russian': 'ru',
    'sanskrit': 'sa', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk',
    'slovenian': 'sl', 'shona': 'sn', 'somali': 'so', 'albanian': 'sq',
    'serbian': 'sr', 'sundanese': 'su', 'swedish': 'sv', 'swahili': 'sw',
    'tamil': 'ta', 'telugu': 'te', 'tajik': 'tg', 'thai': 'th',
    'turkmen': 'tk', 'tagalog': 'tl', 'turkish': 'tr', 'tatar': 'tt',
    'ukrainian': 'uk', 'urdu': 'ur', 'uzbek': 'uz', 'vietnamese': 'vi',
    'yiddish': 'yi', 'yoruba': 'yo', 'chinese': 'zh', 'cantonese': 'yue',
    # Aliases (same code, different name for translation context)
    'traditional chinese': 'zh', 'taiwanese': 'zh',
}

# Reverse mapping: code → primary name (first occurrence wins)
LANGUAGE_NAMES = {}
for _name, _code in LANGUAGE_CODES.items():
    if _code not in LANGUAGE_NAMES:
        LANGUAGE_NAMES[_code] = _name.title()

# Prompt-specific language names: how to describe a language to the LLM
# "Chinese" is ambiguous, so we clarify it as Traditional Chinese (Taiwan) in prompts
PROMPT_LANGUAGE_NAMES = {
    'chinese': 'Traditional Chinese (Taiwan, 繁體中文)',
}


def get_prompt_language(language: str) -> str:
    """
    Get the language name to use in translation prompts.

    Some languages need clarification for the LLM (e.g., "Chinese" → "Traditional Chinese (Taiwan)").

    Args:
        language: Language name (e.g., 'Chinese', 'English')

    Returns:
        Prompt-friendly language name
    """
    return PROMPT_LANGUAGE_NAMES.get(language.lower(), language)


def parse_language(language: str):
    """
    Parse user language input and return (name, code) pair.

    Accepts either a language name (e.g., 'Korean') or code (e.g., 'ko').

    Args:
        language: User input - language name or code

    Returns:
        Tuple of (name, code) e.g., ('Korean', 'ko'), or None if unrecognized
    """
    lower = language.lower().strip()

    # Check if it's a known name
    if lower in LANGUAGE_CODES:
        code = LANGUAGE_CODES[lower]
        return (lower.title(), code)

    # Check if it's a known code
    if lower in LANGUAGE_NAMES:
        name = LANGUAGE_NAMES[lower]
        return (name, lower)

    # Unrecognized
    return None


def get_language_code(language: str) -> str:
    """
    Get ISO 639-1 language code from language name.

    Args:
        language: Language name (e.g., 'English', 'Chinese')

    Returns:
        ISO 639-1 code (e.g., 'en', 'zh') or lowercase language name if not found
    """
    return LANGUAGE_CODES.get(language.lower(), language.lower()[:2])


def get_language_name(code: str) -> str:
    """
    Get language name from ISO 639-1 code.

    Args:
        code: ISO 639-1 code (e.g., 'ko', 'en')

    Returns:
        Language name (e.g., 'Korean', 'English') or the code itself if not found
    """
    return LANGUAGE_NAMES.get(code.lower(), code)


def load_config() -> dict:
    """
    Load configuration from config.json file.

    Returns:
        Configuration dictionary with default values merged with file values.
    """
    config_path = Path(__file__).parent.parent / 'config.json'
    default_config = {
        "ollama": {
            "model": "translategemma:4b",
            "base_url": "http://localhost:11434",
            "batch_size": 50,
            "keep_alive": "10m"
        },
        "output": {
            "directory": None  # None means use default locations
        }
    }

    if config_path.exists():
        with open(config_path) as f:
            file_config = json.load(f)
            # Deep merge for ollama section
            if 'ollama' in file_config:
                default_config['ollama'].update(file_config['ollama'])
            # Deep merge for output section
            if 'output' in file_config:
                default_config['output'].update(file_config['output'])
            return default_config

    return default_config


class OllamaTranslator:
    """Translator using local Ollama API for subtitle translation with batch processing."""

    def __init__(self, model: str = None, base_url: str = None, batch_size: int = None, keep_alive: str = None):
        """
        Initialize the translator with Ollama settings.

        Args:
            model: Ollama model name (e.g., 'translategemma:4b'). Loads from config if not provided.
            base_url: Ollama API base URL. Loads from config if not provided.
            batch_size: Number of segments per batch. Loads from config if not provided.
            keep_alive: How long to keep model loaded (e.g., '10m', '1h', '-1'). Loads from config if not provided.
        """
        config = load_config()
        self.model = model or config['ollama']['model']
        self.base_url = base_url or config['ollama']['base_url']
        self.batch_size = batch_size or config['ollama'].get('batch_size', 50)
        self.keep_alive = keep_alive or config['ollama'].get('keep_alive', '10m')

    def _is_translategemma(self) -> bool:
        """Check if the current model is TranslateGemma."""
        return 'translategemma' in self.model.lower()

    def _call_ollama(self, prompt: str, timeout: int = 120) -> str:
        """
        Make a request to Ollama API.

        Args:
            prompt: The prompt to send
            timeout: Request timeout in seconds

        Returns:
            Response text from Ollama

        Raises:
            ConnectionError: If Ollama API is not available
            RuntimeError: If request fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.keep_alive
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Translation request timed out")
        except requests.exceptions.HTTPError as e:
            # Try to extract error message from response
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", str(e))
            except (ValueError, AttributeError):
                error_msg = str(e)
            raise RuntimeError(f"Ollama API error: {error_msg}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Translation failed: {e}")

    # Delimiter used to preserve line breaks during translation
    LINE_DELIMITER = " || "

    def _preserve_linebreaks(self, text: str) -> str:
        """Replace newlines with delimiter for translation."""
        return text.replace('\n', self.LINE_DELIMITER)

    def _restore_linebreaks(self, text: str) -> str:
        """Restore newlines from delimiter after translation."""
        return text.replace(self.LINE_DELIMITER, '\n')

    def _build_translategemma_prompt(self, text: str, source_lang: str, target_lang: str, has_delimiter: bool = False) -> str:
        """
        Build a prompt for TranslateGemma model.

        Args:
            text: Text to translate
            source_lang: Source language name
            target_lang: Target language name
            has_delimiter: Whether text contains line delimiters to preserve

        Returns:
            Formatted prompt for TranslateGemma
        """
        source_code = get_language_code(source_lang)
        target_code = get_language_code(target_lang)
        # Use prompt-specific names (e.g., "Chinese" → "Traditional Chinese (Taiwan, 繁體中文)")
        source_prompt = get_prompt_language(source_lang)
        target_prompt = get_prompt_language(target_lang)

        delimiter_instruction = ' Keep " || " delimiters in the same positions.' if has_delimiter else ''

        return f"""You are a professional {source_prompt} ({source_code}) to {target_prompt} ({target_code}) translator. Your goal is to accurately convey the meaning and nuances of the original {source_prompt} text while adhering to {target_prompt} grammar, vocabulary, and cultural sensitivities. Produce only the {target_prompt} translation, without any additional explanations or commentary.{delimiter_instruction}

{text}"""

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a single text string using Ollama.

        Args:
            text: Text to translate
            source_lang: Source language (e.g., 'English')
            target_lang: Target language (e.g., 'Chinese')

        Returns:
            Translated text

        Raises:
            ConnectionError: If Ollama API is not available
            RuntimeError: If translation fails
        """
        has_linebreaks = '\n' in text

        # Preserve linebreaks using delimiter
        if has_linebreaks:
            text = self._preserve_linebreaks(text)

        if self._is_translategemma():
            prompt = self._build_translategemma_prompt(text, source_lang, target_lang, has_delimiter=has_linebreaks)
        else:
            source_prompt = get_prompt_language(source_lang)
            target_prompt = get_prompt_language(target_lang)
            if has_linebreaks:
                prompt = f"Translate the following from {source_prompt} to {target_prompt}. Only output the translation. Keep \" || \" delimiters in the same positions:\n\n{text}"
            else:
                prompt = f"Translate the following from {source_prompt} to {target_prompt}. Only output the translation, nothing else:\n\n{text}"

        result = self._call_ollama(prompt, timeout=60)

        # Restore linebreaks
        if has_linebreaks:
            result = self._restore_linebreaks(result)

        return result

    def _build_batch_prompt(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Build a prompt for batch translation.

        Args:
            texts: List of texts to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            Formatted prompt string
        """
        # Replace newlines with delimiter to keep each segment on one line
        preserved_texts = [self._preserve_linebreaks(text) for text in texts]
        numbered_lines = "\n".join(
            f"{i + 1}. {text}" for i, text in enumerate(preserved_texts)
        )

        has_delimiters = any(self.LINE_DELIMITER in text for text in preserved_texts)
        delimiter_instruction = ' Keep " || " delimiters in the same positions.' if has_delimiters else ''

        source_prompt = get_prompt_language(source_lang)
        target_prompt = get_prompt_language(target_lang)

        if self._is_translategemma():
            source_code = get_language_code(source_lang)
            target_code = get_language_code(target_lang)
            prompt = f"""You are a professional {source_prompt} ({source_code}) to {target_prompt} ({target_code}) translator. Your goal is to accurately convey the meaning and nuances of the original {source_prompt} text while adhering to {target_prompt} grammar, vocabulary, and cultural sensitivities.

Translate each numbered line below. Return ONLY the translations with the same line numbers. Keep the exact format "N. translation".{delimiter_instruction}

{numbered_lines}"""
        else:
            prompt = f"""Translate each line from {source_prompt} to {target_prompt}.
Return ONLY the translations with the same line numbers. Keep the exact format "N. translation".{delimiter_instruction}

{numbered_lines}"""

        return prompt

    def _parse_batch_response(
        self,
        response: str,
        expected_count: int
    ) -> Optional[List[str]]:
        """
        Parse the batch translation response.

        Args:
            response: Raw response from Ollama
            expected_count: Number of translations expected

        Returns:
            List of translated texts if parsing succeeds, None if validation fails
        """
        # Parse lines with number prefix: "1. translation text"
        pattern = r'^(\d+)\.\s*(.+)$'

        translations = {}
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                num = int(match.group(1))
                text = match.group(2).strip()
                translations[num] = text

        # Validate we got all expected numbers
        result = []
        for i in range(1, expected_count + 1):
            if i not in translations:
                return None  # Missing translation, signal failure
            result.append(translations[i])

        return result

    def _try_translate_batch(
        self,
        segments: List[Dict],
        source_lang: str,
        target_lang: str
    ) -> Optional[List[Dict]]:
        """
        Try to translate a batch of segments.

        Args:
            segments: List of segments to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of translated segments if successful, None if failed
        """
        if not segments:
            return []

        texts = [seg['text'] for seg in segments]
        prompt = self._build_batch_prompt(texts, source_lang, target_lang)

        # Longer timeout for batches
        timeout = max(120, len(segments) * 5)
        response = self._call_ollama(prompt, timeout=timeout)

        translated_texts = self._parse_batch_response(response, len(segments))

        if translated_texts is None:
            return None  # Parsing failed, can retry with smaller batch

        # Restore linebreaks in translated texts
        translated_texts = [self._restore_linebreaks(text) for text in translated_texts]

        # Build result with preserved timestamps
        result = []
        for seg, translated_text in zip(segments, translated_texts):
            result.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': translated_text
            })

        return result

    def _translate_batch_recursive(
        self,
        segments: List[Dict],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[callable] = None,
        progress_offset: int = 0,
        total_segments: int = 0
    ) -> List[Dict]:
        """
        Recursively translate segments with split-on-failure strategy.

        Args:
            segments: List of segments to translate
            source_lang: Source language
            target_lang: Target language
            progress_callback: Optional callback for progress updates
            progress_offset: Current position in overall translation
            total_segments: Total number of segments being translated

        Returns:
            List of translated segments
        """
        if not segments:
            return []

        # Try to translate the batch
        result = self._try_translate_batch(segments, source_lang, target_lang)

        if result is not None:
            # Success - update progress and return
            if progress_callback:
                progress_callback(progress_offset + len(segments), total_segments)
            return result

        # Base case: single segment, can't split further
        if len(segments) == 1:
            # Try single translation as last resort
            translated_text = self.translate_text(
                segments[0]['text'],
                source_lang,
                target_lang
            )
            if progress_callback:
                progress_callback(progress_offset + 1, total_segments)
            return [{
                'start': segments[0]['start'],
                'end': segments[0]['end'],
                'text': translated_text
            }]

        # Recursive case: split in half and try each
        mid = len(segments) // 2

        left = self._translate_batch_recursive(
            segments[:mid],
            source_lang,
            target_lang,
            progress_callback,
            progress_offset,
            total_segments
        )

        right = self._translate_batch_recursive(
            segments[mid:],
            source_lang,
            target_lang,
            progress_callback,
            progress_offset + mid,
            total_segments
        )

        return left + right

    def translate_segments(
        self,
        segments: List[Dict],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Translate all segments using batch processing with recursive retry.

        Segments are processed in batches for better context and speed.
        If a batch fails, it's split in half and retried recursively.

        Args:
            segments: List of segments with 'start', 'end', 'text' keys
            source_lang: Source language (e.g., 'English')
            target_lang: Target language (e.g., 'Chinese')
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            List of translated segments with preserved timestamps

        Raises:
            ConnectionError: If Ollama API is not available
        """
        if not segments:
            return []

        total = len(segments)
        translated_segments = []

        # Process in batches
        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = segments[batch_start:batch_end]

            batch_result = self._translate_batch_recursive(
                batch,
                source_lang,
                target_lang,
                progress_callback,
                batch_start,
                total
            )

            translated_segments.extend(batch_result)

        return translated_segments

    def check_connection(self) -> bool:
        """
        Check if Ollama API is available.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
