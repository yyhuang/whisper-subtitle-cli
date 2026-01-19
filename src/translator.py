"""Ollama-based subtitle translator with batch processing."""

import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def load_config() -> dict:
    """
    Load configuration from config.json file.

    Returns:
        Configuration dictionary with default values merged with file values.
    """
    config_path = Path(__file__).parent.parent / 'config.json'
    default_config = {
        "ollama": {
            "model": "qwen2.5:7b",
            "base_url": "http://localhost:11434",
            "batch_size": 50
        }
    }

    if config_path.exists():
        with open(config_path) as f:
            file_config = json.load(f)
            # Deep merge for ollama section
            if 'ollama' in file_config:
                default_config['ollama'].update(file_config['ollama'])
            return default_config

    return default_config


class OllamaTranslator:
    """Translator using local Ollama API for subtitle translation with batch processing."""

    def __init__(self, model: str = None, base_url: str = None, batch_size: int = None):
        """
        Initialize the translator with Ollama settings.

        Args:
            model: Ollama model name (e.g., 'qwen2.5:7b'). Loads from config if not provided.
            base_url: Ollama API base URL. Loads from config if not provided.
            batch_size: Number of segments per batch. Loads from config if not provided.
        """
        config = load_config()
        self.model = model or config['ollama']['model']
        self.base_url = base_url or config['ollama']['base_url']
        self.batch_size = batch_size or config['ollama'].get('batch_size', 50)

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
                    "stream": False
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
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Translation failed: {e}")

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
        prompt = f"Translate the following from {source_lang} to {target_lang}. Only output the translation, nothing else:\n\n{text}"
        return self._call_ollama(prompt, timeout=60)

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
        numbered_lines = "\n".join(
            f"{i + 1}. {text}" for i, text in enumerate(texts)
        )

        prompt = f"""Translate each line from {source_lang} to {target_lang}.
Return ONLY the translations with the same line numbers. Keep the exact format "N. translation".

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

        try:
            # Longer timeout for batches
            timeout = max(120, len(segments) * 5)
            response = self._call_ollama(prompt, timeout=timeout)

            translated_texts = self._parse_batch_response(response, len(segments))

            if translated_texts is None:
                return None  # Parsing failed

            # Build result with preserved timestamps
            result = []
            for seg, translated_text in zip(segments, translated_texts):
                result.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': translated_text
                })

            return result

        except (ConnectionError, RuntimeError):
            return None

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
            try:
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
            except (ConnectionError, RuntimeError):
                # Last resort: keep original text
                if progress_callback:
                    progress_callback(progress_offset + 1, total_segments)
                return [segments[0]]

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
