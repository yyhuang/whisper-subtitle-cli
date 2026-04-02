import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from src.translator import OllamaTranslator, load_config, get_language_code, get_language_name, parse_language


class TestGetLanguageCode:
    """Tests for the get_language_code function."""

    def test_get_language_code_english(self):
        """Test language code for English."""
        assert get_language_code('English') == 'en'
        assert get_language_code('english') == 'en'
        assert get_language_code('ENGLISH') == 'en'

    def test_get_language_code_chinese(self):
        """Test language code for Chinese."""
        assert get_language_code('Chinese') == 'zh'

    def test_get_language_code_unknown(self):
        """Test fallback for unknown language."""
        assert get_language_code('Klingon') == 'kl'


class TestParseLanguage:
    """Tests for the parse_language function."""

    def test_parse_language_name(self):
        """Test parsing a language name returns (name, code) tuple."""
        assert parse_language('Korean') == ('Korean', 'ko')
        assert parse_language('English') == ('English', 'en')
        assert parse_language('Chinese') == ('Chinese', 'zh')

    def test_parse_language_code(self):
        """Test parsing a language code returns (name, code) tuple."""
        assert parse_language('ko') == ('Korean', 'ko')
        assert parse_language('en') == ('English', 'en')
        assert parse_language('zh') == ('Chinese', 'zh')

    def test_parse_language_case_insensitive(self):
        """Test parsing is case-insensitive."""
        assert parse_language('KOREAN') == ('Korean', 'ko')
        assert parse_language('korean') == ('Korean', 'ko')
        assert parse_language('Ko') == ('Korean', 'ko')

    def test_parse_language_with_whitespace(self):
        """Test parsing strips whitespace."""
        assert parse_language(' Korean ') == ('Korean', 'ko')
        assert parse_language(' ko ') == ('Korean', 'ko')

    def test_parse_language_unrecognized(self):
        """Test parsing unrecognized input returns None."""
        assert parse_language('Klingon') is None
        assert parse_language('xx') is None
        assert parse_language('') is None

    def test_parse_language_all_whisper_codes(self):
        """Test that common Whisper language codes are recognized."""
        # Test a sample of codes across the mapping
        assert parse_language('ja') == ('Japanese', 'ja')
        assert parse_language('fr') == ('French', 'fr')
        assert parse_language('de') == ('German', 'de')
        assert parse_language('haw') == ('Hawaiian', 'haw')
        assert parse_language('yue') == ('Cantonese', 'yue')

    def test_parse_language_names_to_codes(self):
        """Test that language names convert to correct codes."""
        assert parse_language('Japanese') == ('Japanese', 'ja')
        assert parse_language('Hawaiian') == ('Hawaiian', 'haw')
        assert parse_language('Cantonese') == ('Cantonese', 'yue')
        assert parse_language('Vietnamese') == ('Vietnamese', 'vi')

    def test_parse_language_aliases(self):
        """Test that language aliases work and preserve the alias name."""
        assert parse_language('Traditional Chinese') == ('Traditional Chinese', 'zh')
        assert parse_language('Taiwanese') == ('Taiwanese', 'zh')
        # Code 'zh' returns primary name 'Chinese'
        assert parse_language('zh') == ('Chinese', 'zh')


class TestGetLanguageName:
    """Tests for the get_language_name function."""

    def test_get_language_name_from_code(self):
        """Test getting language name from code."""
        assert get_language_name('ko') == 'Korean'
        assert get_language_name('en') == 'English'
        assert get_language_name('zh') == 'Chinese'

    def test_get_language_name_unknown_code(self):
        """Test fallback for unknown code returns the code itself."""
        assert get_language_name('xx') == 'xx'

    def test_get_language_name_case_insensitive(self):
        """Test code lookup is case-insensitive."""
        assert get_language_name('KO') == 'Korean'
        assert get_language_name('En') == 'English'


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_returns_defaults_when_no_file(self):
        """Test that defaults are returned when config file doesn't exist."""
        with patch('src.translator.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value.parent.parent.__truediv__.return_value = mock_path_instance

            # Re-import to use patched Path
            from src.translator import load_config as load_config_fresh

            with patch.object(Path, '__new__', return_value=mock_path_instance):
                config = load_config()

            assert 'ollama' in config
            assert config['ollama']['model'] == 'translategemma:4b'
            assert config['ollama']['base_url'] == 'http://localhost:11434'
            assert config['ollama']['batch_size'] == 50

    def test_load_config_merges_file_values(self):
        """Test that file values override defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary config file
            config_path = Path(tmpdir) / 'config.json'
            config_data = {
                "ollama": {
                    "model": "llama3:8b",
                    "batch_size": 25
                }
            }
            config_path.write_text(json.dumps(config_data))

            with patch('src.translator.Path') as mock_path_class:
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True

                # Mock __truediv__ to return our temp config path
                mock_path_class.return_value.parent.parent.__truediv__.return_value = config_path

                config = load_config()

            # Model should be from file, base_url should be default
            assert config['ollama']['model'] == 'llama3:8b'
            assert config['ollama']['base_url'] == 'http://localhost:11434'
            assert config['ollama']['batch_size'] == 25

    def test_load_config_auto_unload_defaults_to_false(self):
        """auto_unload should default to False when not in config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.json'
            config_path.write_text(json.dumps({"ollama": {"model": "llama3:8b"}}))

            with patch('src.translator.Path') as mock_path_class:
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True
                mock_path_class.return_value.parent.parent.__truediv__.return_value = config_path

                config = load_config()

            assert config['ollama']['auto_unload'] is False

    def test_load_config_reads_auto_unload_true(self):
        """auto_unload: true in config file should be read correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.json'
            config_path.write_text(json.dumps({"ollama": {"auto_unload": True}}))

            with patch('src.translator.Path') as mock_path_class:
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True
                mock_path_class.return_value.parent.parent.__truediv__.return_value = config_path

                config = load_config()

            assert config['ollama']['auto_unload'] is True


class TestOllamaTranslator:
    """Tests for the OllamaTranslator class."""

    @pytest.fixture
    def translator(self):
        """Create a translator instance with explicit config."""
        return OllamaTranslator(model='test-model', base_url='http://localhost:11434', batch_size=50)

    @pytest.fixture
    def sample_segments(self):
        """Sample subtitle segments for testing."""
        return [
            {'start': 0.0, 'end': 2.5, 'text': 'Hello, world!'},
            {'start': 2.5, 'end': 5.0, 'text': 'This is a test.'},
            {'start': 5.0, 'end': 8.3, 'text': 'Testing translation.'},
        ]

    def test_translator_initialization_with_explicit_values(self):
        """Test translator initializes with explicit model, base_url, and batch_size."""
        translator = OllamaTranslator(
            model='custom-model',
            base_url='http://custom:8080',
            batch_size=25
        )
        assert translator.model == 'custom-model'
        assert translator.base_url == 'http://custom:8080'
        assert translator.batch_size == 25

    def test_translator_initialization_uses_config_defaults(self):
        """Test translator uses config values when not provided."""
        with patch('src.translator.load_config') as mock_config:
            mock_config.return_value = {
                'ollama': {
                    'model': 'default-model',
                    'base_url': 'http://default:11434',
                    'batch_size': 30
                }
            }
            translator = OllamaTranslator()
            assert translator.model == 'default-model'
            assert translator.base_url == 'http://default:11434'
            assert translator.batch_size == 30

    def test_is_translategemma_true(self):
        """Test TranslateGemma detection returns True for translategemma models."""
        translator = OllamaTranslator(model='translategemma:4b', base_url='http://localhost:11434', batch_size=50)
        assert translator._is_translategemma() is True

        translator2 = OllamaTranslator(model='translategemma:12b', base_url='http://localhost:11434', batch_size=50)
        assert translator2._is_translategemma() is True

    def test_is_translategemma_false(self):
        """Test TranslateGemma detection returns False for other models."""
        translator = OllamaTranslator(model='qwen2.5:7b', base_url='http://localhost:11434', batch_size=50)
        assert translator._is_translategemma() is False

        translator2 = OllamaTranslator(model='llama3:8b', base_url='http://localhost:11434', batch_size=50)
        assert translator2._is_translategemma() is False

    def test_translategemma_prompt_format(self):
        """Test that TranslateGemma uses the correct prompt format."""
        translator = OllamaTranslator(model='translategemma:4b', base_url='http://localhost:11434', batch_size=50)
        prompt = translator._build_translategemma_prompt('Hello', 'English', 'Chinese')

        # "Chinese" is expanded to "Traditional Chinese (Taiwan, 繁體中文)" in prompts
        assert 'Traditional Chinese (Taiwan, 繁體中文) (zh) translator' in prompt
        assert 'English (en)' in prompt
        assert 'Hello' in prompt

    def test_preserve_linebreaks(self):
        """Test that linebreaks are preserved with delimiter."""
        translator = OllamaTranslator(model='test-model', base_url='http://localhost:11434', batch_size=50)

        text = "Line 1\nLine 2\nLine 3"
        preserved = translator._preserve_linebreaks(text)
        assert preserved == "Line 1 || Line 2 || Line 3"

        restored = translator._restore_linebreaks(preserved)
        assert restored == text

    def test_translate_text_multiline(self):
        """Test that multi-line text is translated with linebreaks preserved."""
        translator = OllamaTranslator(model='test-model', base_url='http://localhost:11434', batch_size=50)

        mock_response = Mock()
        # LLM returns translation with delimiter preserved
        mock_response.json.return_value = {'response': '你好 || 世界'}
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response):
            result = translator.translate_text(
                'Hello\nWorld',
                'English',
                'Chinese'
            )

        # Result should have linebreaks restored
        assert result == '你好\n世界'

    def test_translate_text_success(self, translator):
        """Test successful text translation."""
        mock_response = Mock()
        mock_response.json.return_value = {'response': '你好，世界！'}
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response):
            result = translator.translate_text(
                'Hello, world!',
                'English',
                'Chinese'
            )

        assert result == '你好，世界！'

    def test_translate_text_strips_whitespace(self, translator):
        """Test that translated text is stripped of whitespace."""
        mock_response = Mock()
        mock_response.json.return_value = {'response': '  你好，世界！  \n'}
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response):
            result = translator.translate_text(
                'Hello, world!',
                'English',
                'Chinese'
            )

        assert result == '你好，世界！'

    def test_translate_text_connection_error(self, translator):
        """Test handling of connection errors."""
        import requests as req

        with patch('src.translator.requests.post') as mock_post:
            mock_post.side_effect = req.exceptions.ConnectionError()

            with pytest.raises(ConnectionError) as exc_info:
                translator.translate_text('Hello', 'English', 'Chinese')

            assert 'Cannot connect to Ollama' in str(exc_info.value)

    def test_translate_text_timeout_error(self, translator):
        """Test handling of timeout errors."""
        import requests as req

        with patch('src.translator.requests.post') as mock_post:
            mock_post.side_effect = req.exceptions.Timeout()

            with pytest.raises(RuntimeError) as exc_info:
                translator.translate_text('Hello', 'English', 'Chinese')

            assert 'timed out' in str(exc_info.value)

    def test_translate_text_http_error_with_message(self, translator):
        """Test handling of HTTP errors with error message in response."""
        import requests as req

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {'error': "model 'qwen2.5:7b' not found"}

        http_error = req.exceptions.HTTPError(response=mock_response)

        with patch('src.translator.requests.post') as mock_post:
            mock_post.return_value.raise_for_status.side_effect = http_error

            with pytest.raises(RuntimeError) as exc_info:
                translator.translate_text('Hello', 'English', 'Chinese')

            assert "model 'qwen2.5:7b' not found" in str(exc_info.value)
            assert 'Ollama API error' in str(exc_info.value)

    def test_translate_segments_empty_list(self, translator):
        """Test translating empty segment list."""
        result = translator.translate_segments([], 'English', 'Chinese')
        assert result == []

    def test_check_connection_success(self, translator):
        """Test successful connection check."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch('src.translator.requests.get', return_value=mock_response):
            result = translator.check_connection()

        assert result is True

    def test_check_connection_failure(self, translator):
        """Test failed connection check."""
        import requests as req

        with patch('src.translator.requests.get') as mock_get:
            mock_get.side_effect = req.exceptions.ConnectionError()
            result = translator.check_connection()

        assert result is False

    def test_check_connection_non_200_status(self, translator):
        """Test connection check with non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500

        with patch('src.translator.requests.get', return_value=mock_response):
            result = translator.check_connection()

        assert result is False

    def test_translate_text_sends_correct_prompt(self, translator):
        """Test that the correct prompt is sent to Ollama."""
        mock_response = Mock()
        mock_response.json.return_value = {'response': 'translated'}
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response) as mock_post:
            translator.translate_text('Hello', 'English', 'Spanish')

            # Verify the call
            call_args = mock_post.call_args
            assert call_args[0][0] == 'http://localhost:11434/api/generate'

            json_data = call_args[1]['json']
            assert json_data['model'] == 'test-model'
            assert 'English' in json_data['prompt']
            assert 'Spanish' in json_data['prompt']
            assert 'Hello' in json_data['prompt']
            assert json_data['stream'] is False


class TestBatchTranslation:
    """Tests for batch translation functionality."""

    @pytest.fixture
    def translator(self):
        """Create a translator instance with small batch size for testing."""
        return OllamaTranslator(model='test-model', base_url='http://localhost:11434', batch_size=3)

    @pytest.fixture
    def sample_segments(self):
        """Sample subtitle segments for testing."""
        return [
            {'start': 0.0, 'end': 2.5, 'text': 'Hello, world!'},
            {'start': 2.5, 'end': 5.0, 'text': 'This is a test.'},
            {'start': 5.0, 'end': 8.3, 'text': 'Testing translation.'},
        ]

    def test_build_batch_prompt(self, translator):
        """Test that batch prompt is correctly formatted."""
        texts = ['Hello', 'World', 'Test']
        prompt = translator._build_batch_prompt(texts, 'English', 'Chinese')

        assert 'English' in prompt
        assert 'Chinese' in prompt
        assert '1. Hello' in prompt
        assert '2. World' in prompt
        assert '3. Test' in prompt

    def test_parse_batch_response_success(self, translator):
        """Test successful parsing of batch response."""
        response = """1. 你好
2. 世界
3. 测试"""
        result = translator._parse_batch_response(response, 3)

        assert result == ['你好', '世界', '测试']

    def test_parse_batch_response_with_extra_whitespace(self, translator):
        """Test parsing handles extra whitespace."""
        response = """
1.  你好
2.   世界
3. 测试
"""
        result = translator._parse_batch_response(response, 3)

        assert result == ['你好', '世界', '测试']

    def test_parse_batch_response_missing_line(self, translator):
        """Test parsing returns None when a line is missing."""
        response = """1. 你好
3. 测试"""
        result = translator._parse_batch_response(response, 3)

        assert result is None

    def test_parse_batch_response_wrong_count(self, translator):
        """Test parsing returns None when count doesn't match."""
        response = """1. 你好
2. 世界"""
        result = translator._parse_batch_response(response, 3)

        assert result is None

    def test_try_translate_batch_success(self, translator, sample_segments):
        """Test successful batch translation."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': '1. 你好，世界！\n2. 这是一个测试。\n3. 测试翻译。'
        }
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response):
            result = translator._try_translate_batch(sample_segments, 'English', 'Chinese')

        assert result is not None
        assert len(result) == 3
        assert result[0]['text'] == '你好，世界！'
        assert result[1]['text'] == '这是一个测试。'
        assert result[2]['text'] == '测试翻译。'

    def test_try_translate_batch_preserves_timestamps(self, translator, sample_segments):
        """Test that batch translation preserves timestamps."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': '1. Translation 1\n2. Translation 2\n3. Translation 3'
        }
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response):
            result = translator._try_translate_batch(sample_segments, 'English', 'Chinese')

        assert result[0]['start'] == 0.0
        assert result[0]['end'] == 2.5
        assert result[1]['start'] == 2.5
        assert result[1]['end'] == 5.0
        assert result[2]['start'] == 5.0
        assert result[2]['end'] == 8.3

    def test_try_translate_batch_returns_none_on_parse_failure(self, translator, sample_segments):
        """Test that batch translation returns None when parsing fails."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': 'Invalid response without numbers'
        }
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response):
            result = translator._try_translate_batch(sample_segments, 'English', 'Chinese')

        assert result is None

    def test_try_translate_batch_raises_on_connection_error(self, translator, sample_segments):
        """Test that batch translation raises ConnectionError on connection error."""
        import requests as req

        with patch('src.translator.requests.post') as mock_post:
            mock_post.side_effect = req.exceptions.ConnectionError()

            with pytest.raises(ConnectionError):
                translator._try_translate_batch(sample_segments, 'English', 'Chinese')

    def test_translate_batch_recursive_success(self, translator, sample_segments):
        """Test recursive batch translation succeeds on first try."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': '1. Translation 1\n2. Translation 2\n3. Translation 3'
        }
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response):
            result = translator._translate_batch_recursive(
                sample_segments, 'English', 'Chinese', total_segments=3
            )

        assert len(result) == 3
        assert result[0]['text'] == 'Translation 1'
        assert result[1]['text'] == 'Translation 2'
        assert result[2]['text'] == 'Translation 3'

    def test_translate_batch_recursive_splits_on_failure(self, translator, sample_segments):
        """Test recursive batch splits in half when batch fails."""
        call_count = [0]

        def mock_try_batch(segments, src, tgt, context=None):
            call_count[0] += 1
            if len(segments) == 3:
                return None  # Fail for full batch
            # Succeed for smaller batches
            return [
                {'start': s['start'], 'end': s['end'], 'text': f"Translated: {s['text']}"}
                for s in segments
            ]

        with patch.object(translator, '_try_translate_batch', side_effect=mock_try_batch):
            with patch.object(translator, 'translate_text', return_value='Single translated'):
                result = translator._translate_batch_recursive(
                    sample_segments, 'English', 'Chinese', total_segments=3
                )

        # Should have tried full batch, then split
        assert len(result) == 3
        # Verify splitting happened (more than 1 call)
        assert call_count[0] > 1

    def test_translate_batch_recursive_falls_back_to_single(self, translator):
        """Test recursive batch falls back to single translation when batch size is 1."""
        single_segment = [{'start': 0.0, 'end': 2.5, 'text': 'Hello'}]

        # Make batch translation fail
        with patch.object(translator, '_try_translate_batch', return_value=None):
            with patch.object(translator, 'translate_text', return_value='Translated Hello'):
                result = translator._translate_batch_recursive(
                    single_segment, 'English', 'Chinese', total_segments=1
                )

        assert len(result) == 1
        assert result[0]['text'] == 'Translated Hello'
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == 2.5

    def test_translate_batch_recursive_raises_on_total_failure(self, translator):
        """Test that errors propagate when all translation attempts fail."""
        single_segment = [{'start': 0.0, 'end': 2.5, 'text': 'Hello'}]

        # Make all translation attempts fail
        with patch.object(translator, '_try_translate_batch', return_value=None):
            with patch.object(translator, 'translate_text', side_effect=RuntimeError("Failed")):
                with pytest.raises(RuntimeError) as exc_info:
                    translator._translate_batch_recursive(
                        single_segment, 'English', 'Chinese', total_segments=1
                    )

                assert "Failed" in str(exc_info.value)

    def test_translate_segments_uses_batching(self, translator, sample_segments):
        """Test that translate_segments uses batch processing."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': '1. T1\n2. T2\n3. T3'
        }
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response) as mock_post:
            result = translator.translate_segments(sample_segments, 'English', 'Chinese')

        # Should have made one batch call (batch_size=3, segments=3)
        assert mock_post.call_count == 1
        assert len(result) == 3

    def test_translate_segments_multiple_batches(self):
        """Test translation with multiple batches."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=2)
        segments = [
            {'start': 0.0, 'end': 1.0, 'text': 'One'},
            {'start': 1.0, 'end': 2.0, 'text': 'Two'},
            {'start': 2.0, 'end': 3.0, 'text': 'Three'},
        ]

        call_count = [0]

        def mock_try_batch(segs, src, tgt, context=None):
            call_count[0] += 1
            return [
                {'start': s['start'], 'end': s['end'], 'text': f"T{i+1}"}
                for i, s in enumerate(segs)
            ]

        with patch.object(translator, '_try_translate_batch', side_effect=mock_try_batch):
            result = translator.translate_segments(segments, 'English', 'Chinese')

        # With batch_size=2 and 3 segments, should have 2 batches
        assert call_count[0] == 2
        assert len(result) == 3

    def test_translate_segments_calls_progress_callback(self, translator, sample_segments):
        """Test that progress callback is called during translation."""
        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        mock_response = Mock()
        mock_response.json.return_value = {
            'response': '1. T1\n2. T2\n3. T3'
        }
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response):
            translator.translate_segments(
                sample_segments,
                'English',
                'Chinese',
                progress_callback=progress_callback
            )

        # Progress should be called at least once
        assert len(progress_calls) > 0
        # Final call should show completion
        assert progress_calls[-1] == (3, 3)


class TestContextWindow:
    """Tests for sliding context window feature in batch translation."""

    # ── 1. Config tests ──────────────────────────────────────────────────────

    def test_load_config_default_context_lines(self):
        """context_lines should default to 3 when not in config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.json'
            config_path.write_text(json.dumps({"ollama": {"model": "test"}}))

            with patch('src.translator.Path') as mock_path_class:
                mock_path_class.return_value.parent.parent.__truediv__.return_value = config_path
                config = load_config()

        assert config['ollama']['context_lines'] == 3

    def test_load_config_custom_context_lines(self):
        """context_lines in config file should override default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.json'
            config_path.write_text(json.dumps({"ollama": {"context_lines": 5}}))

            with patch('src.translator.Path') as mock_path_class:
                mock_path_class.return_value.parent.parent.__truediv__.return_value = config_path
                config = load_config()

        assert config['ollama']['context_lines'] == 5

    def test_translator_init_context_lines_zero(self):
        """context_lines=0 must be stored as 0, not treated as falsy."""
        with patch('src.translator.load_config') as mock_config:
            mock_config.return_value = {
                'ollama': {
                    'model': 'test',
                    'base_url': 'http://localhost:11434',
                    'batch_size': 50,
                    'keep_alive': '10m',
                    'context_lines': 0,
                }
            }
            translator = OllamaTranslator()
        assert translator.context_lines == 0

    # ── 2. Prompt building tests ──────────────────────────────────────────────

    def test_build_batch_prompt_no_context(self):
        """Prompt is unchanged when context is None or empty."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=50)
        texts = ['Hello', 'World']

        prompt_none = translator._build_batch_prompt(texts, 'English', 'Chinese')
        prompt_empty = translator._build_batch_prompt(texts, 'English', 'Chinese', context=[])

        assert 'Previous translations' not in prompt_none
        assert prompt_none == prompt_empty

    def test_build_batch_prompt_with_context(self):
        """Context block appears before the numbered lines."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=50)
        context = [('Hello', '你好'), ('World', '世界')]
        texts = ['Goodbye']

        prompt = translator._build_batch_prompt(texts, 'English', 'Chinese', context=context)

        assert 'Previous translations' in prompt
        assert '1. Goodbye' in prompt
        context_pos = prompt.index('Previous translations')
        numbered_pos = prompt.index('1. Goodbye')
        assert context_pos < numbered_pos

    def test_build_batch_prompt_context_format(self):
        """Context pairs must use exact `"orig" → "trans"` format."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=50)
        context = [('Hello world', '你好世界'), ('How are you', '你好嗎')]
        texts = ['Fine']

        prompt = translator._build_batch_prompt(texts, 'English', 'Chinese', context=context)

        assert '"Hello world" → "你好世界"' in prompt
        assert '"How are you" → "你好嗎"' in prompt

    # ── 3. Threading / call-chain tests ──────────────────────────────────────

    def test_try_translate_batch_passes_context(self):
        """_try_translate_batch must forward context to _build_batch_prompt."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=50)
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}]
        context = [('Hi', '嗨')]

        mock_response = Mock()
        mock_response.json.return_value = {'response': '1. 你好'}
        mock_response.raise_for_status = Mock()

        with patch.object(translator, '_build_batch_prompt', wraps=translator._build_batch_prompt) as mock_build:
            with patch('src.translator.requests.post', return_value=mock_response):
                translator._try_translate_batch(segments, 'English', 'Chinese', context=context)

        mock_build.assert_called_once_with(['Hello'], 'English', 'Chinese', context=context)

    def test_translate_batch_recursive_split_uses_original_context(self):
        """Both halves of a recursive split get the same original context."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=50)
        segments = [
            {'start': 0.0, 'end': 1.0, 'text': 'A'},
            {'start': 1.0, 'end': 2.0, 'text': 'B'},
        ]
        context = [('Prior', '先前')]
        received_contexts = []

        def mock_try_batch(segs, src, tgt, context=None):
            received_contexts.append(context)
            if len(segs) == 2:
                return None  # force split
            return [{'start': s['start'], 'end': s['end'], 'text': 'T'} for s in segs]

        with patch.object(translator, '_try_translate_batch', side_effect=mock_try_batch):
            translator._translate_batch_recursive(
                segments, 'English', 'Chinese', context=context
            )

        # Full batch (fails) + left half + right half = 3 calls
        assert len(received_contexts) == 3
        # Every call must receive the original context unchanged
        assert all(c == context for c in received_contexts)

    # ── 4. Accumulation tests ─────────────────────────────────────────────────

    def test_translate_segments_first_batch_empty_context(self):
        """First batch must always receive an empty context list."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=3)
        translator.context_lines = 3
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}]
        received_contexts = []

        def mock_recursive(segs, src, tgt, progress_callback=None,
                           progress_offset=0, total_segments=0, context=None):
            received_contexts.append(context)
            return [{'start': s['start'], 'end': s['end'], 'text': 'T'} for s in segs]

        with patch.object(translator, '_translate_batch_recursive', side_effect=mock_recursive):
            translator.translate_segments(segments, 'English', 'Chinese')

        assert received_contexts[0] == []

    def test_translate_segments_second_batch_gets_context(self):
        """Second batch receives the tail of context accumulated from first batch."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=2)
        translator.context_lines = 3
        segments = [
            {'start': 0.0, 'end': 1.0, 'text': 'One'},
            {'start': 1.0, 'end': 2.0, 'text': 'Two'},
            {'start': 2.0, 'end': 3.0, 'text': 'Three'},
        ]
        received_contexts = []

        def mock_recursive(segs, src, tgt, progress_callback=None,
                           progress_offset=0, total_segments=0, context=None):
            received_contexts.append(list(context))
            return [{'start': s['start'], 'end': s['end'], 'text': f"T_{s['text']}"} for s in segs]

        with patch.object(translator, '_translate_batch_recursive', side_effect=mock_recursive):
            translator.translate_segments(segments, 'English', 'Chinese')

        assert received_contexts[0] == []
        assert received_contexts[1] == [('One', 'T_One'), ('Two', 'T_Two')]

    def test_translate_segments_context_lines_zero_disables(self):
        """context_lines=0 passes empty context to every batch."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=2)
        translator.context_lines = 0
        segments = [
            {'start': 0.0, 'end': 1.0, 'text': 'One'},
            {'start': 1.0, 'end': 2.0, 'text': 'Two'},
            {'start': 2.0, 'end': 3.0, 'text': 'Three'},
        ]
        received_contexts = []

        def mock_recursive(segs, src, tgt, progress_callback=None,
                           progress_offset=0, total_segments=0, context=None):
            received_contexts.append(list(context))
            return [{'start': s['start'], 'end': s['end'], 'text': 'T'} for s in segs]

        with patch.object(translator, '_translate_batch_recursive', side_effect=mock_recursive):
            translator.translate_segments(segments, 'English', 'Chinese')

        assert all(c == [] for c in received_contexts)

    def test_translate_segments_context_sliced_to_context_lines(self):
        """Context passed to each batch is limited to last context_lines pairs."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=1)
        translator.context_lines = 2
        segments = [
            {'start': float(i), 'end': float(i + 1), 'text': f'S{i}'}
            for i in range(4)
        ]
        received_contexts = []

        def mock_recursive(segs, src, tgt, progress_callback=None,
                           progress_offset=0, total_segments=0, context=None):
            received_contexts.append(list(context))
            return [{'start': s['start'], 'end': s['end'], 'text': f"T_{s['text']}"} for s in segs]

        with patch.object(translator, '_translate_batch_recursive', side_effect=mock_recursive):
            translator.translate_segments(segments, 'English', 'Chinese')

        # Batch 0: no prior results
        assert received_contexts[0] == []
        # Batch 1: only the 1 result so far (< context_lines)
        assert received_contexts[1] == [('S0', 'T_S0')]
        # Batch 2: last 2 results
        assert received_contexts[2] == [('S0', 'T_S0'), ('S1', 'T_S1')]
        # Batch 3: last 2 results (window slides)
        assert received_contexts[3] == [('S1', 'T_S1'), ('S2', 'T_S2')]


class TestCustomPrompt:
    """Tests for custom prompt file support in translation."""

    def test_translator_init_custom_prompt_none_by_default(self):
        """custom_prompt defaults to None when not provided."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=50)
        assert translator.custom_prompt is None

    def test_translator_init_custom_prompt_from_arg(self):
        """custom_prompt can be set via constructor argument."""
        translator = OllamaTranslator(
            model='test', base_url='http://localhost:11434', batch_size=50,
            custom_prompt='Use formal tone. Translate "pod" as Pod.'
        )
        assert translator.custom_prompt == 'Use formal tone. Translate "pod" as Pod.'

    def test_build_batch_prompt_includes_custom_prompt(self):
        """Custom prompt text appears in batch prompt."""
        translator = OllamaTranslator(
            model='test', base_url='http://localhost:11434', batch_size=50,
            custom_prompt='Always use formal tone.'
        )
        texts = ['Hello', 'World']
        prompt = translator._build_batch_prompt(texts, 'English', 'Chinese')

        assert 'Always use formal tone.' in prompt
        assert '1. Hello' in prompt

    def test_build_batch_prompt_no_custom_prompt(self):
        """Prompt is unchanged when custom_prompt is None."""
        translator = OllamaTranslator(model='test', base_url='http://localhost:11434', batch_size=50)
        texts = ['Hello']
        prompt = translator._build_batch_prompt(texts, 'English', 'Chinese')

        assert 'Additional instructions' not in prompt

    def test_build_translategemma_prompt_includes_custom_prompt(self):
        """Custom prompt text appears in TranslateGemma single-text prompt."""
        translator = OllamaTranslator(
            model='translategemma:4b', base_url='http://localhost:11434', batch_size=50,
            custom_prompt='Translate "container" as 容器.'
        )
        prompt = translator._build_translategemma_prompt('Hello', 'English', 'Chinese')

        assert 'Translate "container" as 容器.' in prompt
        assert 'Hello' in prompt

    def test_translate_text_non_translategemma_includes_custom_prompt(self):
        """Custom prompt appears in non-TranslateGemma single-text prompt."""
        translator = OllamaTranslator(
            model='llama3:8b', base_url='http://localhost:11434', batch_size=50,
            custom_prompt='Use casual tone.'
        )

        mock_response = Mock()
        mock_response.json.return_value = {'response': 'translated'}
        mock_response.raise_for_status = Mock()

        with patch('src.translator.requests.post', return_value=mock_response) as mock_post:
            translator.translate_text('Hello', 'English', 'Chinese')

        sent_prompt = mock_post.call_args[1]['json']['prompt']
        assert 'Use casual tone.' in sent_prompt

    def test_build_batch_prompt_custom_prompt_before_numbered_lines(self):
        """Custom prompt should appear before the numbered lines to translate."""
        translator = OllamaTranslator(
            model='test', base_url='http://localhost:11434', batch_size=50,
            custom_prompt='Glossary: API=介面'
        )
        texts = ['Hello']
        prompt = translator._build_batch_prompt(texts, 'English', 'Chinese')

        custom_pos = prompt.index('Glossary: API=介面')
        numbered_pos = prompt.index('1. Hello')
        assert custom_pos < numbered_pos


class TestPromptFileConfig:
    """Tests for ollama.prompt_file config option."""

    def test_load_config_prompt_file_default_none(self):
        """prompt_file defaults to None when not in config."""
        with patch('src.translator.Path.exists', return_value=False):
            config = load_config()
        assert config['ollama'].get('prompt_file') is None

    def test_load_config_prompt_file_from_file(self):
        """prompt_file is loaded from config.json when present."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"ollama": {"prompt_file": "prompts/glossary.txt"}}, f)
            f.flush()
            with patch.object(Path, 'exists', return_value=True):
                with patch('builtins.open', Mock(return_value=open(f.name))):
                    config = load_config()
        assert config['ollama']['prompt_file'] == 'prompts/glossary.txt'

    def test_translator_uses_config_prompt_file(self, tmp_path):
        """OllamaTranslator reads prompt_file from config when custom_prompt not passed."""
        prompt_content = "Use formal tone.\nGlossary: API=介面"
        prompt_path = tmp_path / "glossary.txt"
        prompt_path.write_text(prompt_content)

        config = {
            "ollama": {
                "model": "test",
                "base_url": "http://localhost:11434",
                "batch_size": 50,
                "keep_alive": "10m",
                "context_lines": 3,
                "prompt_file": str(prompt_path),
            },
            "output": {"directory": None},
        }
        with patch('src.translator.load_config', return_value=config):
            translator = OllamaTranslator()
        assert translator.custom_prompt == prompt_content

    def test_translator_stores_prompt_file_source_from_config(self, tmp_path):
        """prompt_file_source is set to config path when loaded from config."""
        prompt_path = tmp_path / "glossary.txt"
        prompt_path.write_text("Some instructions")

        config = {
            "ollama": {
                "model": "test",
                "base_url": "http://localhost:11434",
                "batch_size": 50,
                "keep_alive": "10m",
                "context_lines": 3,
                "prompt_file": str(prompt_path),
            },
            "output": {"directory": None},
        }
        with patch('src.translator.load_config', return_value=config):
            translator = OllamaTranslator()
        assert translator.prompt_file_source == str(prompt_path)

    def test_translator_prompt_file_source_none_when_cli(self, tmp_path):
        """prompt_file_source is None when custom_prompt is passed directly."""
        prompt_path = tmp_path / "glossary.txt"
        prompt_path.write_text("Config content")

        config = {
            "ollama": {
                "model": "test",
                "base_url": "http://localhost:11434",
                "batch_size": 50,
                "keep_alive": "10m",
                "context_lines": 3,
                "prompt_file": str(prompt_path),
            },
            "output": {"directory": None},
        }
        with patch('src.translator.load_config', return_value=config):
            translator = OllamaTranslator(custom_prompt="CLI content")
        assert translator.prompt_file_source is None

    def test_translator_prompt_file_source_none_when_no_config(self):
        """prompt_file_source is None when no prompt_file in config."""
        config = {
            "ollama": {
                "model": "test",
                "base_url": "http://localhost:11434",
                "batch_size": 50,
                "keep_alive": "10m",
                "context_lines": 3,
            },
            "output": {"directory": None},
        }
        with patch('src.translator.load_config', return_value=config):
            translator = OllamaTranslator()
        assert translator.prompt_file_source is None

    def test_translator_explicit_custom_prompt_overrides_config(self, tmp_path):
        """Explicit custom_prompt arg takes precedence over config prompt_file."""
        prompt_path = tmp_path / "glossary.txt"
        prompt_path.write_text("Config prompt content")

        config = {
            "ollama": {
                "model": "test",
                "base_url": "http://localhost:11434",
                "batch_size": 50,
                "keep_alive": "10m",
                "context_lines": 3,
                "prompt_file": str(prompt_path),
            },
            "output": {"directory": None},
        }
        with patch('src.translator.load_config', return_value=config):
            translator = OllamaTranslator(custom_prompt="CLI prompt content")
        assert translator.custom_prompt == "CLI prompt content"

    def test_translator_config_prompt_file_missing_file_ignored(self):
        """Non-existent prompt_file in config is silently ignored."""
        config = {
            "ollama": {
                "model": "test",
                "base_url": "http://localhost:11434",
                "batch_size": 50,
                "keep_alive": "10m",
                "context_lines": 3,
                "prompt_file": "/nonexistent/path/glossary.txt",
            },
            "output": {"directory": None},
        }
        with patch('src.translator.load_config', return_value=config):
            translator = OllamaTranslator()
        assert translator.custom_prompt is None

    def test_translator_config_prompt_file_empty_file_ignored(self, tmp_path):
        """Empty prompt file in config results in None custom_prompt."""
        prompt_path = tmp_path / "empty.txt"
        prompt_path.write_text("   \n  ")

        config = {
            "ollama": {
                "model": "test",
                "base_url": "http://localhost:11434",
                "batch_size": 50,
                "keep_alive": "10m",
                "context_lines": 3,
                "prompt_file": str(prompt_path),
            },
            "output": {"directory": None},
        }
        with patch('src.translator.load_config', return_value=config):
            translator = OllamaTranslator()
        assert translator.custom_prompt is None


class TestUnloadAllModels:
    """Tests for the unload_all_models function."""

    @patch('src.translator.requests.get')
    def test_returns_zero_when_no_models_loaded(self, mock_get):
        """Should return 0 and not call generate when no models are loaded."""
        from src.translator import unload_all_models
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {'models': []},
        )

        assert unload_all_models('http://localhost:11434') == 0

    @patch('src.translator.requests.post')
    @patch('src.translator.requests.get')
    def test_unloads_each_loaded_model(self, mock_get, mock_post):
        """Should call generate with keep_alive=0 for every loaded model."""
        from src.translator import unload_all_models
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {'models': [{'model': 'llama3:8b'}, {'model': 'gemma:2b'}]},
        )
        mock_post.return_value = Mock(status_code=200, json=lambda: {'response': ''})

        n = unload_all_models('http://localhost:11434')

        assert n == 2
        assert mock_post.call_count == 2

    @patch('src.translator.requests.post')
    @patch('src.translator.requests.get')
    def test_unload_request_uses_keep_alive_zero(self, mock_get, mock_post):
        """Unload request must pass keep_alive=0 for the correct model."""
        from src.translator import unload_all_models
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {'models': [{'model': 'llama3:8b'}]},
        )
        mock_post.return_value = Mock(status_code=200, json=lambda: {'response': ''})

        unload_all_models('http://localhost:11434')

        payload = mock_post.call_args[1]['json']
        assert payload['keep_alive'] == 0
        assert payload['model'] == 'llama3:8b'

    @patch('src.translator.requests.get')
    def test_returns_zero_on_connection_error(self, mock_get):
        """Should return 0 silently when Ollama is not reachable."""
        import requests as req
        from src.translator import unload_all_models
        mock_get.side_effect = req.exceptions.ConnectionError()

        assert unload_all_models('http://localhost:11434') == 0

    @patch('src.translator.requests.get')
    def test_returns_zero_on_any_request_exception(self, mock_get):
        """Should return 0 silently on any requests error."""
        import requests as req
        from src.translator import unload_all_models
        mock_get.side_effect = req.exceptions.RequestException()

        assert unload_all_models('http://localhost:11434') == 0
