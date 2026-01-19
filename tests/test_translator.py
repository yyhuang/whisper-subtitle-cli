import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from src.translator import OllamaTranslator, load_config


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
            assert config['ollama']['model'] == 'qwen2.5:7b'
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

    def test_try_translate_batch_returns_none_on_connection_error(self, translator, sample_segments):
        """Test that batch translation returns None on connection error."""
        import requests as req

        with patch('src.translator.requests.post') as mock_post:
            mock_post.side_effect = req.exceptions.ConnectionError()
            result = translator._try_translate_batch(sample_segments, 'English', 'Chinese')

        assert result is None

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

        def mock_try_batch(segments, src, tgt):
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

    def test_translate_batch_recursive_keeps_original_on_total_failure(self, translator):
        """Test that original text is kept when all translation attempts fail."""
        single_segment = [{'start': 0.0, 'end': 2.5, 'text': 'Hello'}]

        import requests as req

        # Make all translation attempts fail
        with patch.object(translator, '_try_translate_batch', return_value=None):
            with patch.object(translator, 'translate_text', side_effect=RuntimeError("Failed")):
                result = translator._translate_batch_recursive(
                    single_segment, 'English', 'Chinese', total_segments=1
                )

        assert len(result) == 1
        assert result[0]['text'] == 'Hello'  # Original text preserved

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

        def mock_try_batch(segs, src, tgt):
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
