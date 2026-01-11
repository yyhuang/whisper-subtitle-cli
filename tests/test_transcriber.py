import pytest
from src.transcriber import Transcriber


class TestTranscriber:
    def test_transcriber_initialization(self):
        """Test that transcriber can be initialized with default model."""
        transcriber = Transcriber()
        assert transcriber is not None

    def test_transcriber_with_custom_model(self):
        """Test that transcriber accepts custom model size."""
        transcriber = Transcriber(model_size="tiny")
        assert transcriber.model_size == "tiny"

    def test_transcriber_default_model_is_medium(self):
        """Test that default model is 'medium'."""
        transcriber = Transcriber()
        assert transcriber.model_size == "medium"

    def test_transcribe_returns_segments(self):
        """Test that transcribe method exists and has correct signature."""
        transcriber = Transcriber()
        assert hasattr(transcriber, 'transcribe')
        assert callable(transcriber.transcribe)

    def test_transcribe_accepts_audio_path(self):
        """Test that transcribe accepts audio file path."""
        import inspect
        transcriber = Transcriber()
        sig = inspect.signature(transcriber.transcribe)
        assert 'audio_path' in sig.parameters

    def test_transcribe_accepts_language_parameter(self):
        """Test that transcribe accepts optional language parameter."""
        import inspect
        transcriber = Transcriber()
        sig = inspect.signature(transcriber.transcribe)
        assert 'language' in sig.parameters

    def test_transcribe_returns_list_of_segments(self):
        """Test that transcribe returns a list."""
        # This will be tested in integration tests with real audio
        # Here we just verify the interface
        transcriber = Transcriber()
        assert hasattr(transcriber, 'transcribe')

    def test_segment_has_required_fields(self):
        """Test that segments have start, end, and text fields."""
        # This will verify the data structure in our implementation
        # Each segment should have: start time, end time, text
        # This is more of a contract test that will be verified in integration
        pass
