import platform
from typing import List, Dict, Optional


MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}


class Transcriber:
    """Transcribes audio files using Faster Whisper or mlx-whisper."""

    def __init__(self, model_size: str = "medium"):
        """
        Initialize the transcriber with a Whisper model.
        Automatically detects the best backend and device.

        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.backend, self.device, self.compute_type = self._detect_backend()
        self.model = None

    @staticmethod
    def _detect_backend():
        """Detect the best backend: mlx on Apple Silicon, faster-whisper otherwise."""
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx_whisper  # noqa: F401
                return "mlx", "mlx", "Apple Silicon"
            except ImportError:
                pass

        # Fall back to faster-whisper with CUDA detection
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "faster-whisper", "cuda", "float16"
        except Exception:
            pass
        return "faster-whisper", "cpu", "int8"

    def _load_model(self):
        """Lazy load the Whisper model when needed."""
        if self.model is None and self.backend == "faster-whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'zh'). None for auto-detect.

        Returns:
            List of segments, each containing:
                - start: Start time in seconds
                - end: End time in seconds
                - text: Transcribed text

        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription fails
        """
        import os
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            if self.backend == "mlx":
                return self._transcribe_mlx(audio_path, language)
            else:
                return self._transcribe_faster_whisper(audio_path, language)
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")

    def _transcribe_mlx(self, audio_path: str, language: Optional[str]) -> List[Dict]:
        """Transcribe using mlx-whisper."""
        import mlx_whisper

        model_repo = MLX_MODEL_MAP.get(self.model_size, MLX_MODEL_MAP["medium"])

        kwargs = {"path_or_hf_repo": model_repo}
        if language:
            kwargs["language"] = language

        output = mlx_whisper.transcribe(audio_path, **kwargs)

        result = []
        for segment in output["segments"]:
            result.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        return result

    def _transcribe_faster_whisper(self, audio_path: str, language: Optional[str]) -> List[Dict]:
        """Transcribe using faster-whisper."""
        self._load_model()

        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        result = []
        for segment in segments:
            result.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        return result
