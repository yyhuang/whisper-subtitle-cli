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
    """Transcribes audio files using openai-whisper, mlx-whisper, or stable-ts."""

    def __init__(self, model_size: str = "medium", use_stable: bool = False, use_vad: bool = False):
        """
        Initialize the transcriber with a Whisper model.
        Automatically detects the best backend and device.

        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            use_stable: If True, use stable-ts for better timestamp accuracy
            use_vad: If True, use VAD (Voice Activity Detection) with stable-ts.
                     Requires --stable flag. Uses Silero VAD (neural network based).
        """
        self.model_size = model_size
        self.use_stable = use_stable
        self.use_vad = use_vad

        # VAD requires stable-ts
        if use_vad and not use_stable:
            raise ValueError("--vad requires --stable flag. VAD is only supported with stable-ts backend.")

        self.backend, self.device, self.compute_type = self._detect_backend()
        self.model = None

    def _detect_backend(self):
        """Detect the best backend based on hardware and use_stable flag."""
        # With --stable flag: use stable-ts
        if self.use_stable:
            try:
                import stable_whisper  # noqa: F401
                # Apple Silicon: use stable-ts MLX backend
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    return "stable-ts-mlx", "mlx", "stable-ts (MLX)"
                # Others: use stable-ts with PyTorch
                import torch
                if torch.cuda.is_available():
                    return "stable-ts", "cuda", "float16"
                return "stable-ts", "cpu", "float32"
            except ImportError:
                raise ImportError(
                    "stable-ts not installed. Run: uv sync --extra stable"
                )

        # Without --stable: use standard backends
        # Apple Silicon: prefer mlx-whisper for Metal GPU
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx_whisper  # noqa: F401
                return "mlx", "mlx", "Apple Silicon"
            except ImportError:
                pass

        # Everyone else: use openai-whisper with PyTorch
        import torch
        if torch.cuda.is_available():
            return "openai-whisper", "cuda", "float16"
        return "openai-whisper", "cpu", "float32"

    def _load_model(self):
        """Lazy load the Whisper model when needed."""
        if self.model is None:
            if self.backend == "openai-whisper":
                import whisper
                self.model = whisper.load_model(self.model_size, device=self.device)
            elif self.backend == "stable-ts":
                import stable_whisper
                self.model = stable_whisper.load_model(self.model_size, device=self.device)

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
            elif self.backend == "stable-ts":
                return self._transcribe_stable_ts(audio_path, language)
            elif self.backend == "stable-ts-mlx":
                return self._transcribe_stable_ts_mlx(audio_path, language)
            else:
                return self._transcribe_openai_whisper(audio_path, language)
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

    def _transcribe_openai_whisper(self, audio_path: str, language: Optional[str]) -> List[Dict]:
        """Transcribe using openai-whisper."""
        self._load_model()

        kwargs = {}
        if language:
            kwargs["language"] = language

        output = self.model.transcribe(audio_path, **kwargs)

        result = []
        for segment in output["segments"]:
            result.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        return result

    def _transcribe_stable_ts(self, audio_path: str, language: Optional[str]) -> List[Dict]:
        """Transcribe using stable-ts (CUDA/CPU)."""
        self._load_model()

        kwargs = {}
        if language:
            kwargs["language"] = language
        if self.use_vad:
            kwargs["vad"] = True  # Uses Silero VAD

        # stable-ts returns a WhisperResult object
        output = self.model.transcribe(audio_path, **kwargs)

        return self._format_stable_ts_segments(output)

    def _transcribe_stable_ts_mlx(self, audio_path: str, language: Optional[str]) -> List[Dict]:
        """Transcribe using stable-ts with MLX backend."""
        import stable_whisper

        model_repo = MLX_MODEL_MAP.get(self.model_size, MLX_MODEL_MAP["medium"])

        kwargs = {}
        if language:
            kwargs["language"] = language
        if self.use_vad:
            kwargs["vad"] = True  # Uses Silero VAD

        # stable-ts MLX uses transcribe_with_path for MLX models
        output = stable_whisper.transcribe_with_path(model_repo, audio_path, **kwargs)

        return self._format_stable_ts_segments(output)

    def _format_stable_ts_segments(self, output) -> List[Dict]:
        """Format stable-ts WhisperResult to our segment format."""
        result = []
        for segment in output.segments:
            result.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        return result
