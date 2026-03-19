"""Voice transcription providers for multiple services."""

import os
from pathlib import Path
from typing import Protocol

import httpx
from loguru import logger


class TranscriptionProvider(Protocol):
    """Protocol for transcription providers."""

    async def transcribe(self, file_path: str | Path) -> str:
        """Transcribe an audio file and return the text."""
        ...


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""


class FasterWhisperTranscriptionProvider:
    """
    Voice transcription provider using local Faster-Whisper model.

    Faster-Whisper uses CTranslate2 for optimized inference.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model_path = model_path or os.environ.get(
            "FASTER_WHISPER_MODEL_PATH",
            ""
        )
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel

            if self.model_path and Path(self.model_path).exists():
                logger.info("Loading Faster-Whisper model from: {}", self.model_path)
                self._model = WhisperModel(
                    self.model_path,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            else:
                # Fall back to downloading from HuggingFace
                logger.info("Loading Faster-Whisper model (will download if needed)")
                model_size = Path(self.model_path).name if self.model_path else "small"
                self._model = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            return self._model
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            return None
        except Exception as e:
            logger.error("Failed to load Faster-Whisper model: {}", e)
            return None

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Faster-Whisper.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        import asyncio

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        model = self._load_model()
        if model is None:
            return ""

        try:
            # Run transcription in thread pool to avoid blocking
            def _transcribe():
                segments, info = model.transcribe(str(path), beam_size=5)
                return " ".join(segment.text for segment in segments)

            text = await asyncio.to_thread(_transcribe)
            logger.debug("Faster-Whisper transcribed {} chars from {}", len(text), path.name)
            return text.strip()

        except Exception as e:
            logger.error("Faster-Whisper transcription error: {}", e)
            return ""


def get_transcription_provider(
    provider: str = "groq",
    groq_api_key: str | None = None,
    faster_whisper_model_path: str | None = None,
    faster_whisper_device: str = "cuda",
    faster_whisper_compute_type: str = "float16",
) -> GroqTranscriptionProvider | FasterWhisperTranscriptionProvider:
    """
    Get a transcription provider instance based on configuration.

    Args:
        provider: Provider name ("groq" or "faster_whisper")
        groq_api_key: Groq API key
        faster_whisper_model_path: Path to Faster-Whisper model
        faster_whisper_device: Device for inference ("cuda" or "cpu")
        faster_whisper_compute_type: Compute type ("float16", "int8", etc.)

    Returns:
        Transcription provider instance.
    """
    if provider == "faster_whisper":
        return FasterWhisperTranscriptionProvider(
            model_path=faster_whisper_model_path,
            device=faster_whisper_device,
            compute_type=faster_whisper_compute_type,
        )
    else:
        return GroqTranscriptionProvider(api_key=groq_api_key)
