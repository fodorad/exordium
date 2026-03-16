from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel
from faster_whisper.utils import available_models

from exordium.audio.io import load_audio
from exordium.utils.decorator import timer_with_return


class WhisperWrapper:
    """Wrapper for faster-whisper speech-to-text.

    Uses CTranslate2-optimized Whisper models for fast inference.
    Supports all standard and distilled model variants.

    Args:
        device_id: Device index. Use -1 for CPU, 0+ for GPU.
            On macOS (no CUDA), always uses CPU regardless of value.
        model_name: Whisper model variant. See ``WhisperWrapper.available_models()``
            for the full list. Recommended: ``"distil-large-v3"`` for quality,
            ``"turbo"`` for speed, ``"tiny"`` for minimal footprint.

    """

    def __init__(self, device_id: int = 0, model_name: str = "turbo") -> None:
        if model_name not in available_models():
            raise ValueError(
                f"Unknown model: {model_name}. Available: {', '.join(available_models())}"
            )

        device, device_index = self._resolve_device(device_id)

        self.model_name = model_name
        self.model = WhisperModel(
            model_name,
            device=device,
            device_index=device_index,
            compute_type="int8",
        )

    @staticmethod
    def _resolve_device(device_id: int) -> tuple[str, int]:
        """Map exordium device_id to faster-whisper device/device_index.

        Note: mps is not supported by faster-whisper, so we always return "cpu" on macOS.

        """
        if device_id < 0:
            return "cpu", 0

        if torch.cuda.is_available():
            return "cuda", device_id

        return "cpu", 0

    @staticmethod
    def available_models() -> list[str]:
        """Return list of supported model names."""
        return available_models()

    @timer_with_return
    def __call__(
        self,
        waveform: np.ndarray | torch.Tensor,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        word_timestamps: bool = True,
    ) -> str:
        """Transcribe audio waveform to text.

        Args:
            waveform: 1D audio signal at 16 kHz, mono.
            language: Language code (e.g. ``"en"``). ``None`` for auto-detection.
            beam_size: Beam size for decoding.
            vad_filter: If True, apply Silero VAD to filter non-speech.
            word_timestamps: If True, include word-level timestamps in output.

        Returns:
            Transcribed text as a string.

        """
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()

        waveform = waveform.squeeze().astype(np.float32)

        segments, info = self.model.transcribe(
            waveform,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    def transcribe_file(
        self,
        audio_path: Path | str,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
    ) -> str:
        """Load an audio file and transcribe it.

        Uses ``exordium.audio.io.load_audio`` for loading and resampling
        to 16 kHz mono.

        Args:
            audio_path: Path to audio file.
            language: Language code. ``None`` for auto-detection.
            beam_size: Beam size for decoding.
            vad_filter: If True, apply Silero VAD to filter non-speech.

        Returns:
            Transcribed text.

        """
        waveform, _ = load_audio(audio_path, target_sample_rate=16000, mono=True, squeeze=True)
        return self(waveform, language=language, beam_size=beam_size, vad_filter=vad_filter)
