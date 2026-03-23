"""CLAP audio embedding model wrapper."""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

from exordium.audio.base import AudioModelWrapper

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
"""HuggingFace model ID for the LAION CLAP music-and-speech checkpoint."""
CLAP_SAMPLE_RATE = 48000
"""Required audio sample rate for CLAP inference (48 000 Hz)."""


class ClapWrapper(AudioModelWrapper):
    """Wrapper for LAION CLAP audio embedding model via HuggingFace Transformers.

    Extracts 512-dimensional audio embeddings using the
    ``laion/larger_clap_music_and_speech`` checkpoint.  Input audio at any
    sample rate is resampled to 48 kHz before feature extraction.

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``"laion/larger_clap_music_and_speech"``.
        device_id: GPU device index. ``None`` or negative uses CPU.

    """

    def __init__(
        self,
        model_name: str = CLAP_MODEL_ID,
        device_id: int = -1,
    ) -> None:
        super().__init__(device_id)
        self.model = ClapModel.from_pretrained(model_name)
        assert isinstance(self.model, ClapModel)
        self.model.to(self.device)  # ty: ignore[invalid-argument-type]
        self.model.eval()
        self.processor = ClapProcessor.from_pretrained(model_name)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample a 1-D waveform to ``CLAP_SAMPLE_RATE`` (48 kHz)."""
        if orig_sr == CLAP_SAMPLE_RATE:
            return waveform
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, CLAP_SAMPLE_RATE)
        return self._resamplers[orig_sr](waveform)

    def __call__(
        self,
        waveforms: torch.Tensor,
        sample_rate: int = CLAP_SAMPLE_RATE,
    ) -> torch.Tensor:
        """Extract CLAP embeddings from audio waveforms.

        Args:
            waveforms: Input audio of shape ``(B, T)`` or ``(T,)`` as a
                float32 tensor at ``sample_rate`` Hz.
            sample_rate: Sample rate of the input waveforms. Defaults to
                ``CLAP_SAMPLE_RATE`` (48 kHz).

        Returns:
            Audio embeddings of shape ``(B, 512)``.

        Raises:
            ValueError: If input waveform has an invalid shape.

        """
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)

        if waveforms.ndim != 2:
            raise ValueError(
                f"Expected waveform shape (B, T) or (T,), got {tuple(waveforms.shape)}."
            )

        audio_list = [waveforms[i].cpu().numpy() for i in range(waveforms.shape[0])]

        inputs = self.processor(
            audio=audio_list,
            return_tensors="pt",
            sampling_rate=sample_rate,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            return self.model.get_audio_features(**inputs).pooler_output  # (B, 512)

    def inference(self, waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract CLAP features from a pre-prepared waveform tensor.

        Args:
            waveform: Float tensor of shape ``(B, T)`` or ``(T,)`` at
                ``CLAP_SAMPLE_RATE`` (48 kHz).

        Returns:
            Audio embeddings of shape ``(B, 512)``.

        """
        t = waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
        return self(t, sample_rate=CLAP_SAMPLE_RATE)

    def audio_to_feature(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Extract CLAP embeddings from a single audio file or waveform.

        File paths are loaded and resampled to 48 kHz automatically via
        :meth:`~exordium.audio.base.AudioModelWrapper._prepare_waveform`.
        Pre-loaded tensors are assumed to be at ``sample_rate``; set
        ``sample_rate`` accordingly so resampling is applied when needed.

        Args:
            audio: Audio file path or pre-loaded 1-D float tensor.
            sample_rate: Sample rate of a pre-loaded tensor (ignored for
                file paths, which are loaded at their native rate and
                resampled). Defaults to ``CLAP_SAMPLE_RATE``.
            **kwargs: Ignored; reserved for ``load_or_create`` compatibility.

        Returns:
            Embeddings tensor of shape ``(1, 512)``.

        """
        sample_rate = kwargs.pop("sample_rate", CLAP_SAMPLE_RATE)
        if isinstance(audio, (Path, str)):
            # _prepare_waveform loads at native rate; resample separately
            from exordium.audio.io import load_audio

            waveform, orig_sr = load_audio(audio, target_sample_rate=None, mono=True, squeeze=True)
            waveform = self._resample(waveform, orig_sr)
        else:
            waveform = self._prepare_waveform(audio, sample_rate)
            waveform = self._resample(waveform, sample_rate)
        return self(waveform, sample_rate=CLAP_SAMPLE_RATE)

    def batch_audio_to_features(
        self,
        audios: list,
        **kwargs,
    ) -> torch.Tensor:
        """Extract CLAP embeddings from multiple audio inputs in one forward pass.

        Args:
            audios: List of audio file paths or pre-loaded 1-D tensors.
            sample_rate: Sample rate of pre-loaded tensors. Ignored for file
                paths. Defaults to ``CLAP_SAMPLE_RATE``.
            **kwargs: Ignored; reserved for ``load_or_create`` compatibility.

        Returns:
            Embeddings tensor of shape ``(B, 512)``.

        """
        from exordium.audio.io import load_audio

        sample_rate = kwargs.pop("sample_rate", CLAP_SAMPLE_RATE)

        waveforms = []
        for audio in audios:
            if isinstance(audio, (Path, str)):
                w, orig_sr = load_audio(audio, target_sample_rate=None, mono=True, squeeze=True)
                w = self._resample(w, orig_sr)
            else:
                w = self._prepare_waveform(audio, sample_rate)
                w = self._resample(w, sample_rate)
            waveforms.append(w)

        padded, _ = self._pad_waveforms(waveforms)
        return self(padded, sample_rate=CLAP_SAMPLE_RATE)
