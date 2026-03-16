from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

from exordium.utils.decorator import timer_with_return
from exordium.utils.device import get_torch_device

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SAMPLE_RATE = 48000


class ClapWrapper:
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
        device_id: int | None = None,
    ) -> None:
        self.device = get_torch_device(device_id)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.to(self.device)
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

    @timer_with_return
    def __call__(
        self,
        waveforms: torch.Tensor | np.ndarray,
        sample_rate: int = CLAP_SAMPLE_RATE,
    ) -> torch.Tensor:
        """Extract CLAP embeddings from audio waveforms.

        Args:
            waveforms: Input audio of shape ``(B, T)`` or ``(T,)`` at
                ``sample_rate`` Hz.
            sample_rate: Sample rate of the input waveforms. Defaults to
                ``CLAP_SAMPLE_RATE`` (48 kHz).

        Returns:
            Audio embeddings of shape ``(B, 512)``.

        Raises:
            ValueError: If input waveform has an invalid shape.

        """
        waveforms = torch.as_tensor(waveforms, dtype=torch.float32)

        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)

        if waveforms.ndim != 2:
            raise ValueError(
                f"Expected waveform shape (B, T) or (T,), got {tuple(waveforms.shape)}."
            )

        audio_list = [waveforms[i].cpu().numpy() for i in range(waveforms.shape[0])]

        inputs = self.processor(
            audios=audio_list,
            return_tensors="pt",
            sampling_rate=sample_rate,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            embeddings = self.model.get_audio_features(**inputs)

        return embeddings  # (B, 512)

    def read_audio(
        self,
        audio_path: Path | str | torch.Tensor,
        sample_rate: int = CLAP_SAMPLE_RATE,
        resample: bool = True,
    ) -> tuple[torch.Tensor, int]:
        """Load and optionally resample audio to ``CLAP_SAMPLE_RATE``.

        Args:
            audio_path: Path to audio file or pre-loaded 1-D tensor.
            sample_rate: Original sample rate of a pre-loaded tensor.
                Ignored when loading from a file (rate is read from the file).
            resample: If ``True``, resample to ``CLAP_SAMPLE_RATE``.

        Returns:
            Tuple of ``(waveform, effective_sample_rate)`` where waveform is
            a 1-D float32 tensor.

        """
        if isinstance(audio_path, (Path, str)):
            waveform, orig_sr = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)  # stereo → mono
            else:
                waveform = waveform.squeeze(0)
            if resample:
                waveform = self._resample(waveform, orig_sr)
                return waveform, CLAP_SAMPLE_RATE
            return waveform, orig_sr

        # Pre-loaded tensor path
        waveform = audio_path
        if resample and sample_rate != CLAP_SAMPLE_RATE:
            if waveform.ndim == 1:
                waveform = self._resample(waveform, sample_rate)
            else:
                waveform = self._resample(waveform.squeeze(0), sample_rate)
            return waveform, CLAP_SAMPLE_RATE
        return waveform, sample_rate

    def audio_to_feature(
        self,
        audio_path: Path | str,
        resample: bool = True,
    ) -> torch.Tensor:
        """Extract CLAP embeddings from a single audio file.

        Args:
            audio_path: Path to audio file.
            resample: If ``True``, resample to 48 kHz. Defaults to ``True``.

        Returns:
            Embeddings tensor of shape ``(1, 512)``.

        """
        waveform, sr = self.read_audio(audio_path, resample=resample)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        return self(waveform, sample_rate=sr)

    def batch_audio_to_features(
        self,
        audios: list,
        sample_rate: int = CLAP_SAMPLE_RATE,
        resample: bool = True,
    ) -> torch.Tensor:
        """Extract CLAP embeddings from multiple audio inputs in one forward pass.

        Args:
            audios: List of audio file paths or pre-loaded 1-D tensors.
            sample_rate: Original sample rate of pre-loaded tensors.
            resample: If ``True``, resample to 48 kHz. Defaults to ``True``.

        Returns:
            Embeddings tensor of shape ``(B, 512)``.

        """
        waveforms = []
        for audio in audios:
            waveform, _ = self.read_audio(audio, sample_rate=sample_rate, resample=resample)
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)
            waveforms.append(waveform)

        T_max = max(w.shape[0] for w in waveforms)
        padded = torch.zeros(len(waveforms), T_max, dtype=torch.float32)
        for i, w in enumerate(waveforms):
            padded[i, : w.shape[0]] = w

        return self(padded, sample_rate=CLAP_SAMPLE_RATE)
