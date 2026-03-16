from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from exordium.audio.io import load_audio
from exordium.utils.device import get_torch_device


class AudioModelWrapper(ABC):
    """Abstract base class for audio feature extraction models.

    Provides shared device setup, waveform preparation, and padding utilities.
    Subclasses must implement :meth:`audio_to_feature`, :meth:`batch_audio_to_features`,
    and :meth:`inference`.

    The key shared behaviour is :meth:`_prepare_waveform`, which normalises
    any input — file path, numpy array, or torch tensor — to a mono 1D
    waveform tensor at the requested sample rate.
    """

    def __init__(self, device_id: int = -1) -> None:
        """Initialize with device setup.

        Args:
            device_id: GPU device ID. Use -1 for CPU.

        """
        self.device = get_torch_device(device_id)

    @abstractmethod
    def audio_to_feature(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        **kwargs,
    ):
        """Extract features from a single audio path or waveform.

        Subclasses decorate this with
        :func:`~exordium.utils.decorator.load_or_create` using the appropriate
        file format (``"npy"`` or ``"pkl"``).

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            **kwargs: Forwarded to the caching decorator
                      (``output_path``, ``overwrite``).

        """

    @abstractmethod
    def batch_audio_to_features(
        self,
        audios: list[Path | str | np.ndarray | torch.Tensor],
        **kwargs,
    ):
        """Extract features from multiple audio inputs in a single forward pass.

        Args:
            audios: List of audio file paths, numpy arrays, or torch tensors.
            **kwargs: Additional arguments.

        """

    @abstractmethod
    def inference(self, waveform: np.ndarray | torch.Tensor):
        """Extract features in inference mode, returning torch tensors.

        Args:
            waveform: Audio waveform as numpy array or torch tensor.

        """

    def _prepare_waveform(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """Convert any audio input to a mono 1D waveform tensor.

        Accepts a file path (loads and resamples via
        :func:`~exordium.audio.io.load_audio`) or an already-loaded waveform
        (numpy array or torch tensor). If the waveform is 2D ``(C, T)`` the
        first channel is used.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Target sample rate in Hz (only used when loading
                         from a file).

        Returns:
            Mono waveform tensor of shape ``(T,)``.

        Raises:
            ValueError: If input has more than 2 dimensions.

        """
        if isinstance(audio, (Path, str)):
            waveform, _ = load_audio(audio, target_sample_rate=sample_rate, mono=True, squeeze=True)
            return waveform

        waveform = torch.as_tensor(audio, dtype=torch.float32)

        if waveform.ndim == 2:
            waveform = waveform[0, :]
        elif waveform.ndim != 1:
            raise ValueError(f"Expected shape (T,) or (C, T), got {tuple(waveform.shape)}.")

        return waveform

    @staticmethod
    def _pad_waveforms(waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        """Pad a list of 1D waveforms to the same length with zeros.

        Args:
            waveforms: List of 1D waveform tensors, possibly different lengths.

        Returns:
            Tuple of ``(padded_batch, original_lengths)``:

            - ``padded_batch``: Float tensor of shape ``(B, T_max)``.
            - ``original_lengths``: List of original waveform lengths.

        """
        lengths = [w.shape[0] for w in waveforms]
        T_max = max(lengths)
        padded = torch.zeros(len(waveforms), T_max, dtype=torch.float32)
        for i, w in enumerate(waveforms):
            padded[i, : w.shape[0]] = w
        return padded, lengths
