"""WavLM speech encoding model wrapper."""

from pathlib import Path

import numpy as np
import torch
from transformers import WavLMModel

from exordium.audio.base import AudioModelWrapper
from exordium.utils.decorator import load_or_create

WAVLM_SAMPLE_RATE = 16000  # all WavLM variants operate at 16 kHz

_MODEL_IDS: dict[str, str] = {
    "base": "microsoft/wavlm-base",
    "base+": "microsoft/wavlm-base-plus",
    "large": "microsoft/wavlm-large",
}


class WavlmWrapper(AudioModelWrapper):
    """Wrapper for WavLM audio feature extraction via HuggingFace Transformers.

    Extracts layer-wise hidden states using Microsoft's WavLM model.
    Supports base, base+, and large model variants.

    Args:
        device_id: GPU device ID. Use -1 for CPU.
        model_name: Model variant — ``"base"``, ``"base+"``, or ``"large"``.
            Defaults to ``"base+"``.

    Raises:
        ValueError: If ``model_name`` is not one of the supported variants.

    """

    def __init__(self, device_id: int = -1, model_name: str = "base+") -> None:
        super().__init__(device_id)

        if model_name not in _MODEL_IDS:
            raise ValueError(f"Invalid model_name: {model_name!r}. Choose from {list(_MODEL_IDS)}.")

        self.sample_rate = WAVLM_SAMPLE_RATE
        self.model = WavLMModel.from_pretrained(_MODEL_IDS[model_name])
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, waveform: np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        """Extract hidden-state features from a waveform.

        Args:
            waveform: Audio signal of shape ``(T,)`` or ``(B, T)``
                      at ``WAVLM_SAMPLE_RATE`` Hz.

        Returns:
            List of hidden-state tensors, one per transformer layer,
            each of shape ``(B, T', hidden_size)``.

        Raises:
            ValueError: If waveform has an invalid shape (not 1D or 2D).

        """
        waveform = torch.as_tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 2:
            raise ValueError(f"Expected shape (B, T) or (T,) but got {tuple(waveform.shape)}.")

        waveform = waveform.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_values=waveform, output_hidden_states=True)

        # hidden_states[0] is the CNN feature extractor output;
        # [1:] are the transformer layers (12 for base/base+, 24 for large).
        return list(outputs.hidden_states[1:])

    @load_or_create("pkl")
    def audio_to_feature(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        **_kwargs,
    ) -> list[np.ndarray]:
        """Extract WavLM features from a single audio path or waveform.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
                   If a path is given, audio is loaded and resampled to 16 kHz.
            **kwargs: Passed to :func:`~exordium.utils.decorator.load_or_create`
                      (``output_path``, ``overwrite``).

        Returns:
            List of layer features as numpy arrays,
            each of shape ``(T, hidden_size)``.

        """
        features = self(self._prepare_waveform(audio, self.sample_rate))
        return [f.detach().cpu().numpy().squeeze(0) for f in features]

    def _feat_extract_output_lengths(self, lengths: list[int]) -> list[int]:
        """Compute output frame counts for given waveform sample lengths.

        Applies the stride/kernel formula for each CNN layer in the feature
        extractor, matching what the model computes internally.

        Args:
            lengths: Waveform lengths in samples.

        Returns:
            Output frame counts, one per input length.

        """
        out = lengths
        for layer in self.model.feature_extractor.conv_layers:
            k = layer.conv.kernel_size[0]
            s = layer.conv.stride[0]
            out = [(n - k) // s + 1 for n in out]
        return out

    def batch_audio_to_features(
        self,
        audios: list[Path | str | np.ndarray | torch.Tensor],
        **_kwargs,
    ) -> list[list[np.ndarray]]:
        """Extract WavLM features from multiple audio inputs in one forward pass.

        Variable-length inputs are zero-padded to the longest waveform.
        Output frames that correspond to padding are trimmed.

        Args:
            audios: List of audio file paths, numpy arrays, or torch tensors.

        Returns:
            List of per-file features. Each element is a list of layer numpy
            arrays, each of shape ``(T_i, hidden_size)``.

        """
        waveforms = [self._prepare_waveform(a, self.sample_rate) for a in audios]
        padded, lengths = self._pad_waveforms(waveforms)
        padded = padded.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_values=padded, output_hidden_states=True)

        features = list(outputs.hidden_states[1:])  # list[Tensor(B, T_max, hidden_size)]
        out_lengths = self._feat_extract_output_lengths(lengths)
        return [
            [layer[i, : out_lengths[i]].detach().cpu().numpy() for layer in features]
            for i in range(len(waveforms))
        ]

    @torch.inference_mode()
    def inference(self, waveform: np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        """Extract features in inference mode.

        Args:
            waveform: Audio signal of shape ``(T,)`` or ``(B, T)``.

        Returns:
            List of hidden-state tensors, each of shape ``(B, T', hidden_size)``.

        Raises:
            ValueError: If waveform has an invalid shape.

        """
        waveform = torch.as_tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 2:
            raise ValueError(f"Expected shape (B, T) or (T,) but got {tuple(waveform.shape)}.")

        waveform = waveform.to(self.device)
        outputs = self.model(input_values=waveform, output_hidden_states=True)
        return list(outputs.hidden_states[1:])
