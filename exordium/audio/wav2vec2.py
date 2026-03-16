from pathlib import Path

import numpy as np
import torch
import transformers as tfm

from exordium.audio.base import AudioModelWrapper
from exordium.utils.decorator import load_or_create


class Wav2vec2Wrapper(AudioModelWrapper):
    """Wrapper for Wav2Vec2 audio feature extraction.

    Extracts self-supervised learning features from audio using Facebook's
    wav2vec2-base-960h model. Expects 16kHz mono audio input.
    """

    SAMPLE_RATE = 16000

    def __init__(self, device_id: int = -1) -> None:
        """Initialize Wav2Vec2 wrapper with pretrained model."""
        super().__init__(device_id)
        self.preprocessor = tfm.Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = tfm.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        waveform: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Extract audio features from a waveform.

        Args:
            waveform: Mono audio signal of shape ``(T,)`` (single) or
                      ``(B, T)`` (batch of same-length signals).

        Returns:
            Feature tensor of shape ``(T, 768)`` for 1D input, or
            ``(B, T, 768)`` for 2D input.

        Raises:
            ValueError: If waveform is not 1D or 2D.

        """
        waveform = torch.as_tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 1:
            input_values = self.preprocessor(
                waveform, return_tensors="pt", sampling_rate=self.SAMPLE_RATE
            ).input_values.to(self.device)
            with torch.inference_mode():
                feature = self.model(input_values).last_hidden_state
            return feature.squeeze(0)  # (T, 768)

        if waveform.ndim == 2:
            # Batch of same-length signals — pass each row to the processor as a list
            waveforms_np = [waveform[i].numpy() for i in range(waveform.shape[0])]
            input_values = self.preprocessor(
                waveforms_np,
                return_tensors="pt",
                padding=False,
                sampling_rate=self.SAMPLE_RATE,
            ).input_values.to(self.device)
            with torch.inference_mode():
                features = self.model(input_values).last_hidden_state
            return features  # (B, T, 768)

        raise ValueError(f"Expected shape (T,) or (B, T), got {tuple(waveform.shape)}.")

    @load_or_create("npy")
    def audio_to_feature(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        **_kwargs,
    ) -> np.ndarray:
        """Extract Wav2Vec2 features from an audio path or waveform.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
                   If a path is given, audio is loaded and resampled to 16kHz.
            **kwargs: Passed to :func:`~exordium.utils.decorator.load_or_create`
                      (``output_path``, ``overwrite``).

        Returns:
            Feature array of shape ``(T, 768)`` where T is time frames.

        """
        return self(self._prepare_waveform(audio, self.SAMPLE_RATE)).detach().cpu().numpy()

    def batch_audio_to_features(
        self,
        audios: list[Path | str | np.ndarray | torch.Tensor],
        **_kwargs,
    ) -> list[np.ndarray]:
        """Extract Wav2Vec2 features from multiple audio inputs in one forward pass.

        Supports variable-length inputs: each waveform is padded to the maximum
        length in the batch. Output frames corresponding to padding are
        discarded using the model's own feature-length calculation.

        Args:
            audios: List of audio file paths, numpy arrays, or torch tensors.

        Returns:
            List of feature arrays, each of shape ``(T_i, 768)``.

        """
        waveforms = [self._prepare_waveform(a, self.SAMPLE_RATE) for a in audios]
        lengths = [w.shape[0] for w in waveforms]

        # Processor handles padding and normalization for variable-length batches
        waveforms_np = [w.numpy() for w in waveforms]
        input_values = self.preprocessor(
            waveforms_np,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.SAMPLE_RATE,
        ).input_values.to(self.device)

        # Compute output lengths to trim padding from each sequence
        out_lengths = self.model._get_feat_extract_output_lengths(
            torch.tensor(lengths, dtype=torch.long)
        ).tolist()

        with torch.inference_mode():
            hidden = self.model(input_values).last_hidden_state  # (B, T_max, 768)

        return [
            hidden[i, : int(out_len)].detach().cpu().numpy()
            for i, out_len in enumerate(out_lengths)
        ]

    @torch.inference_mode()
    def inference(self, waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract features in inference mode, returning a tensor.

        Args:
            waveform: Mono audio signal of shape ``(T,)``.

        Returns:
            Feature tensor of shape ``(T, 768)``.

        """
        input_values = self.preprocessor(
            torch.as_tensor(waveform, dtype=torch.float32),
            return_tensors="pt",
            sampling_rate=self.SAMPLE_RATE,
        ).input_values.to(self.device)
        return self.model(input_values).last_hidden_state.squeeze(0)  # (T, 768)
