"""Wav2Vec2 speech encoding model wrapper.

Supports multiple pretrained weight configurations:

- ``"base-960h"`` — Facebook's wav2vec2-base-960h (ASR-finetuned).
- ``"emotion-iemocap"`` — SpeechBrain's emotion-recognition-wav2vec2-IEMOCAP
  (emotion-finetuned on IEMOCAP: neutral, anger, happy, sad).

Both variants produce 768-dimensional frame-level features at ~50 Hz from
16 kHz mono audio.
"""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
import transformers as tfm

from exordium import WEIGHT_DIR
from exordium.audio.base import AudioModelWrapper
from exordium.utils.ckpt import download_file
from exordium.utils.decorator import load_or_create

logger = logging.getLogger(__name__)
"""Module-level logger."""

_MODELS: dict[str, dict[str, str]] = {
    "base-960h": {
        "hf_id": "facebook/wav2vec2-base-960h",
    },
    "emotion-iemocap": {
        "hf_id": "facebook/wav2vec2-base",
        "weight_url": (
            "https://huggingface.co/speechbrain/"
            "emotion-recognition-wav2vec2-IEMOCAP/resolve/main/wav2vec2.ckpt"
        ),
    },
}
"""Supported model configurations."""

SUPPORTED_MODELS = list(_MODELS.keys())
"""List of supported model name strings."""


class Wav2vec2Wrapper(AudioModelWrapper):
    """Wrapper for Wav2Vec2 audio feature extraction.

    Extracts self-supervised learning features from audio using wav2vec2-base
    architecture. Expects 16 kHz mono audio input.

    Args:
        device_id: GPU device index.  ``-1`` or ``None`` uses CPU.
        model_name: Pretrained weight variant.  One of:

            - ``"base-960h"`` (default) — ASR-finetuned features.
            - ``"emotion-iemocap"`` — emotion-finetuned features (IEMOCAP).

    Raises:
        ValueError: If *model_name* is not in :data:`SUPPORTED_MODELS`.

    """

    SAMPLE_RATE = 16000
    """Expected audio sample rate for Wav2Vec2 (16 000 Hz)."""

    def __init__(self, device_id: int = -1, model_name: str = "base-960h") -> None:
        """Initialize Wav2Vec2 wrapper with pretrained model."""
        super().__init__(device_id)

        if model_name not in _MODELS:
            raise ValueError(f"Unknown model_name {model_name!r}. Supported: {SUPPORTED_MODELS}")

        cfg = _MODELS[model_name]
        hf_id = cfg["hf_id"]

        self.preprocessor = tfm.Wav2Vec2Processor.from_pretrained(hf_id)
        self.model = tfm.Wav2Vec2Model.from_pretrained(hf_id)
        assert isinstance(self.model, tfm.Wav2Vec2Model)

        # Load custom weights if the variant has a separate checkpoint
        if "weight_url" in cfg:
            self._load_custom_weights(cfg["weight_url"], model_name)

        self.model.to(self.device)  # ty: ignore[invalid-argument-type]
        self.model.eval()
        logger.info("Wav2Vec2 (%s) loaded to %s.", model_name, self.device)

    def _load_custom_weights(self, url: str, model_name: str) -> None:
        """Download and load a SpeechBrain-style wav2vec2 checkpoint.

        SpeechBrain stores the state dict with a ``model.`` key prefix that
        must be stripped before loading into the HuggingFace
        :class:`~transformers.Wav2Vec2Model`.

        Args:
            url: Remote URL for the ``.ckpt`` file.
            model_name: Used as a subdirectory under the cache.

        """
        weight_dir = WEIGHT_DIR / "wav2vec2" / model_name
        local_path = weight_dir / "wav2vec2.ckpt"
        download_file(url, local_path)

        ckpt = torch.load(str(local_path), map_location="cpu", weights_only=False)
        remapped = {k.replace("model.", "", 1): v for k, v in ckpt.items()}
        self.model.load_state_dict(remapped, strict=False)
        logger.info("Loaded custom weights for %s from %s.", model_name, local_path)

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
        lengths_tensor = cast("torch.LongTensor", torch.tensor(lengths, dtype=torch.long))
        out_lengths = self.model._get_feat_extract_output_lengths(lengths_tensor).tolist()

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
