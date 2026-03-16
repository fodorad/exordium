"""OpenSmile feature extractor wrapper."""

from pathlib import Path

import numpy as np
import opensmile
import torch

from exordium.utils.decorator import load_or_create


class OpensmileWrapper:
    """Extract acoustic features using OpenSmile.

    Wrapper for OpenSmile feature extraction with support for multiple
    predefined feature sets and levels.
    """

    def __init__(self, feature_set: str = "egemaps", feature_level: str = "lld"):
        """OpenSmile wrapper class.

        Args:
            feature_set: Predefined feature set. Supported values are 'egemaps' and 'compare'.
            feature_level: Predefined feature level. Supported values are 'lld' and 'functionals'.

        Raises:
            ValueError: If feature_set or feature_level is invalid.

        """
        if feature_set not in {"egemaps", "compare"}:
            raise ValueError(
                f"Unsupported feature_set: {feature_set}. "
                "Supported values are: 'egemaps', 'compare'."
            )

        if feature_level not in {"lld", "functionals"}:
            raise ValueError(
                f"Unsupported feature_level: {feature_level}. "
                "Supported values are: 'lld', 'functionals'."
            )

        feature_set = (
            opensmile.FeatureSet.eGeMAPSv02
            if feature_set == "egemaps"
            else opensmile.FeatureSet.ComParE_2016
        )
        feature_level = (
            opensmile.FeatureLevel.LowLevelDescriptors
            if feature_level == "lld"
            else opensmile.FeatureLevel.Functionals
        )

        self.smile = opensmile.Smile(feature_set=feature_set, feature_level=feature_level)

    @load_or_create("npy")
    def audio_to_feature(self, audio_path: Path | str, **_kwargs) -> np.ndarray:
        """Extract OpenSmile features from audio file.

        Args:
            audio_path: Path to audio file.
            **kwargs: Additional arguments (unused, for decorator compatibility).

        Returns:
            Feature array from OpenSmile processing.

        """
        feature = self(audio_path)
        return feature

    def __call__(
        self,
        waveform: Path | str | np.ndarray | torch.Tensor,
        sr: int = 16000,
        return_tensors: str = "npy",
    ) -> np.ndarray | torch.Tensor:
        """Extract OpenSmile features from waveform or audio file.

        Args:
            waveform: Path to audio file or audio signal array.
            sr: Sample rate in Hz (used if waveform is array).
            return_tensors: Return format - "npy" for numpy array, "pt" for torch tensor.

        Returns:
            Feature array as numpy array or torch tensor based on return_tensors.

        """
        if isinstance(waveform, (np.ndarray, torch.Tensor)):
            feature = np.array(self.smile.process_signal(np.array(waveform), sr))
        else:
            feature = np.array(self.smile.process_file(waveform))

        if return_tensors == "pt":
            feature = torch.Tensor(feature)

        return feature
