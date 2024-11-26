import opensmile
import numpy as np
import torch
from exordium import PathType
from exordium.utils.decorator import load_or_create


class OpensmileWrapper():

    def __init__(self, feature_set: str = 'egemaps', feature_level: str = 'lld'):
        """OpenSmile wrapper class.

        Args:
            feature_set (str, optional): predefined feature set. Supported values are 'egemaps' and 'compare'. Defaults to 'egemaps'.
            feature_level (str, optional): predefined feature level. Supported values are 'lld' and 'functionals'. Defaults to 'lld'.

        Raises:
            ValueError:
                - given feature_set is invalid.
                - given feature_level is invalid.
        """
        if feature_set not in {'egemaps', 'compare'}:
            raise ValueError(f'Given feature_set ({feature_set}) is not supported.')

        if feature_level not in {'lld', 'functionals'}:
            raise ValueError(f'Given feature_level ({feature_level}) is not supported.')

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02 if feature_set == 'egemaps' else opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors if feature_level == 'lld' else opensmile.FeatureLevel.Functionals,
        )

    @load_or_create('npy')
    def audio_to_feature(self, audio_path: PathType, **kwargs):
        feature = self(audio_path)
        return feature

    def __call__(self, waveform: PathType | np.ndarray | torch.Tensor, sr: int = 16000, return_tensors: str = "npy") -> np.ndarray | torch.Tensor:
        if isinstance(waveform, np.ndarray | torch.Tensor):
            feature = np.array(self.smile.process_signal(np.array(waveform), sr))
        else: # PathType
            feature = np.array(self.smile.process_file(waveform))

        if return_tensors == "pt":
            feature = torch.Tensor(feature)

        return feature