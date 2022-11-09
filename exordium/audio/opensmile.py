import opensmile
import numpy as np


class OpensmileWrapper():

    def __init__(self, feature_set: str = 'egemaps', feature_level: str = 'lld') -> None:
        assert feature_set in ['egemaps', 'compare']
        assert feature_level in ['lld', 'functionals']
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02 if feature_set == 'egemaps' else opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors if feature_level == 'lld' else opensmile.FeatureLevel.Functionals,
        )


    def extract_features(self, input_path: str) -> np.ndarray:
        return np.array(self.smile.process_file(input_path))
