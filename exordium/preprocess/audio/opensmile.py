import opensmile
import numpy as np


def load_opensmile(feature_set: str = 'egemaps', feature_level: str = 'lld'):
    assert feature_set in ['egemaps', 'compare']
    assert feature_level in ['lld', 'functionals']
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02 if feature_set == 'egemaps' else opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors if feature_level == 'lld' else opensmile.FeatureLevel.Functionals,
    )
    return smile


def audio2opensmile(input_path: str, smile: opensmile.Smile) -> np.ndarray:
    return np.array(smile.process_file(input_path))
