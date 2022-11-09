from pathlib import Path

try:
    from pyAudioAnalysis import MidTermFeatures
except:
    raise ImportError('pyAudioAnalysis cannot be found. Follow the README to install the package.')


def audio2pyaa(input_path: str, output_path: str) -> None:
    """Extract short and mid term features with pyAudioAnalysis, then save

    Args:
        input_path (str): audio path
        output_path (str): pyaa feature path
    """
    output_path = Path(output_path).resolve()

    if (output_path.parent / (output_path.name + '_mt.csv')).exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    MidTermFeatures.mid_feature_extraction_to_file(
        file_path=input_path,
        mid_window=1.0, 
        mid_step=1.0, 
        short_window=0.05, 
        short_step=0.05, 
        output_file=str(output_path),
        store_short_features=True, 
        store_csv=False, 
        plot=False)
