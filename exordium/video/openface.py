import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from exordium import PathType


def read_au(csv_path: PathType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads OpenFace output csv file and returns the presence and intensities of facial action units.

    Args:
        csv_path (PathType): path to the OpenFace output csv file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: action units of shape (T, 35), timestamps of shape (35,) and success of shape (35,).
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f'Missing file: {csv_path}')

    df = pd.read_csv(csv_path, delimiter=',')
    data = np.array(df.iloc[:, 679:].values.tolist())
    timestamp = np.array(df.iloc[:, 2].astype(float).tolist())
    success = np.array(df.iloc[:, 4].astype(int).tolist())
    return data, timestamp, success


def action_unit_interpolation(data: np.ndarray, success: np.ndarray, thr: int = 50) -> np.ndarray:
    """Interpolates values between successful detections.
    Leading and trailing missdetections do not need linear interpolation, they are filled with zeros (mean values after standardization).

    Args:
        data (np.ndarray): action units of shape (N, M).
        success (np.ndarray): bool confidence of detections (N,).
        thr (int, optional): interpolation is not applied if the successful detections are further in timesteps than a given threshold. Defaults to 50.

    Returns:
        np.ndarray: modified action units.
    """
    # ignore data without detection
    for i, status in enumerate(success):
        if status == 0:
            data[i,:] = np.nan

    # get leading and trailing indices of NaNs
    nan_intervals = []
    start = None
    for i in range(1, len(success)):
        # success followed by failed detection
        if success[i-1] and not success[i]:
            start = i-1

        # failed detection followed by success
        elif not success[i-1] and success[i]:

            # video starts with missdetection, ignore for interpolation
            if start is None:
                continue

            end = i
            nan_intervals.append((start, end, end-start))

    for (start, end, length) in nan_intervals:

        if length > thr:
            continue

        # interpolate between successful detections
        src = np.stack([data[start,:], data[end,:]], axis=0).T # (2, 35) -> (35, 2)
        xi = np.linspace(0, 1, length+1)
        f = interp1d(np.arange(src.shape[1]), src, kind='linear', axis=-1)
        dst = f(xi) # (35, 2) -> (35, L)
        dst = dst.T # (L, 35)
        data[start:start+dst.shape[0],:] = dst

    data[np.isnan(data)] = 0
    return data


def extract_openface_singularity(input_path: PathType,
                                 output_dir: PathType,
                                 singularity_container: PathType,
                                 singularity_args: str = '',
                                 single_person: bool = True) -> None:
    """OpenFace feature extractor with Apptainer/Singularity.

    If this function silently fails to generate the output, then you must add write permission to the output directory.

    Args:
        input_path (PathType): path to the video file.
        output_dir (PathType): path to the output directory.
        singularity_container (PathType): path to the singularity container.
        singularity_args (str): arguments passed to the singularity container.
        single_person (bool, optional): if True, then a single person is expected within the video. Defaults to True.
    """
    if Path(input_path).suffix.lower() in ['.jpeg', '.jpg', '.png']:
        binary = 'FaceLandmarkImg'
    elif not single_person:
        binary = 'FaceLandmarkVidMulti'
    else:
        binary = 'FeatureExtraction'

    output_dir = Path(output_dir).resolve()

    if output_dir.exists():
        return

    CMD = f'singularity exec {singularity_args} {str(singularity_container)} tools/OpenFace/build/bin/{binary} -f {str(input_path)} -nomask -out_dir {str(output_dir)}'
    logging.info(CMD)
    os.system(CMD)


def extract_openface_docker(input_path: PathType, output_dir: PathType, single_person: bool = True) -> None:
    """OpenFace feature extractor with Docker.

    If this function silently fails to generate the output, then you must add write permission to the output directory.

    Args:
        input_path (PathType): path to the video file.
        output_dir (PathType): path to the output directory.
        single_person (bool, optional): if True, then a single person is expected within the video. Defaults to True.
    """
    if Path(input_path).suffix.lower() in ['.jpeg', '.jpg', '.png']:
        binary = 'FaceLandmarkImg'
    elif not single_person:
        binary = 'FaceLandmarkVidMulti'
    else:
        binary = 'FeatureExtraction'

    parent_dir = Path(input_path).resolve().parent
    output_dir = Path(output_dir).resolve()

    if output_dir.exists():
        return

    CMD = f'docker run --entrypoint /home/openface-build/build/bin/{binary} -it -v {str(parent_dir)}:/input_dir -v {str(output_dir)}:/output_dir ' \
          f'--rm algebr/openface:latest -f /{str(Path("/input_dir") / Path(input_path).name)} -out_dir /output_dir'
    logging.info(CMD)
    os.system(CMD)