import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from exordium import PathType


def read_openface_frame_au(csv_path: PathType, confidence_thr: float = 0.9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads OpenFace output csv file and returns the presence and intensities of facial action units.

    Args:
        csv_path (PathType): path to the OpenFace output csv file.

    Returns:
        tuple[np.ndarray, list[str], np.ndarray, np.ndarray]: action units of shape (T, 35), list of action unit names, timestamps of shape (35,) and success of shape (35,).
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f'Missing file: {csv_path}')

    df = pd.read_csv(csv_path, delimiter=',')
    header = df.columns.tolist()
    au_names = np.array(header[header.index('AU01_r'):])
    au_values = np.array(df.iloc[:, header.index('AU01_r'):].values.tolist())

    confidence = np.array(df.iloc[:, header.index('confidence')].astype(float).tolist())
    success = confidence > confidence_thr
    au_values = au_values[success,:]

    if au_values.shape[0] == 0:
        print('No valid data... skip')
        return None, None
    elif au_values.shape[0] > 1:
        print('More than one face detected. Select the biggest bb...')
        x = np.array(df.iloc[:, header.index('x_0'):header.index('x_67')].values.tolist())[success,:]
        y = np.array(df.iloc[:, header.index('y_0'):header.index('y_67')].values.tolist())[success,:]

        max_ind = 0
        max_value = 0
        for i in range(x.shape[0]):
            x_i = x[i,:]
            y_i = y[i,:]
            bb_x_min, bb_x_max = x_i.min(), x_i.max()
            bb_y_min, bb_y_max = y_i.min(), y_i.max()
            bb_w = bb_x_max - bb_x_min
            bb_h = bb_y_max - bb_y_min
            max_i = max(bb_w, bb_h)
            if max_value < max_i:
                max_value = max_i
                max_ind = i

        au_values = au_values[max_ind,:]
    else:
        au_values = au_values[0,:]

    if au_values.shape != (35,):
        raise ValueError(f'Expected shape is (35,) got instead {au_values.shape}')

    return au_values, au_names


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
        source = '-f'
    elif (Path(input_path).is_dir() and any([elem.suffix in ['.jpeg', '.jpg', '.png'] for elem in Path(input_path).glob('*')])):
        binary = 'FaceLandmarkImg'
        source = '-fdir'
    elif not single_person:
        binary = 'FaceLandmarkVidMulti'
        source = '-f'
    else:
        binary = 'FeatureExtraction'
        source = '-f'

    output_dir = Path(output_dir).resolve()

    if output_dir.exists():
        return

    CMD = f'singularity exec {singularity_args} {str(singularity_container)} tools/OpenFace/build/bin/{binary} {source} {str(input_path)} -nomask -out_dir {str(output_dir)}'
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