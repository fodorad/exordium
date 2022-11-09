import os
import csv
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d


def read_au(csv_file: str) -> tuple[np.ndarray, list, list]:
    """Read OpenFace output, and select only facial action units: presence and intensities

    Args:
        csv_file (str): path to OpenFace csv file

    Returns:
        np.ndarray: tensor with shape (T,35)
    """
    assert Path(csv_file).exists(), f'Missing file: {csv_file}'

    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')

        lines = []
        frame = []
        timestamp = []
        success = []
        for line_count, row in enumerate(csv_reader):
            if line_count == 0: continue
            lines.append([float(item) for item in row[679:]])
            frame.append(int(row[0]))
            timestamp.append(float(row[2]))
            success.append(int(row[4]))

    data = np.array(lines)
    frame = np.array(frame)-1 # index starts from 0
    if frame.shape[0] != frame[-1] + 1:
        print(f'OpenFace has missing frames in the output file: {csv_file}. Fixed.')
        out = np.zeros(shape=(frame[-1]+1, data.shape[1]))
        s = np.zeros(shape=out.shape[0], dtype=int)
        t = -1*np.ones(shape=out.shape[0], dtype=float)

        for ni, di in zip(list(frame), range(data.shape[0])):
           out[ni] = data[di,:]
           s[ni] = success[di]
           t[ni] = timestamp[di]

        data, success, timestamp = out, list(s), list(t)

    return data, success, timestamp


def au_interpolation(data: np.ndarray, success: list, len_sec: int = 2):
    # ignore data without detection
    for i, status in enumerate(success):
        if status == 0: data[i,:] = np.nan

    # get leading and trailing indices of NaNs
    nan_intervals = []
    start = None
    for i in range(1, len(success)):
        if success[i-1] and not success[i]: # success followed by failed detection
            start = i-1
        elif not success[i-1] and success[i]: # failed detection followed by success
            if start is None: continue # video starts with missdetection, skip for interpolation
            end = i
            nan_intervals.append((start, end, end-start))

    # interpolate between successful detections
    for (start, end, length) in nan_intervals:
        if length > 50: continue # do not interpolate if missing detections are longer than 2 sec (25 fps)
        src = np.stack([data[start,:], data[end,:]], axis=0).T # (2, 35) -> (35, 2)
        xi = np.linspace(0, 1, length+1)
        f = interp1d(np.arange(src.shape[1]), src, kind='linear', axis=-1)
        dst = f(xi) # (35, 2) -> (35, L)
        dst = dst.T # (L, 35)
        data[start:start+dst.shape[0],:] = dst

    # leading and trailing missdetections do not need linear interpolation - filled with zeros (mean values)
    data[np.isnan(data)] = 0
    return data


def extract_openface_singularity(input_path: str,
                                 output_dir: str,
                                 single_person: bool = True,
                                 singularity_container: str = 'tools/OpenFace/openface_latest.sif',
                                 singularity_args: str = '') -> None:
    """OpenFace feature extractor

    Args:
        input_path (str): path to a video clip (.mp4)
        output_dir (str): path to an output directory.
                          a new directory is created with the name of the video
        single_person (bool, optional): Number of partipants in the video. Defaults to True.
    """
    # if this silently fails to generate the output, add write permission to the output directory
    if Path(input_path).suffix.lower() in ['.jpeg', '.jpg', '.png']:
        binary = 'FaceLandmarkImg'
    elif not single_person:
        binary = 'FaceLandmarkVidMulti'
    else:
        binary = 'FeatureExtraction'
    output_dir = Path(output_dir).resolve()
    if output_dir.exists(): return
    CMD = f'singularity exec {singularity_args} {singularity_container} tools/OpenFace/build/bin/{binary} -f {input_path} -nomask -out_dir {output_dir}'
    print(CMD)
    os.system(CMD)


def extract_openface_docker(input_path: str,
                            output_dir: str,
                            single_person: bool = True) -> None:
    """OpenFace feature extractor with Docker

    Args:
        input_path (str): path to a video clip (.mp4)
        output_dir (str): path to an output directory.
                          a new directory is created with the name of the video
        single_person (bool, optional): Number of partipants in the video. Defaults to True.
    """
    # if this silently fails to generate the output, add write permission to the output directory
    if Path(input_path).suffix.lower() in ['.jpeg', '.jpg', '.png']:
        binary = 'FaceLandmarkImg'
    elif not single_person:
        binary = 'FaceLandmarkVidMulti'
    else:
        binary = 'FeatureExtraction'
    parent_dir = Path(input_path).resolve().parent
    output_dir = Path(output_dir).resolve()
    #if output_dir.exists(): return
    CMD = f'docker run --entrypoint /home/openface-build/build/bin/{binary} -it -v {str(parent_dir)}:/input_dir -v {str(output_dir)}:/output_dir ' \
          f'--rm algebr/openface:latest -f /{str(Path("/input_dir") / Path(input_path).name)} -out_dir /output_dir'
    print(CMD)
    os.system(CMD)
