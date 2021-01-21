import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from Katna.video import Video


def video2openface(input_path: str, output_dir: str, single_person: bool = True):
    if Path(input_path).suffix.lower() in ['.jpeg', '.jpg', '.png']:
        binary = 'FaceLandmarkImg'
    elif not single_person:
        binary = 'FaceLandmarkVidMulti'
    else:
        binary = 'FeatureExtraction'
    parent_dir = Path(input_path).resolve().parent
    output_dir = Path(output_dir).resolve()
    if output_dir.exists(): return
    CMD = f'docker run --entrypoint /home/openface-build/build/bin/{binary} -it -v {str(parent_dir)}:/input_dir -v {str(output_dir)}:/output_dir ' \
          f'--rm algebr/openface:latest -f /{str(Path("/input_dir") / Path(input_path).name)} -out_dir /output_dir'
    print(CMD)
    os.system(CMD)


def frames2dynimgs(input_dir, output_dir):
    WINDOW_LENGTH = 30
    STRIDE = 30
    input_paths = sorted(Path(input_dir).glob('*.jpg'))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(0, len(input_paths) - WINDOW_LENGTH, STRIDE):
        forward_path = Path(output_dir) / 'dynimg_f_{0:05d}.png'.format(i)
        backward_path = Path(output_dir) / 'dynimg_b_{0:05d}.png'.format(i)
        if forward_path.exists(): continue
        frames_forward = np.array([cv2.imread(str(x)) for x in input_paths])
        chunk_forward = frames_forward[i:i + WINDOW_LENGTH]
        dynamic_image_forward = get_dynamic_image(chunk_forward)
        cv2.imwrite(str(forward_path), dynamic_image_forward)
        if backward_path.exists(): continue
        frames_backward = frames_forward[::-1,...]
        chunk_backward = frames_backward[i:i + WINDOW_LENGTH]
        dynamic_image_backward = get_dynamic_image(chunk_backward)
        cv2.imwrite(str(backward_path), dynamic_image_backward)


def get_dynamic_image(frames, normalized: bool = True):
    """ Takes a list of frames and returns either a raw or normalized dynamic image.
    
    Example:
        # 1 dyn img from list of frames
        frames = glob.glob('./example_frames/*.jpg')
        frames = sorted(frames, key=lambda x: int(Path(x).stem))
        frames = [cv2.imread(f) for f in frames]
        dyn_image = get_dynamic_image(frames, normalized=True)

        # sliding window over a long set of frames
        frames = np.array([cv2.imread(str(x)) for x in frame_folder.glob('*.jpg')])
        for i in range(0, len(frames) - WINDOW_LENGTH, STRIDE):
            chunk = frames[i:i + WINDOW_LENGTH]
            assert len(chunk) == WINDOW_LENGTH
            dynamic_image = get_dynamic_image(chunk)

    Args:
        frames (np.ndarray): spectrogram
        normalized (bool): normalize dynamic image. Defaults to True.

    """
    num_channels = frames[0].shape[2]
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]
    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')
    return dynamic_image


def _get_channel_frames(iter_frames: np.ndarray, num_channels: int) -> List[np.ndarray]:
    """Takes a list of frames and returns a list of frame lists split by channel

    Args:
        iter_frames (np.ndarray): list of frames
        num_channels (int): number of channels

    Returns:
        List[np.ndarray]: list of frame lists split by channel
    """
    frames = [[] for channel in range(num_channels)]
    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
    for i in range(len(frames)):
        frames[i] = np.array(frames[i])
    return frames


def _compute_dynamic_image(frames: np.ndarray) -> np.ndarray:
    """Compute dynamic image from sets of images
    Adapted from
    https://github.com/hbilen/dynamic-image-nets
    https://github.com/tcvrick/dynamic-images-for-action-recognition

    Args:
        frames (np.ndarray): sets of frame. Expected shape is (T,H,W,1)

    Returns:
        np.ndarray: dynamic image of shape (H,W,1)
    """
    num_frames, h, w, depth = frames.shape
    # Compute the coefficients for the frames.
    coefficients = np.zeros(num_frames)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)
    # Multiply by the frames by the coefficients and sum the result.
    x1 = np.expand_dims(frames, axis=0)
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
    result = x1 * x2
    return np.sum(result[0], axis=0).squeeze()


def video2keyframes(input_path: str, output_dir: str, no_of_frames: int = 10) -> List[np.ndarray]:
    """Extracts a set of keyframes from a video

    Args:
        input_path (str): input video path
        output_dir (str): output dir path
        no_of_frames (int, optional): max number of keyframes. Defaults to 10.

    Returns:
        List[np.ndarray]: list of keyframes
    """
    vd = Video()
    imgs = vd.extract_frames_as_images(no_of_frames=no_of_frames, file_path=input_path)
    output_dir = Path(output_dir).resolve() / Path(input_path).stem / 'keyframes'
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        cv2.imwrite(str(output_dir / 'keyframe_{0:05d}.png'.format(i)), img)
    return imgs


def video2frames(input_path, output_dir, fps):# data: Tuple[str, str, int]) -> Tuple[int, List[str]]:
    """Extracts the frames from a video

    Args:
        data (Tuple[str, str, int]): video path, output path and fps

    Returns:
        Tuple[int, List[str]]: number of frames and paths of frames
    """
    output_dir = Path(output_dir).resolve()
    if output_dir.exists(): return
    output_dir.mkdir(parents=True, exist_ok=True)
    CMD = f'ffmpeg -loglevel panic -i {str(input_path)} -r {int(fps)} {str(output_dir)}/frame_%05d.jpg -nostdin -vf -an -hide_banner'
    os.system(CMD)

