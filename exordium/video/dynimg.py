from typing import Sequence
from pathlib import Path
import cv2
import numpy as np
from exordium import PathType


def frames2dynimgs(input_dir: PathType, output_dir: PathType, window_length: int = 30, stride: int = 30) -> None:
    input_paths = sorted(list(Path(input_dir).glob('*.png')))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(input_paths) - window_length, stride):
        forward_path = output_dir / f'dynimg_f_{i:05d}.png'
        backward_path = output_dir / f'dynimg_b_{i:05d}.png'

        if not forward_path.exists():
            frames_forward = np.array([cv2.imread(str(x)) for x in input_paths[i:i + window_length]])
            dynamic_image_forward = get_dynamic_image(frames_forward)
            cv2.imwrite(str(forward_path), dynamic_image_forward)

        if not backward_path.exists():
            frames_backward = np.array([cv2.imread(str(x)) for x in input_paths[::-1][i:i + window_length]])
            dynamic_image_backward = get_dynamic_image(frames_backward)
            cv2.imwrite(str(backward_path), dynamic_image_backward)


def get_dynamic_image(frames: np.ndarray | Sequence[np.ndarray], normalized: bool = True):
    """Takes multiple frames and returns either a raw or normalized dynamic image.

    Examples:
        # 1 dyn img from list of frames
        frames = sorted(list(frame_dir.glob('*.png')))
        frames = [cv2.imread(f) for f in frames]
        dyn_image = get_dynamic_image(frames)

        # sliding window over a long set of frames
        frames = np.array([cv2.imread(str(x)) for x in frame_dir.glob('*.png')])
        for i in range(0, len(frames) - WINDOW_LENGTH, STRIDE):
            chunk = frames[i:i + WINDOW_LENGTH]
            dynamic_image = get_dynamic_image(chunk)

    Args:
        frames (np.ndarray): input images
        normalized (bool): normalize dynamic image. Defaults to True.
    """
    num_channels = frames[0].shape[2]
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]
    dynamic_image = cv2.merge(tuple(channel_dynamic_images))

    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype('uint8')

    return dynamic_image


def _get_channel_frames(iter_frames: np.ndarray | Sequence[np.ndarray], num_channels: int) -> list[np.ndarray]:
    """Takes multiple frames and returns a list of frame lists split by channel.

    Args:
        iter_frames (np.ndarray | Sized[np.ndarray]): list of frames or ndarray of shape (T, H, W, 3).
        num_channels (int): number of channels.

    Returns:
        list[np.ndarray]: list of frame lists split by channel.
    """
    frames: list[list[np.ndarray]] = [[] for _ in range(num_channels)]

    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))

    return [np.array(frame) for frame in frames]


def _compute_dynamic_image(frames: np.ndarray) -> np.ndarray:
    """Compute dynamic image from multiple images.

    Adapted from
    https://github.com/hbilen/dynamic-image-nets
    https://github.com/tcvrick/dynamic-images-for-action-recognition

    Args:
        frames (np.ndarray): frames of shape (T, H, W, 1).

    Returns:
        np.ndarray: dynamic image of shape (H, W, 1).
    """
    num_frames = frames.shape[0]

    coefficients = np.zeros(num_frames)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)

    x1 = np.expand_dims(frames, axis=0)
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
    result = x1 * x2
    return np.sum(result[0], axis=0).squeeze()