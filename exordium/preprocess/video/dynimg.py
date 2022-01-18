import cv2
import numpy as np
from pathlib import Path


def frames2dynimgs(input_dir, output_dir):
    WINDOW_LENGTH = 30
    STRIDE = 30
    input_paths = sorted(Path(input_dir).glob('*.png'))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(0, len(input_paths) - WINDOW_LENGTH, STRIDE):
        forward_path = Path(output_dir) / 'dynimg_f_{0:05d}.png'.format(i)
        backward_path = Path(output_dir) / 'dynimg_b_{0:05d}.png'.format(i)
        if not forward_path.exists():
            frames_forward = np.array([cv2.imread(str(x)) for x in input_paths[i:i + WINDOW_LENGTH]])
            dynamic_image_forward = get_dynamic_image(frames_forward)
            cv2.imwrite(str(forward_path), dynamic_image_forward)
        if not backward_path.exists():
            frames_backward = np.array([cv2.imread(str(x)) for x in input_paths[::-1][i:i + WINDOW_LENGTH]])
            dynamic_image_backward = get_dynamic_image(frames_backward)
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
