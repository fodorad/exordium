import os
import logging
from pathlib import Path
from typing import Sequence, Iterable, Generator
from itertools import islice
from PIL import Image
import cv2
import numpy as np
from scipy.interpolate import interp1d
import decord
import moviepy.editor as mpy
from exordium import PathType


def video2frames(input_path: PathType,
                 output_dir: PathType,
                 start_number: int = 0,
                 fps: int | float | None = None,
                 smallest_dim: int | None = None,
                 crop: tuple[int, int, int, int] | None = None,
                 overwrite: bool = False) -> None:
    """Extracts and saves the frames from a video.

    Note:
        The start_number is preferred to be 0 as more functionalities in this package assumes it.
        e.g.: 000000.png -> frame_id 0 -> 0. index of extracted features in a (num_frames, feature_dim) tensor.

    Args:
        input_path (str): path to the input video.
        output_dir (str): path to the output directory.
        start_number (int): start index of the extracted frames. Defaults to 0.
        fps (int | float | None, optional): frame per sec. None means that the original fps of the video is used. Defaults to None.
        smallest_dim (int | None, optional): smallest dimension of the frames, height or width.
                            None means that the frames is not resized. Defaults to None.
        crop (tuple[int, int, int, int] | None, optional): crop bounding box defined by (x, y, h, w).
                                                           If crop is given, then first the video will be scaled, then cropped. Defaults to None.
                                                           [0,0]-[0,w]
                                                             |     |
                                                           [h,0]-[h,w]
    """
    output_dir = Path(output_dir).resolve()

    if output_dir.exists() and not overwrite:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    vid = cv2.VideoCapture(str(input_path))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    if smallest_dim is not None:
        w, h = (-1, smallest_dim) if height > width else (smallest_dim, -1)
    else:
        w, h = height, width

    if crop is None:
        crop_str = f'scale={h}:{w}'
    else:
        cy, cx, cw, ch = crop
        crop_str = f"scale={h}:{w}:flags=neighbor,crop={ch}:{cw}:{cx}:{cy}"

    if fps is None:
        CMD = f'ffmpeg -loglevel panic -i {str(input_path)} -vf {crop_str} -start_number {start_number} {str(output_dir)}/%06d.png -nostdin -vf -an -hide_banner'
    else:
        CMD = f'ffmpeg -loglevel panic -i {str(input_path)} -r {fps} -vf {crop_str} -start_number {start_number} {str(output_dir)}/%06d.png -nostdin -vf -an -hide_banner'

    logging.info(CMD)
    os.system(CMD)


def frames2video(frames: PathType | Sequence[str] | Sequence[np.ndarray],
                 output_path: PathType,
                 fps: int | float = 25,
                 extension: str = '.png',
                 overwrite: bool = False) -> None:
    """Saves frames to a video without audio.

    Args:
        frames (PathType | Sequence[str] | Sequence[np.ndarray]): frames or path to the frames.
        output_path (PathType): path to the output video.
        fps (int | float, optional): frame per sec. Defaults to 25.
        extension (str, optional): frame file extension. Defaults to '.png'.
        overwrite (bool, optional): if True it overwrites existing file. Defaults to False.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        logging.info(f'Video already exists')
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, (str, os.PathLike)):
        frames = sorted([str(elem) for elem in list(Path(frames).iterdir()) if elem.suffix == extension])

    logging.info(f'Found {len(frames)} frames.')
    movie_clip = mpy.ImageSequenceClip(frames, fps)
    movie_clip.write_videofile(str(output_path), logger="bar")
    movie_clip.close()
    logging.info(f'Video is done: {str(output_path)}')


def sequence2video(frames: Sequence[np.ndarray],
                   output_path: PathType,
                   fps: int | float = 25,
                   overwrite: bool = True) -> None:
    """Saves a video to a .mp4 file without audio.

    Args:
        frames (Sequence[np.ndarray]): sequence of frames of shape (H, W, 3) and RGB channel order.
        output_path (PathType): path to the output file.
        fps (int | float, optional): frame per sec. Defaults to 25.
        overwrite (bool, optional): if True it overwrites the existing file. Defaults to True.
    """
    height, width = frames[0].shape[:2]

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        logging.info(f'Video already exists')
        return

    fourcc = cv2.VideoWriter_fourcc(*'avc1') # type: ignore
    output_video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for idx in range(len(frames)):
        frame = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
        output_video.write(frame)

    output_video.release()
    logging.info(f'Video is done: {output_path}')


def vr2video(video: decord.VideoReader, output_path: PathType, fps: int | float = 25) -> None:
    """Saves a video to a .mp4 file without audio.

    Args:
        video (decord.VideoReader): video as a VideoReader object.
        output_path (PathType): path to the output file.
        fps (int | float, optional): frame per sec. Defaults to 25.
    """
    height, width = video[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    output_video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for idx in range(len(video)):
        frame = video[idx]
        frame = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR)
        output_video.write(frame)

    output_video.release()


def write_frames_with_audio(video: decord.VideoReader, audio_path: PathType, output_video_path: PathType, fps: int | float = 25) -> None:
    """Write frames to a video file with audio.

    Args:
        video (decord.VideoReader): video as a VideoReader object.
        audio (PathType): path to the audio file.
        output_video_path (PathType): path to the video file.
        fps (int | float, optional): frame per sec. Defaults to 25.
    """
    height, width = video[0].shape[:2]

    # create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    output_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    for idx in range(len(video)):
        frame = video[idx]
        frame = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR)
        output_video.write(frame)
    output_video.release()

    # add audio to video
    CMD = f"ffmpeg -i {str(output_video_path)} -i {str(audio_path)} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {output_video_path}_with_audio.mp4"
    os.system(CMD)
    os.system(f'rm {audio_path}')


def batch_iterator(iterable: Iterable, batch_size: int) -> Generator[list, None, None]:
    """Yields batch size list of objects from an iterable."""
    iterator = iter(iterable)

    while True:
        batch = list(islice(iterator, batch_size))

        if not batch:
            break

        yield batch


def image2np(image: np.ndarray | Image.Image | PathType, channel_order: str = 'RGB') -> np.ndarray:
    """Converts commonly used image formats (path to image, grayscale np.ndarray, BGR np.ndarray, PIL.Image)
    of shape (H, W) or (H, W, 1) or (H, W, 3) and BGR channel order
    to image of shape (H, W, 3) and RGB channel order.

    Args:
        image (np.ndarray | Image.Image | PathType): image or path to the image.
        channel_order (str, optional): channel order. Supported values are 'RGB', 'BGR', 'HSV',
                                       'LAB' and 'GRAY'. Defaults to RGB.

    Returns:
        np.ndarray: image of shape (H, W, 3) and RGB channel order.
    """
    if isinstance(image, (str, os.PathLike)):

        if not Path(image).exists():
            raise FileNotFoundError(f'The file cannot be found: {image}')

        image_path = image
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED) # BGR
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # RGB

        if image is None:
            image = Image.open(image_path) # RGB

    if isinstance(image, Image.Image):
        return np.array(image.convert(channel_order))

    if image.ndim == 3 and image.shape[-1] == 3:

        flag: int | None = None
        match channel_order.lower():
            case 'bgr':
                flag = cv2.COLOR_RGB2BGR
            case 'hsv':
                flag = cv2.COLOR_RGB2HSV
            case 'lab':
                flag = cv2.COLOR_RGB2LAB
            case 'gray':
                flag = cv2.COLOR_RGB2GRAY
            case _: # stay in RGB
                pass

        if flag is not None:
            image = cv2.cvtColor(image, flag)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    return image


def images2np(images: Sequence[np.ndarray | Image.Image | PathType], channel_order: str = 'RGB', resize: tuple[int, int] | None = None) -> np.ndarray:
    """Converts multiple images to a single np.ndarray of shape (H, W, 3) and RGB channel order.

    Args:
        images (Sequence[np.ndarray | Image.Image | PathType]): multiple images or image paths.
        channel_order (str, optional): channel order. Supported values are 'RGB', 'BGR', 'HSV',
                                       'LAB' and 'GRAY'. Defaults to RGB.
        resize (tuple[int, int] | None, optional): resize images to a common size. None means that the resize is applied. Defaults to None.

    Raises:
        ValueError: if the images do not have the same H, W dimensions.

    Returns:
        np.ndarray: images of shape (N, H, W, 3) and RGB channel order.
    """
    N = len(images)
    H, W  = image2np(images[0]).shape if resize is None else resize
    batched_images = np.empty((N, H, W, 3), dtype=np.uint8)

    for index, image in enumerate(images):
        img = image2np(image, channel_order)

        if resize:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

        else:
            if img.shape[:2] != (H, W):
                raise ValueError(f'The {index}. image in the list has different dimensions.' \
                                 f'Expected image of shape {(H, W)} got instead {(img.shape[:2])}')

        batched_images[index,:,:,:] = img

    return batched_images


def check_same_image_dims(images: Sequence) -> None:
    """Checks that every image in the sequence of images has the same dimensionality.

    Args:
        images (Sequence): multiple images.

    Raises:
        ValueError: if the images do not have the same H, W dimensions.
    """
    h, w = images[0].shape[:2]
    for index, image in enumerate(images):
        if image.shape[:2] != (h, w):
            raise ValueError(f'The {index}. image in the list has different dimensions.' \
                             f'Expected image of shape {(h, w)} got instead {(image.shape[:2])}')


def interpolate_1d(start_index: int, end_index: int, start_data: np.ndarray, end_data: np.ndarray) -> np.ndarray:
    """Interpolates data using a range.

    Args:
        start_index (int): start index.
        end_index (int): end index.
        start_data (np.ndarray): start data.
        end_data (np.ndarray): end data.

    Returns:
        np.ndarray: interpolated data between start and end data.
    """
    interp = interp1d(np.array([start_index, end_index]),
                      np.array([start_data, end_data]).T)
    interp_data: np.ndarray = interp(np.arange(start_index, end_index + 1, 1))
    return interp_data[:, 1:-1].T


if __name__ == '__main__':
    video2frames(input_path='data/videos/9KAqOrdiZ4I.001.mp4',
                 output_dir='data/processed/frames/9KAqOrdiZ4I.001',
                 crop=(100, 0, 100, 500))
    video2frames(input_path='data/videos/multispeaker_720p.mp4',
                 output_dir='data/processed/frames/multispeaker_720p',
                 smallest_dim=720)
    video2frames(input_path='data/videos/multispeaker_360p.mp4',
                 output_dir='data/processed/frames/multispeaker_360p',
                 smallest_dim=360)