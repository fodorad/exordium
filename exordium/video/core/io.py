"""Video input/output utilities."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Sequence
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from exordium.utils.device import get_torch_device


class VideoBackend(ABC):
    """Abstract interface for video decoding backends."""

    @abstractmethod
    def decode_frames(
        self,
        path: str | Path,
        indices: list[int] | np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Decode frames at specific indices.

        Args:
            path: Video file path
            indices: Frame indices to decode
            device: torch.device for decoding

        Returns:
            Tensor of shape (T, C, H, W) with decoded frames

        """

    @abstractmethod
    def decode_all(self, path: str | Path, device: torch.device) -> torch.Tensor:
        """Decode entire video.

        Args:
            path: Video file path
            device: torch.device for decoding

        Returns:
            Tensor of shape (T, C, H, W) with all frames

        """

    @abstractmethod
    def get_metadata(self, path: str | Path) -> dict:
        """Get video metadata.

        Args:
            path: Video file path

        Returns:
            dict with keys: fps, num_frames, height, width, duration

        """


class TorchCodecBackend(VideoBackend):
    """TorchCodec implementation of VideoBackend."""

    def decode_frames(
        self,
        path: str | Path,
        indices: list[int] | np.ndarray,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Decode frames at specific indices using torchcodec."""
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path), device=device)
        frames = decoder.get_frames_at(list(indices)).data  # (T, C, H, W)
        return frames

    def decode_all(
        self,
        path: str | Path,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Decode entire video using torchcodec."""
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path), device=device)
        metadata = decoder.metadata
        frames = decoder.get_frames_in_range(0, metadata.num_frames).data
        return frames

    def get_metadata(self, path: str | Path) -> dict:
        """Get video metadata from torchcodec."""
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path))
        m = decoder.metadata
        return {
            "fps": float(m.average_fps),
            "num_frames": int(m.num_frames),
            "height": int(m.height),
            "width": int(m.width),
            "duration": float(m.num_frames / m.average_fps),
        }


# Global backend configuration
_BACKEND = TorchCodecBackend()


def get_video_metadata(input_path: str | Path) -> dict:
    """Get video metadata using the configured backend.

    Args:
        input_path: Path to video file

    Returns:
        dict with keys: fps, num_frames, height, width, duration

    Example:
        >>> meta = get_video_metadata("video.mp4")
        >>> print(f"Video: {meta['num_frames']} frames at {meta['fps']} fps")

    """
    return _BACKEND.get_metadata(str(input_path))


def load_video(
    input_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    fps: int | float | None = None,
    batch_size: int = 32,
    resize: int | tuple[int, int] | None = None,
    crop: tuple[int, int, int, int] | None = None,
    device_id: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Loads video frames as a tensor.

    Args:
        input_path: Path to the input video.
        start_frame: Start frame index. Defaults to 0.
        end_frame: End frame index (exclusive). None means all frames. Defaults to None.
        fps: Target FPS for frame sampling. If None, uses native FPS. Defaults to None.
             Example: Native 25fps video with fps=10 → samples every 2.5 frames
        batch_size: Number of frames to decode at once. Defaults to 32.
        resize: Resize frames. Can be int (smallest dimension) or (H, W) tuple.
        crop: Crop bounding box (cy, cx, ch, cw). Applied after resize.
        device_id: Device ID for decoding. None uses CPU.

    Returns:
        tuple: (frames, actual_fps)
            - frames: Tensor of shape (T, C, H, W)
            - actual_fps: The FPS of returned frames

    Note:
        Video decoding currently is only supported on CUDA devices and CPU.
        MPS decoding is not supported by torchcodec and will raise an error.

    """
    device = get_torch_device(device_id)

    # Get metadata using backend
    metadata = _BACKEND.get_metadata(str(input_path))
    native_fps = metadata["fps"]
    num_frames = metadata["num_frames"]
    height = metadata["height"]
    width = metadata["width"]

    # Calculate frame indices based on FPS conversion
    if end_frame is None:
        end_frame = num_frames

    if fps is None:
        # No FPS conversion - decode all frames in range
        frame_indices = np.arange(start_frame, end_frame)
        output_fps = native_fps
    else:
        # FPS conversion - sample frames at target rate
        duration = (end_frame - start_frame) / native_fps
        target_frame_count = int(duration * fps)
        if target_frame_count == 0:
            target_frame_count = 1
        frame_interval = (end_frame - start_frame) / target_frame_count
        frame_indices = np.array(
            [int(start_frame + i * frame_interval) for i in range(target_frame_count)]
        )
        output_fps = fps

    # Calculate output dimensions for resize
    aspect_ratio = height / width

    if isinstance(resize, int):
        if aspect_ratio >= 1:
            resize_shape = (int(resize * aspect_ratio), resize)
        else:
            resize_shape = (resize, int(resize / aspect_ratio))
    elif isinstance(resize, tuple):
        resize_shape = resize
    else:
        resize_shape = (height, width)  # native resolution

    # Determine final output shape after crop
    if crop is not None:
        cy, cx, ch, cw = crop
        final_shape = (ch, cw)
    else:
        final_shape = resize_shape

    # Preallocate output tensor
    frames_out = torch.empty(
        (len(frame_indices), 3, final_shape[0], final_shape[1]),
        dtype=torch.uint8,
        device=device,
    )
    decoder_device = str(device) if device.type == "cuda" else None
    with Video(input_path, device=decoder_device) as video:
        for batch_start in range(0, len(frame_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(frame_indices))
            batch_indices = frame_indices[batch_start:batch_end]

            frames = video.get_frames_at(list(batch_indices))

            # Apply resize to entire batch if needed
            if resize is not None:
                frames = TF.resize(
                    frames,
                    size=resize_shape,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )

            # Apply crop to entire batch if needed
            if crop is not None:
                cy, cx, ch, cw = crop
                frames = TF.crop(frames, cy, cx, ch, cw)

            frames_out[batch_start:batch_end] = frames

    return frames_out, output_fps


def load_frames(
    input_path: str | Path,
    frame_ids: list[int] | np.ndarray,
    start_frame: int = 0,
    batch_size: int = 32,
    resize: int | tuple[int, int] | None = None,
    crop: tuple[int, int, int, int] | None = None,
    device_id: int | None = None,
) -> torch.Tensor:
    """Loads specific video frames by their IDs as a tensor.

    Args:
        input_path: Path to the input video.
        frame_ids: List or array of frame indices to load (relative to start_frame).
        start_frame: Offset to add to frame_ids. Defaults to 0.
        batch_size: Number of frames to decode at once. Defaults to 32.
        resize: Resize frames. Can be int (smallest dimension) or (H, W) tuple.
        crop: Crop bounding box (cy, cx, ch, cw). Applied after resize.
        device_id: Device ID for decoding. None uses CPU.

    Returns:
        Tensor of shape (T, C, H, W) containing the requested frames

    Note:
        Video decoding currently is only supported on CUDA devices and CPU.
        MPS decoding is not supported by torchcodec and will raise an error.

    Example:
        >>> frames = load_frames("video.mp4", frame_ids=[0, 10, 20, 30])
        >>> print(frames.shape)  # (4, 3, H, W)

    """
    device = get_torch_device(device_id)

    # Get metadata using backend
    metadata = _BACKEND.get_metadata(str(input_path))
    height = metadata["height"]
    width = metadata["width"]

    # Convert frame_ids to absolute indices
    if isinstance(frame_ids, np.ndarray):
        frame_indices = frame_ids + start_frame
    else:
        frame_indices = np.array(frame_ids) + start_frame

    # Calculate output dimensions for resize
    aspect_ratio = height / width

    if isinstance(resize, int):
        if aspect_ratio >= 1:
            resize_shape = (int(resize * aspect_ratio), resize)
        else:
            resize_shape = (resize, int(resize / aspect_ratio))
    elif isinstance(resize, tuple):
        resize_shape = resize
    else:
        resize_shape = (height, width)  # native resolution

    # Determine final output shape after crop
    if crop is not None:
        cy, cx, ch, cw = crop
        final_shape = (ch, cw)
    else:
        final_shape = resize_shape

    # Preallocate output tensor
    frames_out = torch.empty(
        (len(frame_indices), 3, final_shape[0], final_shape[1]),
        dtype=torch.uint8,
        device=device,
    )

    # Decode frames in batches using persistent decoder
    batch_size = max(1, batch_size)
    decoder_device = str(device) if device.type == "cuda" else None
    with Video(input_path, device=decoder_device) as video:
        for batch_start in range(0, len(frame_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(frame_indices))
            batch_indices = frame_indices[batch_start:batch_end]

            frames = video.get_frames_at(list(batch_indices))

            # Apply resize to entire batch if needed
            if resize is not None:
                frames = TF.resize(
                    frames,
                    size=resize_shape,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )

            # Apply crop to entire batch if needed
            if crop is not None:
                cy, cx, ch, cw = crop
                frames = TF.crop(frames, cy, cx, ch, cw)

            frames_out[batch_start:batch_end] = frames

    return frames_out


def video_to_frames(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    start_number: int = 0,
    fps: int | float | None = None,
    smallest_dim: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
    device_id: int | None = None,
    extension: str = ".png",
    overwrite: bool = False,
) -> None:
    """Extracts and saves the frames from a video.

    Note:
        The start_number is preferred to be 0 as more functionalities in this package assumes it.
        e.g.: 000000.png -> frame_id 0 -> 0. index of extracted features
            in a (num_frames, feature_dim) tensor.

    Args:
        input_path: Path to the input video.
        output_dir: Path to the output directory.
        start_number: Start index of the extracted frames. Defaults to 0.
        fps: Frame per sec. None means that the original fps of the video is used. Defaults to None.
        smallest_dim: Smallest dimension of the frames, height or width.
            None means that the frames is not resized. Defaults to None.
        crop: Crop bounding box defined by (cy, cx, ch, cw)
            where cy=top, cx=left, ch=height, cw=width.
            If crop is given, then first the video will be scaled, then cropped. Defaults to None.
            [0,0]-[0,w]
              |     |
            [h,0]-[h,w]
        device_id: GPU device id for decoding. None or -1 means CPU decoding. Defaults to None.
        extension: File extension for saved frames. Defaults to ".png".
        overwrite: If True, overwrites existing output directory. Defaults to False.

    """
    output_dir = Path(output_dir).resolve()

    # Skip if directory exists and overwrite is False
    if output_dir.exists() and not overwrite:
        logging.info(f"Output directory already exists, skipping: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video using backend
    frames, actual_fps = load_video(
        input_path=input_path,
        fps=fps,
        resize=smallest_dim,
        crop=crop,
        device_id=device_id,
    )

    metadata = get_video_metadata(input_path)
    logging.info(
        f"Extracting frames from {input_path}: "
        f"{metadata['num_frames']} frames at {metadata['fps']:.2f} fps, "
        f"output fps={actual_fps:.2f}, "
        f"extracting {len(frames)} frames"
    )

    save_frames(frames, output_dir, zfill=6, start_number=start_number, extension=extension)
    logging.info(f"Extracted {len(frames)} frames to {output_dir}")


def save_video(
    frames: torch.Tensor | np.ndarray | Sequence[torch.Tensor] | Sequence[np.ndarray],
    output_path: str | Path,
    fps: int | float = 25,
    codec: str = "mp4v",
    overwrite: bool = True,
) -> None:
    """Saves frames as a video file.

    Args:
        frames: Video frames. Can be:
            - torch.Tensor of shape (T, C, H, W) or (T, H, W, C)
            - numpy array of shape (T, C, H, W) or (T, H, W, C)
            - Sequence of tensors (C, H, W) or (H, W, C)
            - Sequence of numpy arrays (H, W, C)
        output_path: Path to output video file
        fps: Frames per second for output video
        codec: FourCC codec string (e.g., "mp4v", "avc1", "h264")
        overwrite: If True, overwrites existing file

    Example:
        >>> frames, fps = load_video("input.mp4")
        >>> save_video(frames, "output.mp4", fps=30)

    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logging.info(f"Video already exists: {output_path}")
        return

    # Convert frames to numpy array (H, W, C) BGR format for cv2
    if isinstance(frames, torch.Tensor):
        # Handle (T, C, H, W) or (T, H, W, C) format
        if frames.ndim == 4:
            if frames.shape[1] == 3:  # (T, C, H, W)
                frames = frames.permute(0, 2, 3, 1)  # → (T, H, W, C)
            frames = frames.cpu().numpy()
        frames_list = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    elif isinstance(frames, np.ndarray):
        # Handle (T, C, H, W) or (T, H, W, C) format
        if frames.ndim == 4:
            if frames.shape[1] == 3:  # (T, C, H, W)
                frames = frames.transpose(0, 2, 3, 1)  # → (T, H, W, C)
        frames_list = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    elif isinstance(frames[0], torch.Tensor):
        # Sequence of (C, H, W) or (H, W, C) tensors
        frames_list = []
        for f in frames:
            if f.ndim == 3 and f.shape[0] == 3:
                f = f.permute(1, 2, 0)  # (C, H, W) → (H, W, C)
            f = f.cpu().numpy()
            frames_list.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    else:
        # Sequence of numpy arrays (H, W, C)
        frames_list = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]

    # Get dimensions
    height, width = frames_list[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Write frames
    for frame in frames_list:
        writer.write(frame)

    writer.release()
    logging.info(f"Saved video to: {output_path} ({len(frames_list)} frames at {fps} fps)")


def sequence_to_video(
    frames: Path | Sequence[np.ndarray] | Sequence[Path],
    output_path: Path,
    fps: int | float = 25,
    overwrite: bool = True,
) -> None:
    """Saves a video to a .mp4 file without audio.

    Args:
        frames (Sequence[np.ndarray]): sequence of frames of shape (H, W, 3) and RGB channel order.
        output_path (Path): path to the output file.
        fps (int | float, optional): frame per sec. Defaults to 25.
        overwrite (bool, optional): if True it overwrites the existing file. Defaults to True.

    """
    # Handle directory input - load all frames as file paths
    if isinstance(frames, (str, Path)):
        frames = sorted([str(elem) for elem in list(Path(frames).iterdir())])

    # Handle file paths - load as numpy arrays
    if isinstance(frames[0], (str, Path)):
        frames = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in frames]

    # Use save_video() for actual encoding
    save_video(frames, output_path, fps=fps, overwrite=overwrite)


def save_frames(
    frames: torch.Tensor | np.ndarray | Sequence[torch.Tensor] | Sequence[np.ndarray],
    output_dir: str | Path,
    start_number: int = 0,
    zfill: int = 6,
    extension: str = ".jpg",
) -> None:
    """Saves a sequence of frames to individual image files in a directory.

    Args:
        frames (torch.Tensor | np.ndarray | Sequence[torch.Tensor] | Sequence[np.ndarray]):
            frames to save, either as a single tensor of shape (T, C, H, W)
            or a sequence of tensors of shape (C, H, W) or numpy arrays of shape (H, W, C).
        output_dir (str | Path): directory to save the frames.
        start_number (int, optional): starting index for frame file names. Defaults to 0.
        zfill (int, optional): zero-padding width for frame file names. Defaults to 6.
        extension (str, optional): file extension for saved frames. Defaults to ".jpg".

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_ind, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
        cv2.imwrite(
            str(output_dir / f"{str(start_number + frame_ind).zfill(zfill)}{extension}"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )


def save_frames_with_ids(
    frames: torch.Tensor | np.ndarray | Sequence[torch.Tensor] | Sequence[np.ndarray],
    frame_ids: Sequence[int] | np.ndarray,
    output_dir: Path,
    zfill: int = 6,
    extension: str = ".jpg",
) -> None:
    """Saves a sequence of frames to individual image files in a directory, using frame IDs.

    Args:
        frames (torch.Tensor | np.ndarray | Sequence[torch.Tensor] | Sequence[np.ndarray]):
            frames to save, either as a single tensor of shape (T, C, H, W)
            or a sequence of tensors of shape (C, H, W) or numpy arrays of shape (H, W, C).
        frame_ids (Sequence[int] | np.ndarray): frame IDs for each frame.
        output_dir (Path): directory to save the frames.
        zfill (int, optional): zero-padding width for frame file names. Defaults to 6.
        extension (str, optional): file extension for saved frames. Defaults to ".jpg".

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_ind, (frame, frame_id) in enumerate(zip(frames, frame_ids)):
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
        cv2.imwrite(
            str(output_dir / f"{str(frame_id).zfill(zfill)}{extension}"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )


def batch_iterator(iterable: Iterable, batch_size: int) -> Generator[list, None, None]:
    """Yields batch size list of objects from an iterable."""
    iterator = iter(iterable)

    while True:
        batch = list(islice(iterator, batch_size))

        if not batch:
            break

        yield batch


def image_to_np(image: str | Path | np.ndarray, channel_order: str = "RGB") -> np.ndarray:
    """Converts commonly used image formats to standardized numpy array.

    Converts image of shape (H, W) or (H, W, 1) or (H, W, 3) and BGR channel order
    to image of shape (H, W, 3) and RGB channel order.

    Args:
        image (np.ndarray | str | Path): image or path to the image.
        channel_order (str, optional): channel order. Supported values are 'RGB', 'BGR', 'HSV',
                                       'LAB' and 'GRAY'. Defaults to RGB.

    Returns:
        np.ndarray: image of shape (H, W, 3) and RGB channel order.

    """
    if isinstance(image, (str, Path)):
        if not Path(image).exists():
            raise FileNotFoundError(f"The file cannot be found: {image}")

        image_bgr = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)  # BGR
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # RGB

    if image.ndim == 3 and image.shape[-1] == 3:
        flag: int | None = None
        match channel_order.lower():
            case "bgr":
                flag = cv2.COLOR_RGB2BGR
            case "hsv":
                flag = cv2.COLOR_RGB2HSV
            case "lab":
                flag = cv2.COLOR_RGB2LAB
            case "gray":
                flag = cv2.COLOR_RGB2GRAY
            case _:  # stay in RGB
                pass

        if flag is not None:
            image = cv2.cvtColor(image, flag)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    return image


def images_to_np(
    images: Sequence[np.ndarray | str | Path],
    channel_order: str = "RGB",
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Converts multiple images to a single np.ndarray of shape (H, W, 3) and RGB channel order.

    Args:
        images (Sequence[np.ndarray | str | Path]): multiple images or image paths.
        channel_order (str, optional): channel order. Supported values are 'RGB', 'BGR', 'HSV',
                                       'LAB' and 'GRAY'. Defaults to RGB.
        resize (tuple[int, int] | None, optional): resize images to a common H x W size.
            None means that the resize is applied. Defaults to None.

    Raises:
        ValueError: if the images do not have the same H, W dimensions.

    Returns:
        np.ndarray: images of shape (N, H, W, 3) and RGB channel order.

    """
    N = len(images)

    if resize is not None:
        H, W = resize
    else:
        H, W = image_to_np(images[0]).shape[:2]  # (H, W, C)

    batched_images = np.empty((N, H, W, 3), dtype=np.uint8)

    for index, image in enumerate(images):
        img = image_to_np(image, channel_order)

        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        batched_images[index, :, :, :] = img

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
            raise ValueError(
                f"The {index}. image in the list has different dimensions."
                f"Expected image of shape {(h, w)} got instead {(image.shape[:2])}"
            )


def interpolate_1d(
    start_index: int, end_index: int, start_data: np.ndarray, end_data: np.ndarray
) -> np.ndarray:
    """Interpolates data using a range.

    Args:
        start_index (int): start index.
        end_index (int): end index.
        start_data (np.ndarray): start data.
        end_data (np.ndarray): end data.

    Returns:
        np.ndarray: interpolated data between start and end data.

    """
    from scipy.interpolate import interp1d

    interp = interp1d(np.array([start_index, end_index]), np.array([start_data, end_data]).T)
    interp_data: np.ndarray = interp(np.arange(start_index, end_index + 1, 1))
    return interp_data[:, 1:-1].T


class ImageSequenceReader(Dataset):
    """PyTorch Dataset for loading image sequences from a directory."""

    def __init__(self, frames_dir: str | Path, transform=None):
        self.frames_dir = Path(frames_dir)
        self.files = sorted(f for f in self.frames_dir.iterdir() if f.is_file())
        self.transform = transform

    def __len__(self):
        """Return number of frames in sequence.

        Returns:
            Number of frames.

        """
        return len(self.files)

    def __getitem__(self, idx):
        """Load image at index.

        Args:
            idx: Frame index.

        Returns:
            Image as numpy array or transformed tensor.

        """
        img = image_to_np(self.frames_dir / self.files[idx])

        if self.transform is not None:
            return self.transform(img)

        return img


class Video:
    """Stateful video reader that keeps the decoder open across operations.

    Wraps torchcodec.VideoDecoder with a persistent handle for efficient
    batch-wise iteration over long videos without reopening the file.

    Args:
        path: Path to the video file.
        device: Device for decoded tensors (e.g., "cpu", "cuda:0").
            Defaults to None (CPU).

    Note:
        Video decoding is only supported on CUDA devices and CPU.
        MPS decoding is not supported by torchcodec.

    Example:
        >>> with Video("long_video.mp4") as v:
        ...     print(f"{v.num_frames} frames at {v.fps} fps")
        ...     for batch in v.iter_batches(batch_size=64):
        ...         process(batch)  # batch is (T, C, H, W)

    """

    def __init__(self, path: str | Path, device: str | torch.device | None = None) -> None:
        from torchcodec.decoders import VideoDecoder

        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._device = device
        self._decoder = VideoDecoder(str(self._path), device=device)
        self._metadata = self._decoder.metadata

    @property
    def fps(self) -> float:
        """Average frames per second."""
        return float(self._metadata.average_fps)

    @property
    def num_frames(self) -> int:
        """Total number of frames."""
        return int(self._metadata.num_frames)

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self._metadata.height)

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self._metadata.width)

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return float(self._metadata.num_frames / self._metadata.average_fps)

    def get_batch(self, start: int, stop: int, step: int = 1) -> torch.Tensor:
        """Decode a range of frames.

        Args:
            start: Start frame index (inclusive).
            stop: Stop frame index (exclusive).
            step: Step between frame indices.

        Returns:
            Tensor of shape (T, C, H, W).

        """
        return self._decoder.get_frames_in_range(start, stop, step).data

    def get_frames_at(self, indices: list[int]) -> torch.Tensor:
        """Decode frames at specific indices.

        Args:
            indices: List of frame indices to decode.

        Returns:
            Tensor of shape (T, C, H, W).

        """
        return self._decoder.get_frames_at(indices).data

    def iter_batches(self, batch_size: int = 32) -> Generator[torch.Tensor, None, None]:
        """Iterate over all frames in fixed-size batches.

        Args:
            batch_size: Number of frames per batch.

        Yields:
            Tensor of shape (T, C, H, W) where T <= batch_size.

        """
        for start in range(0, self.num_frames, batch_size):
            stop = min(start + batch_size, self.num_frames)
            yield self.get_batch(start, stop)

    def __len__(self) -> int:
        """Return total number of frames.

        Returns:
            Number of frames in video.

        """
        return self.num_frames

    def __getitem__(self, key: int | slice) -> torch.Tensor:
        """Get frame(s) by index or slice.

        Args:
            key: Frame index (int) or slice of frame indices.

        Returns:
            Tensor of shape (C, H, W) for single frame or (T, C, H, W) for slice.

        Raises:
            IndexError: If frame index is out of bounds.
            TypeError: If key is not int or slice.

        """
        if isinstance(key, int):
            if key < 0:
                key += self.num_frames
            if key < 0 or key >= self.num_frames:
                raise IndexError(
                    f"Frame index {key} out of range for video with {self.num_frames} frames"
                )
            return self._decoder.get_frame_at(key).data

        if isinstance(key, slice):
            start, stop, step = key.indices(self.num_frames)
            if start >= stop:
                return torch.empty(0, 3, self.height, self.width, dtype=torch.uint8)
            if step == 1:
                return self._decoder.get_frames_in_range(start, stop).data
            indices = list(range(start, stop, step))
            return self._decoder.get_frames_at(indices).data

        raise TypeError(f"Invalid key type: {type(key).__name__}. Expected int or slice.")

    def __repr__(self) -> str:
        """Return string representation of video.

        Returns:
            String with video metadata.

        """
        return (
            f"Video(path={self._path.name!r}, "
            f"frames={self.num_frames}, "
            f"fps={self.fps:.2f}, "
            f"size={self.width}x{self.height}, "
            f"duration={self.duration:.2f}s)"
        )

    def __enter__(self) -> "Video":
        """Enter context manager.

        Returns:
            Video instance.

        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager.

        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value if an error occurred.
            exc_tb: Exception traceback if an error occurred.

        """
        self.close()

    def close(self) -> None:
        """Release the decoder."""
        self._decoder = None
