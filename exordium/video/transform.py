from typing import Callable, Sequence
import cv2
import numpy as np
import torch


def rotate_vector(xy: np.ndarray, rotation_degree: float) -> np.ndarray:
    """Rotates a vector.

    Args:
        xy (np.ndarray): vector represented as XY coords.
        rotation_degree (float): rotation in degree.

    Returns:
        np.ndarray: rotated vector.
    """
    theta = np.deg2rad(rotation_degree)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    xy_rot = np.dot(R, xy)
    return xy_rot


def rotate_face(face: np.ndarray, rotation_degree: float) -> tuple[np.ndarray, np.ndarray]:
    """Align the X axes of the Head Coordinate System and the Camera Coordinate System.
    Args:
        face (np.ndarray): face image of shape (H, W, 3).
        rotation_degree (float): rotation in degree
    Returns:
        tuple[np.ndarray, np.ndarray]: rotated face, rotation matrix
    """
    height, width = face.shape[:2]
    R = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_degree, 1)
    face_rotated = cv2.warpAffine(face, R, (width, height))
    return face_rotated, R


def spec_augment(spec: np.ndarray,
                 num_mask: int = 2,
                 freq_masking_max_percentage: float = 0.05,
                 time_masking_max_percentage: float = 0.05,
                 mean: float = 0) -> np.ndarray:
    """Spectrogram augmentation technique.

    paper: https://arxiv.org/pdf/1904.08779.pdf

    Args:
        spec (np.ndarray): zero centered (or standardized) input spectrogram of shape (H,W,C). Height is frequency axis, Width is time axis.
        num_mask (int, optional): number of masks. Defaults to 2.
        freq_masking_max_percentage (float, optional): maximum high of masks (frequency axis). Defaults to 0.05.
        time_masking_max_percentage (float, optional): maximum width of masks (time axis). Defaults to 0.05.
        mean (int, optional): mean value of the input spectrogram. Defaults to 0.

    Returns:
        np.ndarray: augmented spectrogram.
    """
    spec = spec.copy()

    for _ in range(num_mask):
        n_freq, n_frames, _ = spec.shape

        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * n_freq)
        f0 = int(np.random.uniform(low=0.0, high=n_freq - num_freqs_to_mask))
        spec[f0:f0+num_freqs_to_mask,:,:] = mean

        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)
        num_frames_to_mask = int(time_percentage * n_frames)
        t0 = int(np.random.uniform(low=0.0, high=n_frames - num_frames_to_mask))
        spec[:,t0:t0+num_frames_to_mask,:] = mean

    return spec


def crop_and_pad_window(x: np.ndarray, win_size: int, m_freq: int | float, timestep: int | float):
    start = int(np.clip((timestep - win_size) * m_freq, a_min=0, a_max=None))
    end = int(timestep * m_freq)
    x_padded = np.zeros((int(win_size*m_freq),x.shape[1]))
    x_padded[:x.shape[0],:] = x[start:end,...]
    return x_padded


def get_random_eraser(p: float = 0.5,
                      s_l: float = 0.02,
                      s_h: float = 0.4,
                      r_1: float = 0.3,
                      r_2: float = 1/0.3,
                      v_l: float = 0,
                      v_h: float = 255,
                      pixel_level: bool = False) -> Callable[[np.ndarray], np.ndarray]:

    def eraser(input_img: np.ndarray) -> np.ndarray:
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h, (h, w, img_c)) if pixel_level else np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img

    return eraser


class ToTensor:
    """Converts np.ndarray with value range [0..255] to torch Tensor with value range [0..1].
    Video processing class, similar to torchvision but for videos.
    """
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(x)).float() / 255.


class Resize:
    """Resizes video represented as a torch Tensor of shape (C, T, H, W).
    Video processing class, similar to torchvision but for videos.
    """
    def __init__(self, size: int | None, mode: str = "bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        size = self.size
        scale = None

        if isinstance(size, int):
            scale = float(size) / min(video.shape[-2:])
            size = None

        return torch.nn.functional.interpolate(video, size=size, scale_factor=scale,
                                               mode=self.mode, align_corners=False, recompute_scale_factor=True)


class CenterCrop:
    """Center crops a video represented as a torch Tensor of shape (C, T, H, W).
    Video processing class, similar to torchvision but for videos.
    """
    def __init__(self, size: int | tuple[int, int]):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        size = self.size

        if isinstance(size, int):
            size = size, size

        th, tw = size
        h, w = video.shape[-2:]
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return video[..., i:(i + th), j:(j + tw)]


class Normalize:
    """Standardizes a video represented as a torch Tensor of shape (C, T, H, W).
    Video processing class, similar to torchvision but for videos.
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        shape = (-1,) + (1,) * (video.dim() - 1)
        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)
        return (video - mean) / std


class Denormalize:
    """Reverses the standardization of a video represented as a torch Tensor of shape (C, T, H, W).
    Video processing class, similar to torchvision but for videos.
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        shape = (-1,) + (1,) * (video.dim() - 1)
        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)
        return (video * std) + mean