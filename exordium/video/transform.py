from PIL import Image
import numpy as np
import torch


def center_crop(x: np.ndarray, center_crop_size: tuple[int, int], **kwargs) -> np.ndarray:
    """Center crop

    Args:
        x (np.array): input image
        center_crop_size (tuple[int, int]): crop size

    Returns:
        np.ndarray: cropped image
    """
    centerh, centerw = x.shape[0]//2, x.shape[1]//2
    halfh, halfw = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerh-halfh:centerh+halfh, centerw-halfw:centerw+halfw, :]


def apply_10_crop(img: np.ndarray, crop_size: tuple[int, int] = (224,224)) -> np.ndarray:
    """Applies 10-crop method to image

    Args:
        img (np.ndarray): image. Expected shape is (H,W,C)
        crop_size (tuple[int, ...], optional): crop size. Defaults to (224,224).

    Returns:
        np.ndarray: 10-crop with shape (10,crop_size[0],crop_size[1],C)
    """
    h = crop_size[0]
    w = crop_size[1]
    flipped_X = np.fliplr(img)
    crops = [
        img[:h,:w, :], # Upper Left
        img[:h, img.shape[1]-w:, :], # Upper Right
        img[img.shape[0]-h:, :w, :], # Lower Left
        img[img.shape[0]-h:, img.shape[1]-w:, :], # Lower Right
        center_crop(img, (h, w)),

        flipped_X[:h,:w, :],
        flipped_X[:h, flipped_X.shape[1]-w:, :],
        flipped_X[flipped_X.shape[0]-h:, :w, :],
        flipped_X[flipped_X.shape[0]-h:, flipped_X.shape[1]-w:, :],
        center_crop(flipped_X, (h, w))
    ]
    return np.array(crops)


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    def eraser(input_img):
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

        c = np.random.uniform(v_l, v_h)
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))

        input_img[top:top + h, left:left + w, :] = c
        return input_img

    return eraser


def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking_max_percentage=0.05,
                 time_masking_max_percentage=0.05,
                 mean=0) -> np.ndarray:
    """Spectrogram augmentation technique
    paper: https://arxiv.org/pdf/1904.08779.pdf
    The expected input is a zero centered (or standardized) spectrogram of shape (H,W,C).
    Height (frequency axis), Width (time axis), Channels

    Args:
        spec (np.ndarray): spectrogram
        num_mask (int, optional): number of masks. Defaults to 2.
        freq_masking_max_percentage (float, optional): maximum high of masks (frequency axis). Defaults to 0.05.
        time_masking_max_percentage (float, optional): maximum width of masks (time axis). Defaults to 0.05.
        mean (int, optional): mean value of the input spectrogram. Defaults to 0.

    Returns:
        np.ndarray: augmented spectrogram
    """
    # spec.shape == (h, w, c), (n_freq, n_win, c)
    assert spec.ndim == 3
    spec = spec.copy()

    for _ in range(num_mask):
        n_freq, n_frames, _ = spec.shape
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * n_freq)
        f0 = np.random.uniform(low=0.0, high=n_freq - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0+num_freqs_to_mask,:,:] = mean
        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)
        num_frames_to_mask = int(time_percentage * n_frames)
        t0 = np.random.uniform(low=0.0, high=n_frames - num_frames_to_mask)
        t0 = int(t0)
        spec[:,t0:t0+num_frames_to_mask,:] = mean

    return spec


def flip(data):
    temp = np.zeros(data.shape)
    for i in range(data.shape[0]):
        img = Image.fromarray(np.reshape(data[i, :], (int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1])))))
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        temp[i, :] = np.reshape(np.array(img), (1, data.shape[1]))
    return temp


def crop_and_pad_window(x: np.ndarray, win_size: int, m_freq: int | float, timestep: int | float):
    start = int(np.clip((timestep - win_size) * m_freq, a_min=0, a_max=None))
    end = int(timestep * m_freq)
    x_padded = np.zeros((int(win_size*m_freq),x.shape[1]))
    x_padded[:x.shape[0],:] = x[start:end,...]
    return x_padded


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(np.array(x)).float() / 255.


class Resize:
    def __init__(self, size, mode="bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, video):
        size = self.size
        scale = None

        if isinstance(size, int):
            scale = float(size) / min(video.shape[-2:])
            size = None

        return torch.nn.functional.interpolate(video, size=size, scale_factor=scale,
                                               mode=self.mode, align_corners=False, recompute_scale_factor=True)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        size = self.size

        if isinstance(size, int):
            size = size, size

        th, tw = size
        h, w = video.shape[-2:]
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return video[..., i:(i + th), j:(j + tw)]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)
        return (video - mean) / std


class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)
        return (video * std) + mean
