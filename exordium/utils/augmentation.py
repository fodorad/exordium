import numpy as np


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
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
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
