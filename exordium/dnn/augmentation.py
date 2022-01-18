import numpy as np


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
