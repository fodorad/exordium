"""Audio spectrogram generation utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def preprocess_audio(
    waveform: torch.Tensor, waveform_sample_rate: int, target_sample_rate: int
) -> tuple[torch.Tensor, int]:
    """Convert waveform to mono and resample if needed.

    Args:
        waveform: Input audio waveform (C, T)
        waveform_sample_rate: Original sample rate of the waveform
        target_sample_rate: Target sample rate to resample to

    Returns:
        Processed waveform and original sample rate

    """
    # Convert to mono
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if waveform_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=waveform_sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    return waveform, target_sample_rate


def apply_preemphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to audio signal.

    Args:
        y: Audio signal
        coef: Pre-emphasis coefficient

    Returns:
        Pre-emphasized signal

    """
    y_preemph = np.empty_like(y)
    if y.size > 0:
        y_preemph[0] = y[0]
        y_preemph[1:] = y[1:] - coef * y[:-1]
    return y_preemph


def compute_mfcc(
    y: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 40,
    log_mels: bool = True,
    save_fig: bool = False,
    output_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute MFCC features from audio signal.

    Args:
        y: Audio signal (mono, 1D array)
        sample_rate: Sample rate
        n_mfcc: Number of MFCC coefficients
        log_mels: Whether to use log mel spectrogram
        save_fig: Whether to save figures
        output_path: Directory to save figures

    Returns:
        Tuple of (mfcc_features, preemphasized_mfcc_features)

    """
    # Ensure mono audio
    if y.ndim > 1:
        y = y.mean(axis=0)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=log_mels
    )

    # Compute MFCC
    mfcc = mfcc_transform(torch.from_numpy(y).unsqueeze(0)).squeeze(0).numpy()

    # Compute pre-emphasized MFCC
    y_preemph = apply_preemphasis(y)
    mfcc_preemph = mfcc_transform(torch.from_numpy(y_preemph).unsqueeze(0)).squeeze(0).numpy()

    # Save figures if requested
    if save_fig and output_path:
        prefix_fig = Path(output_path) / "figures"
        prefix_fig.mkdir(parents=True, exist_ok=True)
        save_mfcc_specshow(mfcc, str(prefix_fig / "mfcc.png"), "MFCC")
        save_mfcc_specshow(mfcc_preemph, str(prefix_fig / "mfcc_preemph.png"), "MFCC preemphasis")

    return mfcc, mfcc_preemph


def compute_melspec(
    y: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    f_max: int = 8000,
    save_fig: bool = False,
    output_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mel spectrogram features from audio signal.

    Args:
        y: Audio signal (mono, 1D array)
        sample_rate: Sample rate
        n_mels: Number of mel bands
        f_max: Maximum frequency
        save_fig: Whether to save figures
        output_path: Directory to save figures

    Returns:
        Tuple of (melspec_db, melspec_preemph_db)

    """
    # Ensure mono audio
    if y.ndim > 1:
        y = y.mean(axis=0)

    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, f_max=f_max
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    # Compute Mel spectrogram
    melspec = melspec_transform(torch.from_numpy(y).unsqueeze(0))
    melspec_db = to_db(melspec).squeeze(0).numpy()

    # Compute pre-emphasized Mel spectrogram
    y_preemph = apply_preemphasis(y)
    melspec_pre = melspec_transform(torch.from_numpy(y_preemph).unsqueeze(0))
    melspec_pre_db = to_db(melspec_pre).squeeze(0).numpy()

    # Save figures if requested
    if save_fig and output_path:
        prefix_fig = Path(output_path) / "figures"
        prefix_fig.mkdir(parents=True, exist_ok=True)
        save_melspec_specshow(melspec_db, str(prefix_fig / "melspec_dB.png"), "Log-Mel spectrogram")
        save_melspec_specshow(
            melspec_pre_db,
            str(prefix_fig / "melspec_dB_preemph.png"),
            "Log-Mel spectrogram preemphasis",
        )

    return melspec_db, melspec_pre_db


def compute_deltas(
    features: np.ndarray, return_all: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute delta and delta-delta features from input features.

    Args:
        features: Input features (n_features, time)
        return_all: Whether to return all delta variants

    Returns:
        Delta features or tuple of (original, delta, delta2)

    """
    # Compute deltas
    deltas = np.zeros_like(features)
    for t in range(1, features.shape[1] - 1):
        deltas[:, t] = (features[:, t + 1] - features[:, t - 1]) / 2

    # Compute delta-deltas
    delta2 = np.zeros_like(features)
    for t in range(1, deltas.shape[1] - 1):
        delta2[:, t] = (deltas[:, t + 1] - deltas[:, t - 1]) / 2

    if return_all:
        return features, deltas, delta2
    return deltas


def save_mfcc_specshow(
    data: np.ndarray, output_path: str, title: str, figsize: tuple[int, int] = (10, 4)
) -> None:
    """Save MFCC plot to file.

    Args:
        data: MFCC data (n_mfcc, time)
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size

    """
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar()
    plt.savefig(output_path)
    plt.close()


def save_melspec_specshow(
    data: np.ndarray,
    output_path: str,
    title: str,
    is_delta: bool = False,
    figsize: tuple[int, int] = (10, 4),
) -> None:
    """Save mel-spectrogram plot to file.

    Args:
        data: Mel-spectrogram data (n_mels, time)
        output_path: Path to save figure
        title: Figure title
        is_delta: Whether data is delta/delta2
        figsize: Figure size

    """
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
    if not is_delta:
        plt.colorbar(format="%+2.0f dB")
    else:
        plt.colorbar()
    plt.savefig(output_path)
    plt.close()


def process_audio_file(
    input_path: Path | str,
    output_path: Path | str,
    sample_rate: int = 44100,
    save_fig: bool = False,
    save_npy: bool = True,
    n_mfcc: int = 40,
    n_mels: int = 128,
    f_max: int = 8000,
) -> dict:
    """Process audio file and return features.

    Args:
        input_path: Path to input audio file
        output_path: Directory to save output files
        sample_rate: Target sample rate
        save_fig: Whether to save figures
        save_npy: Whether to save numpy files
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bands
        f_max: Maximum frequency

    Returns:
        Dictionary containing all computed features

    """
    # Initialize output dictionary
    output = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sample_rate": sample_rate,
        "save_fig": save_fig,
        "save_npy": save_npy,
    }

    waveform, waveform_sr = torchaudio.load(input_path)  # (C, T)

    # Load and preprocess audio
    waveform, waveform_sr = preprocess_audio(
        waveform=waveform, waveform_sample_rate=waveform_sr, target_sample_rate=sample_rate
    )
    y = waveform.squeeze().numpy()

    # Compute features
    output["mfcc"], output["mfcc_preemph"] = compute_mfcc(
        y, sample_rate, n_mfcc, save_fig=save_fig, output_path=output_path
    )

    output["melspec_db"], output["melspec_preemph_db"] = compute_melspec(
        y, sample_rate, n_mels, f_max, save_fig=save_fig, output_path=output_path
    )

    # Compute deltas for all features
    output["mfcc_deltas"] = compute_deltas(output["mfcc"])
    output["mfcc_preemph_deltas"] = compute_deltas(output["mfcc_preemph"])
    output["melspec_deltas"] = compute_deltas(output["melspec_db"])
    output["melspec_preemph_deltas"] = compute_deltas(output["melspec_preemph_db"])

    # Save numpy files if requested
    if save_npy:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "mfcc.npy", output["mfcc"])
        np.save(output_dir / "mfcc_preemph.npy", output["mfcc_preemph"])
        np.save(output_dir / "melspec_db.npy", output["melspec_db"])
        np.save(output_dir / "melspec_preemph_db.npy", output["melspec_preemph_db"])
        np.save(output_dir / "mfcc_deltas.npy", output["mfcc_deltas"])
        np.save(output_dir / "mfcc_preemph_deltas.npy", output["mfcc_preemph_deltas"])
        np.save(output_dir / "melspec_deltas.npy", output["melspec_deltas"])
        np.save(output_dir / "melspec_preemph_deltas.npy", output["melspec_preemph_deltas"])

    return output
