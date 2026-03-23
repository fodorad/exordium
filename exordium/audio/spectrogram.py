"""Audio spectrogram generation utilities."""

from pathlib import Path
from typing import Any

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


def apply_preemphasis(y: torch.Tensor, coef: float = 0.97) -> torch.Tensor:
    """Apply pre-emphasis filter to audio signal.

    Args:
        y: 1-D audio waveform tensor.
        coef: Pre-emphasis coefficient. Defaults to 0.97.

    Returns:
        Pre-emphasized waveform tensor of the same shape.

    """
    if y.numel() == 0:
        return y
    return torch.cat([y[:1], y[1:] - coef * y[:-1]])


def compute_mfcc(
    y: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 40,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    log_mels: bool = True,
    save_fig: bool = False,
    output_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MFCC features from audio signal.

    Defaults are tuned for speech emotion and sentiment analysis at 16 kHz:

    * ``n_mels=80`` — covers all perceptually relevant speech frequencies
      (0–8 kHz) without the redundancy of 128 bands, which are better suited
      to music.  Used by Whisper, WavLM, and most SER benchmarks.
    * ``n_fft=512`` — 32 ms window gives stable frequency estimates while
      preserving temporal resolution for fast prosodic changes.
    * ``hop_length=160`` — 10 ms hop matches the frame rate expected by most
      pretrained speech models and standard SER evaluation protocols.

    The torchaudio default of ``n_fft=400`` with ``n_mels=128`` produces only
    201 frequency bins, leaving the highest mel bands empty and triggering a
    ``UserWarning``.  The chosen ``n_fft=512`` yields 257 bins, which is
    sufficient for 80 mel bands with no empty filterbanks.

    Args:
        y: 1-D mono waveform tensor.
        sample_rate: Sample rate in Hz.
        n_mfcc: Number of MFCC coefficients. Defaults to 40.
        n_mels: Number of mel filterbanks used internally. Defaults to 80.
        n_fft: FFT size (window length in samples). Defaults to 512.
        hop_length: Hop size between frames in samples. Defaults to 160.
        log_mels: Whether to use log mel spectrogram. Defaults to True.
        save_fig: Whether to save figures. Defaults to False.
        output_path: Directory to save figures.

    Returns:
        Tuple of ``(mfcc, mfcc_preemph)`` as ``(n_mfcc, T)`` numpy arrays.

    """
    if y.ndim > 1:
        y = y.mean(dim=0)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        log_mels=log_mels,
        melkwargs={"n_mels": n_mels, "n_fft": n_fft, "hop_length": hop_length},
    )

    mfcc = mfcc_transform(y.unsqueeze(0)).squeeze(0).numpy()
    mfcc_preemph = mfcc_transform(apply_preemphasis(y).unsqueeze(0)).squeeze(0).numpy()

    if save_fig and output_path:
        prefix_fig = Path(output_path) / "figures"
        prefix_fig.mkdir(parents=True, exist_ok=True)
        save_mfcc_specshow(mfcc, str(prefix_fig / "mfcc.png"), "MFCC")
        save_mfcc_specshow(mfcc_preemph, str(prefix_fig / "mfcc_preemph.png"), "MFCC preemphasis")

    return mfcc, mfcc_preemph


def compute_melspec(
    y: torch.Tensor,
    sample_rate: int,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    f_max: int = 8000,
    save_fig: bool = False,
    output_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mel spectrogram features from audio signal.

    Defaults are tuned for speech emotion and sentiment analysis at 16 kHz:

    * ``n_mels=80`` — covers all perceptually relevant speech frequencies
      (0–8 kHz) without the redundancy of 128 bands, which are better suited
      to music.  Used by Whisper, WavLM, and most SER benchmarks.
    * ``n_fft=512`` — 32 ms window gives stable frequency estimates while
      preserving temporal resolution for fast prosodic changes.
    * ``hop_length=160`` — 10 ms hop matches the frame rate expected by most
      pretrained speech models and standard SER evaluation protocols.

    The torchaudio default of ``n_fft=400`` with ``n_mels=128`` produces only
    201 frequency bins, leaving the highest mel bands empty and triggering a
    ``UserWarning``.  The chosen ``n_fft=512`` yields 257 bins, which is
    sufficient for 80 mel bands with no empty filterbanks.

    Args:
        y: 1-D mono waveform tensor.
        sample_rate: Sample rate in Hz.
        n_mels: Number of mel bands. Defaults to 80.
        n_fft: FFT size (window length in samples). Defaults to 512.
        hop_length: Hop size between frames in samples. Defaults to 160.
        f_max: Maximum frequency in Hz. Defaults to 8000.
        save_fig: Whether to save figures. Defaults to False.
        output_path: Directory to save figures.

    Returns:
        Tuple of ``(melspec_db, melspec_preemph_db)`` as ``(n_mels, T)``
        numpy arrays in dB.

    """
    if y.ndim > 1:
        y = y.mean(dim=0)

    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, f_max=f_max
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    melspec_db = to_db(melspec_transform(y.unsqueeze(0))).squeeze(0).numpy()
    melspec_pre_db = to_db(melspec_transform(apply_preemphasis(y).unsqueeze(0))).squeeze(0).numpy()

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
        features: Input features of shape ``(n_features, T)`` as a numpy array.
        return_all: If True, return ``(features, delta, delta2)``.
            Defaults to False.

    Returns:
        Delta features of shape ``(n_features, T)`` or a tuple of three
        ``(n_features, T)`` arrays when ``return_all=True``.

    """
    t = torch.from_numpy(features).unsqueeze(0)  # (1, n_features, T)
    delta = torchaudio.functional.compute_deltas(t).squeeze(0).numpy()
    delta2 = (
        torchaudio.functional.compute_deltas(torch.from_numpy(delta).unsqueeze(0))
        .squeeze(0)
        .numpy()
    )

    if return_all:
        return features, delta, delta2
    return delta


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
    n_mels: int = 80,
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
    output: dict[str, Any] = {
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
    y = waveform.squeeze()  # (T,) tensor

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
