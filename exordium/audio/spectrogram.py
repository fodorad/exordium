from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from exordium import PathType


def deltas(feature: np.ndarray) -> np.ndarray:
    """Calculate differential (delta) and acceleration (deltadelta) features from MFCC or spectrograms.

    Args:
        feature (np.ndarray): MFCC or spectrogram.

    Returns:
        np.ndarray: tensor of feature, deltas and deltadeltas.
    """
    differential = librosa.feature.delta(feature)
    acceleration = librosa.feature.delta(feature, order=2)
    return np.stack([feature, differential, acceleration], axis=2)


def audio2logmelspec(input_path: PathType, output_path: PathType, sr: int, save_fig: bool = False) -> None:
    """Creates the MFCCs, log mel-spectrogram, delta and deltadelta features from audio files.

    Args:
        input_path (PathType): path to the audio file.
        output_path (PathType): path to the output file.
        sr (int): sample rate.
        save_fig (bool, optional). saves the figures. Defaults to False.
    """
    n_fft = 2048
    hop_length = 512
    n_mfcc = 40
    max_frequency = 8000
    output_path = Path(output_path)

    if output_path.exists():
        return

    output_path.mkdir(parents=True, exist_ok=True)
    y, _ = librosa.load(input_path, sr=sr) # sr=22050

    # preemphasis
    y_preemph = librosa.effects.preemphasis(y, coef=0.97, zi=[y[1]])

    # mfcc with differential and acceleration
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_preemph = librosa.feature.mfcc(y=y_preemph, sr=sr, n_mfcc=n_mfcc)
    mfcc_tensor = deltas(mfcc)
    mfcc_preemph_tensor = deltas(mfcc_preemph)

    # save npy
    np.save(str(output_path / 'mfcc_preemph.npy'), mfcc_preemph_tensor)

    # save figures
    if save_fig:
        prefix_fig = output_path / 'figures'
        prefix_fig.mkdir(parents=True, exist_ok=True)
        save_mfcc_specshow(mfcc_tensor[...,0], str(prefix_fig / 'mfcc.png'), 'MFCC')
        save_mfcc_specshow(mfcc_tensor[...,1], str(prefix_fig / 'mfcc_delta.png'), 'MFCC-Delta')
        save_mfcc_specshow(mfcc_tensor[...,2], str(prefix_fig / 'mfcc_delta2.png'), 'MFCC-Delta2')
        save_mfcc_specshow(mfcc_preemph_tensor[...,0], str(prefix_fig / 'mfcc_preemph.png'), 'MFCC preemphasis')
        save_mfcc_specshow(mfcc_preemph_tensor[...,1], str(prefix_fig / 'mfcc_delta_preemph.png'), 'MFCC-Delta preemphasis')
        save_mfcc_specshow(mfcc_preemph_tensor[...,2], str(prefix_fig / 'mfcc_delta2_preemph.png'), 'MFCC-Delta2 preemphasis')

    # melspec db
    for n_mels in [128]: # [80, 128, 224]:
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=max_frequency)
        melspec_dB = librosa.power_to_db(melspec, ref=np.max)
        melspec_dB_tensor = deltas(melspec_dB)

        melspec_preemph = librosa.feature.melspectrogram(y=y_preemph, sr=sr, n_mels=n_mels, fmax=max_frequency)
        melspec_preemph_dB = librosa.power_to_db(melspec_preemph, ref=np.max)
        melspec_preemph_dB_tensor = deltas(melspec_preemph_dB)

        # save npy
        np.save(str(output_path / f'melspec_dB_{n_mels}_preemph.npy'), melspec_preemph_dB_tensor)

        # save figures
        if save_fig:
            save_melspec_specshow(melspec_dB_tensor[...,0], str(prefix_fig / f'melspec_dB_{n_mels}.png'), 'Log-Mel spectrogram', sr=sr)
            save_melspec_specshow(melspec_dB_tensor[...,1], str(prefix_fig / f'melspec_dB_{n_mels}_delta.png'), 'Log-Mel spectrogram - Delta', is_delta=True)
            save_melspec_specshow(melspec_dB_tensor[...,2], str(prefix_fig / f'melspec_dB_{n_mels}_delta2.png'), 'Log-Mel spectrogram - Delta2', is_delta=True)
            save_melspec_specshow(melspec_preemph_dB_tensor[...,0], str(prefix_fig / f'melspec_dB_{n_mels}_preemph.png'), 'Log-Mel spectrogram preemphasis', sr=sr)
            save_melspec_specshow(melspec_preemph_dB_tensor[...,1], str(prefix_fig / f'melspec_dB_{n_mels}_delta_preemph.png'), 'Log-Mel spectrogram - Delta preemphasis', is_delta=True)
            save_melspec_specshow(melspec_preemph_dB_tensor[...,2], str(prefix_fig / f'melspec_dB_{n_mels}_delta2_preemph.png'), 'Log-Mel spectrogram - Delta2 preemphasis', is_delta=True)


def save_mfcc_specshow(data: np.ndarray, output_path: PathType, title: str) -> None:
    """Save MFCC plot to file.

    Args:
        data (np.ndarray): MFCCs.
        output_path (PathType): path to the output file.
        title (str): title of figure.
    """
    plt.figure(figsize=(10,4))
    librosa.display.specshow(data, x_axis='time')
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_path)
    plt.close()


def save_melspec_specshow(data: np.ndarray,
                          output_path: PathType,
                          title: str,
                          sr: int = 44100, # 22050,
                          is_delta: bool = False) -> None:
    """Save mel-spectrogram, delta, deltadelta plots to a file.

    Args:
        data (np.ndarray): mel-spectrogram or delta or deltadelta.
        output_path (PathType): path to the output file.
        title (str): title of figure.
        sr (int, optional): sample rate. Defaults to 44100.
        is_delta (bool, optional): data is delta or deltadelta. Defaults to False.
    """
    plt.figure(figsize=(10,4))
    plt.title(title)

    if not is_delta:
        librosa.display.specshow(data, sr=sr, x_axis='time', y_axis='mel',  fmax=8000)
        plt.colorbar(format='%+2.0f dB')
    else:
        librosa.display.specshow(data, x_axis='time', y_axis='mel',  fmax=8000)
        plt.colorbar()

    plt.savefig(output_path)
    plt.close()