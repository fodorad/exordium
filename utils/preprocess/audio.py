import os
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp

import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import matplotlib.pyplot as plt

from pyAudioAnalysis import MidTermFeatures
from utils.preprocess.shared import parallel_eval


def audio2pyaa(input_path: str, output_path: str) -> None:
    """Extract short and mid term features with pyAudioAnalysis, then save

    Args:
        input_path (str): audio path
        output_path (str): pyaa feature path
    """
    output_path = Path(output_path).resolve()
    if (output_path.parent / (output_path.name + '_mt.csv')).exists(): return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    MidTermFeatures.mid_feature_extraction_to_file(file_path=input_path,
        mid_window=1.0, mid_step=1.0, short_window=0.05, short_step=0.05, output_file=str(output_path),
        store_short_features=True, store_csv=True, plot=False)


def audio2egemaps(input_path: str, lld_path: str, egemaps_path: str, config: str) -> None:
    """Extract eGeMAPS from audio, then save

    Args:
        input_path (str): audio path
        lld_path (str): lld path
        egemaps_path (str): summary path
        config (str): config path
    """
    egemaps_path = Path(egemaps_path).resolve()
    if egemaps_path.exists(): return
    egemaps_path.parent.mkdir(parents=True, exist_ok=True)
    CMD = f'SMILExtract -C {config} -I {input_path} -O {lld_path} -S {egemaps_path} -N {Path(input_path).stem} -l 0'
    os.system(CMD)


def video2audio(input_path: str, output_path: str, sr: int) -> None:
    """Extract audio from video and save

    Args:
        input_path (str): video path
        output_path (str): output path
        sr (int): sampling rate
    """
    output_path = Path(output_path).resolve()
    if output_path.exists(): return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    CMD = f'ffmpeg -loglevel 0 -i {str(input_path)} -nostdin -ab 320k -ac 1 -ar {sr} -c:a pcm_s16le -vn -hide_banner {str(output_path)}'
    os.system(CMD)


def audio2segments(input_path: str, output_dir: str, sr: int, n_segments: int) -> None:
    """Split audio file to N segments

    Args:
        input_path (str): input audio
        output_dir (str): output dir
        sr (int): sampling rate
        n_segments (int): number of segments
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(str(input_path), frame_rate=sr, format="wav")
    max_length = len(audio) # ms
    segment_length = max_length / n_segments
    steps = [int(np.floor(segment_length * i)) for i in range(n_segments)] + [max_length]
    for i in range(len(steps)-1):
        output_path = output_dir / (input_path.stem + '_{:02d}'.format(i) + '.wav')
        if output_path.exists(): return
        segment = audio[steps[i]:steps[i+1]]
        segment.export(str(output_path), format="wav")


def deltas(feature: np.ndarray) -> np.ndarray:
    """Calculate differential (delta) and acceleration (deltadelta) features 
    from MFCC or spectrograms

    Args:
        feature (np.ndarray): MFCC or spectrogram.

    Returns:
        np.ndarray: tensor of feature, deltas and deltadeltas
    """
    differential = librosa.feature.delta(feature)
    acceleration = librosa.feature.delta(feature, order=2)
    feature_tensor = np.stack([feature, differential, acceleration], axis=2)
    return feature_tensor


def audio2logmelspec(input_path, output_path, sr) -> None:
    """Creates the MFCCs, log mel-spectrogram, delta and deltadelta features
    from audio files

    Args:
        input_path (str): audio path
        output_path (str): output path
        sr (int): sampling rate
    """
    n_fft = 2048
    hop_length = 512
    n_mfcc = 40
    max_frequency = 8000
    if Path(output_path).exists(): return
    Path(output_path).mkdir(parents=True, exist_ok=True)
    prefix_fig = Path(output_path) / 'figures'
    prefix_fig.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(input_path, sr=sr) # sr=22050

    # preemphasis
    y_preemph = librosa.effects.preemphasis(y, coef=0.97, zi=[y[1]])

    # mfcc with differential and acceleration
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_preemph = librosa.feature.mfcc(y=y_preemph, sr=sr, n_mfcc=n_mfcc)
    mfcc_tensor = deltas(mfcc)
    mfcc_preemph_tensor = deltas(mfcc_preemph)

    # save npy
    np.save(str(Path(output_path) / 'mfcc.npy'), mfcc_tensor)
    np.save(str(Path(output_path) / 'mfcc_preemph.npy'), mfcc_preemph_tensor)

    # save figures
    save_mfcc_specshow(mfcc_tensor[...,0], str(prefix_fig / 'mfcc.png'), 'MFCC')
    save_mfcc_specshow(mfcc_tensor[...,1], str(prefix_fig / 'mfcc_delta.png'), 'MFCC-Delta')
    save_mfcc_specshow(mfcc_tensor[...,2], str(prefix_fig / 'mfcc_delta2.png'), 'MFCC-Delta2')
    save_mfcc_specshow(mfcc_preemph_tensor[...,0], str(prefix_fig / 'mfcc_preemph.png'), 'MFCC preemphasis')
    save_mfcc_specshow(mfcc_preemph_tensor[...,1], str(prefix_fig / 'mfcc_delta_preemph.png'), 'MFCC-Delta preemphasis')
    save_mfcc_specshow(mfcc_preemph_tensor[...,2], str(prefix_fig / 'mfcc_delta2_preemph.png'), 'MFCC-Delta2 preemphasis')

    # melspec db
    for n_mels in [80, 128, 224]:
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=max_frequency)
        melspec_dB = librosa.power_to_db(melspec, ref=np.max)
        melspec_dB_tensor = deltas(melspec_dB)

        melspec_preemph = librosa.feature.melspectrogram(y=y_preemph, sr=sr, n_mels=n_mels, fmax=max_frequency)
        melspec_preemph_dB = librosa.power_to_db(melspec_preemph, ref=np.max)
        melspec_preemph_dB_tensor = deltas(melspec_preemph_dB)

        # save npy
        np.save(str(Path(output_path) / f'melspec_dB_{n_mels}.npy'), melspec_dB_tensor)
        np.save(str(Path(output_path) / f'melspec_dB_{n_mels}_preemph.npy'), melspec_preemph_dB_tensor)

        # save figures
        save_melspec_specshow(melspec_dB_tensor[...,0], str(prefix_fig / f'melspec_dB_{n_mels}.png'), 'Log-Mel spectrogram', sr=sr)
        save_melspec_specshow(melspec_dB_tensor[...,1], str(prefix_fig / f'melspec_dB_{n_mels}_delta.png'), 'Log-Mel spectrogram - Delta', is_delta=True)
        save_melspec_specshow(melspec_dB_tensor[...,2], str(prefix_fig / f'melspec_dB_{n_mels}_delta2.png'), 'Log-Mel spectrogram - Delta2', is_delta=True)
        save_melspec_specshow(melspec_preemph_dB_tensor[...,0], str(prefix_fig / f'melspec_dB_{n_mels}_preemph.png'), 'Log-Mel spectrogram preemphasis', sr=sr)
        save_melspec_specshow(melspec_preemph_dB_tensor[...,1], str(prefix_fig / f'melspec_dB_{n_mels}_delta_preemph.png'), 'Log-Mel spectrogram - Delta preemphasis', is_delta=True)
        save_melspec_specshow(melspec_preemph_dB_tensor[...,2], str(prefix_fig / f'melspec_dB_{n_mels}_delta2_preemph.png'), 'Log-Mel spectrogram - Delta2 preemphasis', is_delta=True)


def save_mfcc_specshow(data: np.ndarray, output_path: str, title: str) -> None:
    """Save MFCC plot to disk

    Args:
        data (np.ndarray): MFCCs
        output_path (str): output path
        title (str): title of figure
    """
    plt.figure(figsize=(10,4))
    librosa.display.specshow(data, x_axis='time')
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_path)
    plt.close()


def save_melspec_specshow(data: np.ndarray, 
                          output_path: str, 
                          title: str, 
                          sr: int = 44100, # 22050, 
                          is_delta: bool = False) -> None:
    """Save mel-spectrogram, delta, deltadelta plots to disk

    Args:
        data (np.ndarray): mel-spectrogram or delta or deltadelta
        output_path (str): output path
        title (str): title of figure
        sr (int, optional): sampling rate. Defaults to 44100.
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

