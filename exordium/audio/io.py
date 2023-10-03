import os
import logging
from pathlib import Path
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
from decord import AudioReader, cpu
from exordium import PathType


def video2audio(input_path: PathType, output_path: PathType, sr: int = 44100, overwrite: bool = False) -> None:
    """Extract audio from video.

    Args:
        input_path (PathType): path to the video file.
        output_path (PathType): path to the output audio file.
        sr (int, optional): sample rate. Defaults to 44100.
        overwrite (bool, optional): overwrites the output file if it exists. Defaults to False.
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    CMD = f'ffmpeg -loglevel 0 -i {str(input_path)} -nostdin -ab 320k -ac 1 -ar {sr} -c:a pcm_s16le -vn -hide_banner {str(output_path)}'
    logging.info(CMD)
    os.system(CMD)


def audio2segments(input_path: PathType, output_dir: PathType, n_segments: int) -> None:
    """Split audio file to N segments.

    Args:
        input_path (PathType): path to the audio file.
        output_dir (PathType): path to the output directory.
        n_segments (int): number of segments.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate, audio_data = wavfile.read(str(input_path))
    max_length_samples = len(audio_data)
    samples_per_segment = max_length_samples // n_segments

    for i in range(n_segments):
        start_sample = i * samples_per_segment
        end_sample = (i + 1) * samples_per_segment
        segment = audio_data[start_sample:end_sample]

        output_path = output_dir / f'{input_path.stem}_{i:02d}.wav'

        if output_path.exists():
            continue

        wavfile.write(str(output_path), sample_rate, segment)


def load_audio(audio_path: PathType, sr: int | None = 16000) -> tuple[torch.Tensor, int]:
    """Loads audio file at a given sample rate and converts to mono.

    Args:
        audio_path (PathType): path to the audio file.
        sr (int | None, optional): sample rate. None means that the original sample rate is used. Defaults to 16000.

    Returns:
        tuple[torch.Tensor, int]: audio signal as a torch.Tensor and sample rate
    """
    waveform, sample_rate = torchaudio.load(audio_path) # (C, T)
    sr = sr or sample_rate

    if sample_rate != sr:
        resample = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resample(waveform)

    audio = torch.clamp(waveform, -1, 1)

    if audio.shape[0] == 2:
        audio = audio.mean(dim=0) # (C, T) -> (1, T)

    audio = torch.squeeze(audio) # (1, T) -> (T,)
    return audio, sr


def load_audio_from_video(video_path: PathType, sr: int | None = 16000) -> np.ndarray:
    """Loads audio from a video file at a given sample rate and converts to mono.

    Args:
        video_path (PathType): path to the video file.
        sr (int | None, optional): sample rate. None means that the original sample rate is used. Defaults to 16000.

    Returns:
        np.ndarray: audio signal as a numpy vector of shape (T,)
    """
    ar = AudioReader(str(video_path), ctx=cpu(0), sample_rate=sr or -1, mono=True) # (C, T) == (1, T)
    audio = ar[:].asnumpy().squeeze() # (1, T) -> (T,)
    logging.info(f'Loaded audio shape: {ar.shape}, sample_rate: {sr}, length: {round(ar.shape[1]/sr, ndigits=2)} sec')
    return audio


def save_audio(audio: np.ndarray, output_path: PathType, sr: int = 16000) -> None:
    """Saves audio signal to a file.

    Args:
        audio (np.ndarray): singal of shape (T,).
        output_path (PathType): path to the output file.
        sr (int, optional): sample rate. Defaults to 16000.
    """
    wavfile.write(str(output_path), sr, audio)
    logging.info(f'Audio is saved: {output_path}')