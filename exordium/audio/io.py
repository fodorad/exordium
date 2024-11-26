import os
import logging
from pathlib import Path
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
from decord import AudioReader, cpu
from exordium import PathType
from exordium.utils.decorator import timer_with_return


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


def split_audio(audio: np.ndarray, segment_duration: float, sample_rate: int) -> list:
    """
    Splits the audio into segments of specified duration.

    Args:
        audio (np.ndarray): Input audio array.
        segment_duration (float): Duration of each segment in seconds.
        sample_rate (int): Sample rate of the audio.

    Returns:
        list: List of audio segments.
    """
    segment_length = int(segment_duration * sample_rate)
    num_segments = int(np.ceil(len(audio) / segment_length))

    segments = [audio[i*segment_length:(i+1)*segment_length] for i in range(num_segments)]

    return segments


@timer_with_return
def load_audio(audio_path: PathType,
               sr: int | None = 16000,
               resample: bool = True,
               clamp: bool = True,
               mono: bool = True,
               squeeze: bool = True) -> tuple[torch.Tensor, int]:
    """Loads audio file at a given sample rate and converts to mono.

    Args:
        audio_path (PathType): path to the audio file.
        sr (int | None, optional): sample rate. None means that the original sample rate is used. Defaults to 16000.
        start_time_sec (float | None, optional): if given, the audio is extracted starting from this timestamp. Defaults to None.
        end_time_sec (float | None, optional): if given, the audio is extracted until this timestamp. Defaults to None.

    Note:
        Currently, start_time_sec and end_time_sec can only be used together.

    Returns:
        tuple[torch.Tensor, int]: audio signal as a torch.Tensor and sample rate
    """
    waveform, sample_rate = torchaudio.load(audio_path) # (C, T)

    sr = sr or sample_rate

    if resample and sample_rate != sr:
        resample = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resample(waveform)

    if clamp:
        audio = torch.clamp(waveform, -1, 1)

    if mono and audio.shape[0] == 2:
        audio = audio.mean(dim=0) # (C, T) -> (1, T)

    if squeeze:
        audio = torch.squeeze(audio) # (1, T) -> (T,)

    return audio, sr


def load_audio_from_video(video_path: PathType,
                          sr: int | None = 16000,
                          return_tensors: str = "pt",
                          mono: bool = True,
                          squeeze: bool = True) -> np.ndarray | torch.Tensor:
    """Loads audio from a video file at a given sample rate.

    Args:
        video_path (PathType): path to the video file.
        sr (int | None, optional): sample rate. None means that the original sample rate is used. Defaults to 16000.
        return_tensors (str, optional): output format. Supported values: "npy" and "pt". Default: torch.
        mono (bool, optional): convert to mono. Default True.
        squeeze (bool, optional): squeeze dimensions. Default True.

    Returns:
        np.ndarray | torch.Tensor: audio signal
    """
    ar = AudioReader(str(video_path), ctx=cpu(0), sample_rate=sr or -1, mono=mono) # (1, T) if mono is True, otherwise (C, T)
    audio = ar[:].asnumpy()

    if squeeze:
        audio = audio.squeeze() # (1, T) -> (T,)

    if return_tensors == "pt":
        audio = torch.Tensor(audio)

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


class AudioLoader:

    def __init__(self):
        self.cache = {}

    @timer_with_return
    def load_audio(self, audio_file: PathType,
                   start_time_sec: float | None = None,
                   end_time_sec: float | None = None,
                   sr: int | None = None,
                   mono: bool = True,
                   squeeze: bool = True,
                   batch_dim: bool = False,
                   return_tensor: str = 'pt'):

        if audio_file in self.cache:
            waveform, sr = self.cache[audio_file]
        else:
            # Load audio based on file format
            file_extension = Path(audio_file).suffix
            if file_extension in ['.mp3', '.wav']:
                waveform, sr = load_audio(audio_file, sr=sr, mono=mono, squeeze=squeeze)

            elif file_extension == '.mp4':

                if sr is None:
                    raise ValueError("Set the audio sampling rate if the input is a video.")

                waveform = torch.Tensor(load_audio_from_video(audio_file, sr=sr, mono=mono, squeeze=squeeze))
            else:
                raise ValueError(f"Unsupported audio format: {file_extension}")

            self.cache[audio_file] = (waveform, sr)

        if start_time_sec is not None and end_time_sec is not None:

            start_sample = int(start_time_sec * sr)
            end_sample = int(end_time_sec * sr)

            if start_sample < 0 or end_sample > waveform.shape[-1]:
                raise ValueError("Start or end time is out of bounds.")

            if end_time_sec < start_time_sec:
                raise ValueError(f"End time ({end_time_sec} sec) is not greater than the start time ({start_time_sec} sec).")

            waveform = waveform[...,start_sample:end_sample]

        if batch_dim:
            waveform = torch.unsqueeze(waveform, dim=0)

        if return_tensor == 'npy':
            waveform = np.array(waveform)

        return waveform, sr