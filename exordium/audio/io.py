"""Audio input/output utilities."""

import logging
import math
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)
"""Module-level logger."""


def save_audio(audio: torch.Tensor, output_path: Path | str, sr: int = 16000) -> None:
    """Save audio waveform to file using torchaudio.

    Args:
        audio: Audio tensor of shape (T,) or (C, T).
        output_path: Path where audio file will be saved.
        sr: Sample rate in Hz.

    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(output_path), audio, sr)
    logger.info(f"Audio is saved: {output_path}")


def load_audio(
    audio_path: Path | str,
    target_sample_rate: int | None = 16000,
    clamp: bool = True,
    mono: bool = True,
    squeeze: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load audio file with optional resampling and preprocessing.

    Args:
        audio_path: Path to audio file.
        target_sample_rate: Target sample rate in Hz. If None, keeps original sample rate.
        clamp: If True, clamp waveform values to [-1, 1].
        mono: If True and audio is stereo, convert to mono by averaging channels.
        squeeze: If True, remove singleton dimensions from output.

    Returns:
        Tuple of (waveform, sample_rate):
            - waveform: Audio tensor of shape (T,) if mono and squeezed,
                       (C, T) if stereo or not squeezed.
            - sample_rate: Sample rate in Hz.

    """
    waveform, sample_rate = torchaudio.load(audio_path)  # (C, T)

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        resample = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resample(waveform)
        sample_rate = target_sample_rate

    if clamp:
        waveform = torch.clamp(waveform, -1, 1)

    if mono and waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0)

    if squeeze:
        waveform = torch.squeeze(waveform)

    return waveform, sample_rate


def split_audio(
    audio: torch.Tensor, segment_duration: float, sample_rate: int
) -> list[torch.Tensor]:
    """Split audio into fixed-duration segments.

    Args:
        audio: Audio tensor of shape (T,) or (C, T).
        segment_duration: Duration of each segment in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        List of audio segments. Last segment may be shorter than segment_duration.

    """
    segment_length = int(segment_duration * sample_rate)
    num_segments = int(math.ceil(audio.shape[-1] / segment_length))

    segments = [
        audio[..., i * segment_length : (i + 1) * segment_length] for i in range(num_segments)
    ]

    return segments


class AudioLoader:
    """Audio loader with caching support for efficient repeated access.

    Caches loaded audio files to avoid redundant I/O operations.
    """

    def __init__(self):
        """Initialize AudioLoader with empty cache."""
        self.cache = {}

    def load_audio(
        self,
        audio_file: Path | str,
        start_time_sec: float | None = None,
        end_time_sec: float | None = None,
        target_sample_rate: int | None = None,
        mono: bool = True,
        squeeze: bool = True,
        batch_dim: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Load audio file with caching, optional time slicing, and format support.

        Args:
            audio_file: Path to audio or video file with audio.
            start_time_sec: Start time in seconds for audio slice. Requires end_time_sec.
            end_time_sec: End time in seconds for audio slice. Requires start_time_sec.
            target_sample_rate: Target sample rate in Hz. Required for .mp4 files.
            mono: If True, convert to mono.
            squeeze: If True, remove singleton dimensions.
            batch_dim: If True, add batch dimension at index 0.

        Returns:
            Tuple of (waveform, sample_rate):
                - waveform: Audio tensor.
                - sample_rate: Sample rate in Hz.

        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if audio_file in self.cache:
            waveform, sr = self.cache[audio_file]
        else:
            waveform, sr = load_audio(
                audio_file, target_sample_rate=target_sample_rate, mono=mono, squeeze=squeeze
            )
            self.cache[audio_file] = (waveform, sr)

        if start_time_sec is not None and end_time_sec is not None:
            start_sample = int(start_time_sec * sr)
            end_sample = int(end_time_sec * sr)

            if start_sample < 0 or end_sample > waveform.shape[-1]:
                duration = waveform.shape[-1] / sr
                raise ValueError(
                    f"Start or end time is out of bounds. \
                        {start_time_sec} sec to {end_time_sec} sec exceeds \
                            {duration:.2f} second audio duration."
                )

            if end_time_sec < start_time_sec:
                raise ValueError(
                    f"End time ({end_time_sec} sec) is not greater \
                        than the start time ({start_time_sec} sec)."
                )

            waveform = waveform[..., start_sample:end_sample]

        if batch_dim:
            waveform = torch.unsqueeze(waveform, dim=0)

        return waveform, sr
