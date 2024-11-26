import numpy as np
import torch
from msclap import CLAP
import torchaudio.transforms as T
from exordium.utils.decorator import timer_with_return


class ClapWrapper(CLAP):

    def __init__(self, version='2023', use_cuda: bool = True) -> None:
        """CLAP wrapper class."""
        super().__init__(version=version, use_cuda=use_cuda)

    @timer_with_return
    def __call__(self, waveforms: np.ndarray | torch.Tensor) -> np.ndarray:
        """CLAP feature extraction.

        Args:
            waveform (np.ndarray | torch.Tensor): batch audio signal of shape (B,T).

        Returns:
            np.ndarray: feature of shape (B, C) == (B, 1024)
        """
        waveforms = torch.Tensor(waveforms) # (B,T)

        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(dim=0)

        if waveforms.ndim != 2:
            raise Exception(f'Expected shape (B,T) got instead {waveforms.shape}.')

        feature = self.get_audio_embeddings(waveforms)

        return feature # (B, C) == (B, 1024)

    def read_audio(self, audio_path, resample=True, sample_rate: int = 44100):
        """Randomly sample a segment of audio_duration from the clip or pad to match duration"""
        if isinstance(audio_path, str):
            return super().read_audio(audio_path=audio_path, resample=resample)

        audio_time_series = audio_path

        resample_rate = self.args.sampling_rate
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)

        return audio_time_series, resample_rate