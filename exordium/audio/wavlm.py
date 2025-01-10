import torchaudio
import torch
import numpy as np
import torchaudio
import torch.nn.functional as F
from exordium import PathType
from exordium.utils.decorator import load_or_create, timer_with_return

from exordium.utils import decorator
decorator.TIMING_ENABLED = True


class WavlmWrapper():


    def __init__(self, gpu_id: int = 0, model_name: str = 'base+') -> None:
        """WavLM wrapper class."""
        self.device = torch.device(f'cuda:{gpu_id}') if gpu_id >= 0 else torch.device('cpu')

        if model_name not in ('base', 'base+', 'large'):
            raise ValueError('Invalid model_name')

        if model_name == 'base':
            self.bundle = torchaudio.pipelines.WAVLM_BASE
        elif model_name == 'base+':
            self.bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
        else: # large
            self.bundle = torchaudio.pipelines.WAVLM_LARGE

        self.model = self.bundle.get_model()
        self.model.eval()
        self.model.to(self.device)


    @load_or_create('pkl')
    def audio_to_feature(self, audio_path: PathType, **kwargs) -> list[np.ndarray]:

        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        if sample_rate != self.bundle.sample_rate: # 16000 for WavLM
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)

        # expects a single channel (mono) audio, take the first channel if stereo
        if waveform.ndim == 2: # (2, T)
            waveform = waveform[0, :]  # Use the first channel, (T,)

        features = self(waveform)

        features = [feature.detach().cpu().numpy().squeeze(0) for feature in features]
        return features


    @timer_with_return
    def __call__(self, waveform: np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        """Audio SSL feature extraction.

        Args:
            waveform (np.ndarray | torch.Tensor): audio signal of shape (T,).

        Returns:
            list(torch.Tensor): list of layer features of shape (B, T, C) == (1, T, 768)
        """
        waveform = torch.Tensor(waveform) # (T,)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(dim=0) # (B, T)

        if waveform.ndim != 2:
            raise Exception(f'Expected shape (B, T) got instead {waveform.shape}.')

        waveform = waveform.to(self.device)

        with torch.no_grad():
            features, _ = self.model.extract_features(waveform)

        return features # list[(B, T, C),...]; C==768


def pad_wavlm_time_dim(tensor, target_time_dim, pad_value=0):
    """
    Pads the time dimension of a tensor and generates a mask.

    Args:
        tensor (torch.Tensor): Input tensor of shape (L, N, F).
        target_time_dim (int): Desired size for the time dimension (N).
        pad_value (int or float): Value to use for padding (default is 0).

    Returns:
        tuple: 
            - torch.Tensor: Padded tensor with shape (L, target_time_dim, F).
            - torch.BoolTensor: Mask tensor of shape (target_time_dim,) with `1` for original data and `0` for padding.
    """
    _, N, _ = tensor.shape

    # Create the mask
    mask = torch.ones(N, dtype=torch.bool)
    if N < target_time_dim:
        # Extend the mask for padding
        mask = torch.cat([mask, torch.zeros(target_time_dim - N, dtype=torch.bool)], dim=0)

    if N >= target_time_dim:
        # No padding needed, crop to target_time_dim
        return tensor[:, :target_time_dim, :], mask[:target_time_dim]
    
    # Calculate padding
    pad_amount = target_time_dim - N
    padded_tensor = F.pad(tensor, (0, 0, 0, pad_amount, 0, 0), mode='constant', value=pad_value)
    return padded_tensor, mask


if __name__ == "__main__":
    audio_file = "data/sounds/example_multispeaker.wav"
    wavlm = WavlmWrapper()
    features = wavlm.audio_to_feature(audio_file)
    print(len(features), features[0].shape)