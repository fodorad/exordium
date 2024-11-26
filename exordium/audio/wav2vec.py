import torch
import numpy as np
import torchaudio
import transformers as tfm
from exordium import PathType
from exordium.utils.decorator import load_or_create


class Wav2vec2Wrapper():

    def __init__(self) -> None:
        """Wav2Vec2 wrapper class."""
        self.preprocessor = tfm.Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = tfm.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval()

    @load_or_create('npy')
    def audio_to_feature(self, audio_path: PathType, **kwargs):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Resample to 16 kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Since Wav2Vec2 expects a single channel (mono) audio, take the first channel if stereo
        if waveform.ndim == 2: # (2, T)
            waveform = waveform[0, :]  # Use the first channel, (T,)
        
        feature = self(waveform)

        return feature

    def __call__(self, waveform: np.ndarray | torch.Tensor) -> np.ndarray:
        """Audio SSL feature extraction.

        Args:
            waveform (np.ndarray | torch.Tensor): audio signal of shape (T,).

        Returns:
            np.ndarray: feature of shape (B, T, C) == (1, T, 768)
        """
        waveform = torch.Tensor(waveform) # (T,)

        if waveform.ndim != 1:
            raise Exception(f'Expected shape (T,) got instead {waveform.shape}.')

        input_values = self.preprocessor(waveform, return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            feature = self.model(input_values).last_hidden_state

        feature = feature.detach().cpu().numpy().squeeze(0)
        return feature # (T, C) == (T, 768)


if __name__ == "__main__":
    audio_file = "data/sounds/example_multispeaker.wav"
    wav2vec = Wav2vec2Wrapper()
    feature = wav2vec.audio_to_feature(audio_file)
    print(feature.shape)