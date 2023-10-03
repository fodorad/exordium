import torch
import numpy as np
import transformers as tfm


class Wav2vec2Wrapper():

    def __init__(self) -> None:
        """Wav2Vec2 wrapper class."""
        self.preprocessor = tfm.Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = tfm.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval()

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

        feature = feature.detach().cpu().numpy()
        return feature # (B, T, C) == (1, T, 768)