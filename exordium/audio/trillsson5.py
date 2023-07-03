import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np

class Trillsson5Wrapper():

    def __init__(self) -> None:
        self.sampling_rate = 16000
        self.model = hub.KerasLayer('https://tfhub.dev/google/trillsson5/1')
        
    def load_audio(self, path: str):
        # Audio should be floats in [-1, 1], sampled at 16kHz.
        audio, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return audio

    def extract_trillsson5(self, audio: np.ndarray, length_sec: int = 2):
        # Model input is of the shape [batch size, time].
        assert audio.ndim == 1
        # 2 sec chunks
        length = length_sec * self.sampling_rate
        num_chunks = len(audio) // length
        embeddings = []
        for chunk in range(num_chunks):
            audio_chunk = audio[chunk*length:(chunk+1)*length]
            audio_chunk = np.expand_dims(audio_chunk, axis=0)
            audio_chunk = tf.constant(audio_chunk)
            embed = self.model(audio_chunk)['embedding'].numpy()
            embeddings.append(embed)
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


if __name__ == '__main__':
    m = Trillsson5Wrapper()
    a = m.load_audio('data/processed/audio/9KAqOrdiZ4I.001.wav')
    e = m.extract_trillsson5(a)
    print(e.shape)