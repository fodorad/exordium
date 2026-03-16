import unittest

import torch

from exordium.audio.io import load_audio
from exordium.text.whisper import WhisperWrapper
from tests.fixtures import AUDIO_MULTISPEAKER


class WhisperTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = WhisperWrapper(model_name="tiny")
        cls.waveform, cls.sr = load_audio(
            AUDIO_MULTISPEAKER, target_sample_rate=16000, mono=True, squeeze=True
        )

    # --- Initialization ---

    def test_model_loaded(self):
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model_name, "tiny")

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            WhisperWrapper(model_name="nonexistent_model")

    def test_available_models(self):
        models = WhisperWrapper.available_models()
        self.assertIsInstance(models, list)
        self.assertIn("tiny", models)
        self.assertIn("turbo", models)

    # --- __call__ with waveform ---

    def test_call_torch_tensor(self):
        text = self.model(self.waveform, language="en", beam_size=1)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_call_numpy_array(self):
        audio_np = self.waveform.numpy()
        text = self.model(audio_np, language="en", beam_size=1)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_call_silence(self):
        silence = torch.zeros(16000)
        text = self.model(silence, language="en", beam_size=1)
        self.assertIsInstance(text, str)

    def test_call_short_audio(self):
        short = self.waveform[:8000]  # 0.5s
        text = self.model(short, language="en", beam_size=1)
        self.assertIsInstance(text, str)

    def test_torch_and_numpy_same_result(self):
        chunk = self.waveform[16000:48000]
        text_pt = self.model(chunk, language="en", beam_size=1)
        text_np = self.model(chunk.numpy(), language="en", beam_size=1)
        self.assertEqual(text_pt, text_np)

    # --- transcribe_file ---

    def test_transcribe_file(self):
        text = self.model.transcribe_file(AUDIO_MULTISPEAKER, language="en", beam_size=1)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 50)

    def test_transcribe_file_matches_call(self):
        text_file = self.model.transcribe_file(AUDIO_MULTISPEAKER, language="en", beam_size=1)
        text_call = self.model(self.waveform, language="en", beam_size=1)
        self.assertEqual(text_file, text_call)

    # --- Content sanity ---

    def test_transcription_contains_expected_words(self):
        text = self.model(self.waveform, language="en", beam_size=1).lower()
        self.assertTrue(
            any(word in text for word in ["lunch", "burger", "food", "eat"]),
            f"Transcription should contain food-related words: {text[:200]}",
        )


if __name__ == "__main__":
    unittest.main()
