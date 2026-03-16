import unittest

import numpy as np
import torch
import torchaudio

from exordium.audio.base import AudioModelWrapper
from tests.fixtures import AUDIO_MULTISPEAKER


class _ConcreteWrapper(AudioModelWrapper):
    """Minimal concrete subclass used only for testing the base class."""

    def audio_to_feature(self, audio, **kwargs):
        return self._prepare_waveform(audio, 16000)

    def batch_audio_to_features(self, audios, **kwargs):
        return [self._prepare_waveform(a, 16000) for a in audios]

    def inference(self, waveform):
        return self._prepare_waveform(waveform, 16000)


class TestAudioModelWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = _ConcreteWrapper(device_id=-1)
        cls.audio_path = AUDIO_MULTISPEAKER

    # --- ABC enforcement ---

    def test_cannot_instantiate_abstract_class(self):
        with self.assertRaises(TypeError):
            AudioModelWrapper()  # type: ignore[abstract]

    def test_abstract_methods_defined(self):
        self.assertIn("audio_to_feature", AudioModelWrapper.__abstractmethods__)
        self.assertIn("batch_audio_to_features", AudioModelWrapper.__abstractmethods__)
        self.assertIn("inference", AudioModelWrapper.__abstractmethods__)

    # --- Initialization ---

    def test_device_is_cpu_for_minus_one(self):
        self.assertEqual(str(_ConcreteWrapper(device_id=-1).device), "cpu")

    def test_default_device_id_is_cpu(self):
        self.assertEqual(str(_ConcreteWrapper().device), "cpu")

    def test_device_attribute_is_torch_device(self):
        self.assertIsInstance(self.wrapper.device, torch.device)

    # --- _prepare_waveform: file path ---

    def test_path_returns_1d_tensor(self):
        waveform = self.wrapper._prepare_waveform(self.audio_path, 16000)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(waveform.ndim, 1)

    def test_str_path_works(self):
        waveform = self.wrapper._prepare_waveform(str(self.audio_path), 16000)
        self.assertEqual(waveform.ndim, 1)

    def test_path_resamples_to_target_rate(self):
        w16k = self.wrapper._prepare_waveform(self.audio_path, 16000)
        w8k = self.wrapper._prepare_waveform(self.audio_path, 8000)
        self.assertGreater(len(w16k), len(w8k))

    def test_path_positive_length(self):
        self.assertGreater(len(self.wrapper._prepare_waveform(self.audio_path, 16000)), 0)

    # --- _prepare_waveform: numpy array ---

    def test_numpy_1d_returns_tensor(self):
        arr = np.random.rand(16000).astype(np.float32)
        waveform = self.wrapper._prepare_waveform(arr, 16000)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(waveform.ndim, 1)

    def test_numpy_2d_uses_first_channel(self):
        arr = np.random.rand(2, 16000).astype(np.float32)
        waveform = self.wrapper._prepare_waveform(arr, 16000)
        self.assertEqual(waveform.ndim, 1)
        self.assertTrue(torch.allclose(waveform, torch.as_tensor(arr[0])))

    def test_numpy_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            self.wrapper._prepare_waveform(np.random.rand(1, 2, 16000), 16000)

    # --- _prepare_waveform: torch tensor ---

    def test_tensor_1d_passthrough(self):
        t = torch.rand(16000)
        waveform = self.wrapper._prepare_waveform(t, 16000)
        self.assertEqual(waveform.ndim, 1)
        self.assertTrue(torch.equal(waveform, t))

    def test_tensor_2d_uses_first_channel(self):
        t = torch.rand(2, 16000)
        waveform = self.wrapper._prepare_waveform(t, 16000)
        self.assertTrue(torch.equal(waveform, t[0]))

    def test_tensor_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            self.wrapper._prepare_waveform(torch.rand(1, 2, 16000), 16000)

    # --- path vs waveform parity ---

    def test_path_and_tensor_same_length(self):
        _, native_sr = torchaudio.load(self.audio_path)
        from_path = self.wrapper._prepare_waveform(self.audio_path, native_sr)
        native_waveform, _ = torchaudio.load(self.audio_path)
        from_tensor = self.wrapper._prepare_waveform(native_waveform, native_sr)
        self.assertEqual(len(from_path), len(from_tensor))

    # --- _pad_waveforms ---

    def test_pad_waveforms_output_shape(self):
        waveforms = [torch.rand(100), torch.rand(200), torch.rand(150)]
        padded, lengths = AudioModelWrapper._pad_waveforms(waveforms)
        self.assertEqual(padded.shape, (3, 200))

    def test_pad_waveforms_lengths(self):
        waveforms = [torch.rand(100), torch.rand(200), torch.rand(150)]
        _, lengths = AudioModelWrapper._pad_waveforms(waveforms)
        self.assertEqual(lengths, [100, 200, 150])

    def test_pad_waveforms_data_preserved(self):
        w1 = torch.rand(100)
        w2 = torch.rand(200)
        padded, _ = AudioModelWrapper._pad_waveforms([w1, w2])
        self.assertTrue(torch.equal(padded[0, :100], w1))
        self.assertTrue(torch.equal(padded[1, :200], w2))

    def test_pad_waveforms_padding_is_zero(self):
        w1 = torch.rand(100)
        w2 = torch.rand(200)
        padded, _ = AudioModelWrapper._pad_waveforms([w1, w2])
        self.assertTrue(torch.all(padded[0, 100:] == 0))

    def test_pad_waveforms_single_element(self):
        w = torch.rand(50)
        padded, lengths = AudioModelWrapper._pad_waveforms([w])
        self.assertEqual(padded.shape, (1, 50))
        self.assertEqual(lengths, [50])

    # --- batch_audio_to_features ---

    def test_batch_audio_to_features_returns_list(self):
        waveforms = [np.random.rand(16000).astype(np.float32) for _ in range(3)]
        results = self.wrapper.batch_audio_to_features(waveforms)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

    def test_batch_audio_to_features_from_paths(self):
        results = self.wrapper.batch_audio_to_features([self.audio_path, self.audio_path])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
