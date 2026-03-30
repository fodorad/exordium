"""Tests for BlinkDenseNet121Wrapper — static utilities and full inference."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.face.blink.densenet import BlinkDenseNet121Wrapper


def _make_landmarks_6(B, left_x=80, left_y=112, right_x=144, right_y=112):
    """Return (B, 6, 2) float32 landmarks."""
    lmks = np.zeros((B, 6, 2), dtype=np.float32)
    for i in range(B):
        lmks[i, 0] = [left_x, left_y]
        lmks[i, 1] = [right_x, right_y]
        lmks[i, 2] = [112, 140]
        lmks[i, 3] = [112, 160]
        lmks[i, 4] = [40, 112]
        lmks[i, 5] = [184, 112]
    return lmks


class TestBlinkDenseNetStaticMethods(unittest.TestCase):
    """Static method tests — no model weights required."""

    def test_eyes_open_both_open(self):
        left_state = torch.tensor([0.1, 0.9, 0.1, 0.9])
        right_state = torch.tensor([0.1, 0.9, 0.9, 0.1])
        left_valid = torch.ones(4, dtype=torch.bool)
        right_valid = torch.ones(4, dtype=torch.bool)
        result = BlinkDenseNet121Wrapper.eyes_open(left_state, right_state, left_valid, right_valid)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (4,))
        self.assertTrue(result[0])
        self.assertFalse(result[1])

    def test_visualize_numpy_in_numpy_out(self):
        B = 2
        frames = np.random.randint(0, 255, (B, 112, 112, 3), dtype=np.uint8)
        result = BlinkDenseNet121Wrapper.visualize(
            frames,
            np.array([0.3, 0.7]),
            np.array([0.2, 0.8]),
            np.ones(B, dtype=bool),
            np.ones(B, dtype=bool),
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], B)

    def test_visualize_tensor_input_returns_tensor(self):
        B = 2
        frames_t = torch.randint(0, 255, (B, 3, 112, 112), dtype=torch.uint8)
        result = BlinkDenseNet121Wrapper.visualize(
            frames_t,
            np.array([0.3, 0.7]),
            np.array([0.2, 0.8]),
            np.ones(B, dtype=bool),
            np.ones(B, dtype=bool),
        )
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], B)

    def test_visualize_with_landmarks(self):
        B = 2
        frames = np.random.randint(0, 255, (B, 112, 112, 3), dtype=np.uint8)
        lmks = np.zeros((B, 6, 2), dtype=np.float32)
        lmks[:, 0] = [30, 30]
        lmks[:, 1] = [80, 30]
        result = BlinkDenseNet121Wrapper.visualize(
            frames,
            np.array([0.3, 0.7]),
            np.array([0.2, 0.8]),
            np.ones(B, dtype=bool),
            np.ones(B, dtype=bool),
            landmarks=lmks,
        )
        self.assertIsInstance(result, np.ndarray)

    def test_visualize_with_invalid_eye_skips_text(self):
        B = 2
        frames = np.random.randint(0, 255, (B, 112, 112, 3), dtype=np.uint8)
        result = BlinkDenseNet121Wrapper.visualize(
            frames,
            np.array([0.3, 0.7]),
            np.array([0.2, 0.8]),
            np.array([False, True], dtype=bool),
            np.array([True, False], dtype=bool),
        )
        self.assertEqual(result.shape[0], B)

    def test_visualize_with_output_path(self):
        B = 2
        frames = np.random.randint(0, 255, (B, 112, 112, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "blink_vis.png"
            BlinkDenseNet121Wrapper.visualize(
                frames,
                np.array([0.3, 0.7]),
                np.array([0.2, 0.8]),
                np.ones(B, dtype=bool),
                np.ones(B, dtype=bool),
                output_path=str(out),
            )
            self.assertGreater(len(list(Path(d).glob("*.png"))), 0)


class TestBlinkDenseNet121WrapperModel(unittest.TestCase):
    """Inference tests — loads the model once for the entire suite."""

    @classmethod
    def setUpClass(cls):
        cls.model = BlinkDenseNet121Wrapper(device_id=None)

    def test_call_single_returns_prob(self):
        probs = self.model(torch.rand(1, 3, 64, 64))
        self.assertIsInstance(probs, torch.Tensor)
        self.assertEqual(probs.shape, (1,))

    def test_probs_in_range(self):
        probs = self.model(torch.rand(1, 3, 64, 64))
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_batch_shape(self):
        probs = self.model(torch.rand(4, 3, 64, 64))
        self.assertEqual(probs.shape, (4,))

    def test_predict_pipeline_returns_4_tensors(self):
        B = 4
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        result = self.model.predict_pipeline(frames, _make_landmarks_6(B))
        self.assertEqual(len(result), 4)
        for arr in result:
            self.assertIsInstance(arr, torch.Tensor)
            self.assertEqual(arr.shape, (B,))

    def test_predict_pipeline_probs_in_range(self):
        B = 2
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        left_state, right_state, _, _ = self.model.predict_pipeline(frames, _make_landmarks_6(B))
        for arr in [left_state, right_state]:
            self.assertTrue((arr >= 0.0).all())
            self.assertTrue((arr <= 1.0).all())

    def test_predict_pipeline_tensor_input(self):
        B = 2
        frames_t = torch.randint(0, 255, (B, 3, 224, 224), dtype=torch.uint8)
        result = self.model.predict_pipeline(frames_t, _make_landmarks_6(B))
        self.assertEqual(len(result), 4)
        for t in result:
            self.assertEqual(t.shape[0], B)

    def test_predict_pipeline_with_headpose(self):
        B = 2
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        headpose = np.array([[60.0, 0.0, 0.0], [-60.0, 0.0, 0.0]], dtype=np.float32)
        _, _, left_valid, right_valid = self.model.predict_pipeline(
            frames, _make_landmarks_6(B), headpose
        )
        self.assertFalse(left_valid[0])
        self.assertFalse(right_valid[1])

    def test_predict_pipeline_small_eye_distance_marks_invalid(self):
        """Eye coords near each other → invalid detection branch."""
        B = 1
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        lmks = _make_landmarks_6(B, left_x=100, left_y=112, right_x=105, right_y=112)
        _, _, left_valid, right_valid = self.model.predict_pipeline(frames, lmks)
        self.assertFalse(left_valid[0])
        self.assertFalse(right_valid[0])

    def test_predict_pipeline_empty_crop_marks_invalid(self):
        """Eye coords outside frame bounds → empty crop branch."""
        B = 1
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        lmks = _make_landmarks_6(B, left_x=500, left_y=112, right_x=510, right_y=112)
        _, _, left_valid, right_valid = self.model.predict_pipeline(frames, lmks)
        self.assertFalse(left_valid[0])
        self.assertFalse(right_valid[0])

    def test_predict_frame_returns_4_values(self):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = self.model.predict_frame(frame, _make_landmarks_6(1)[0])
        self.assertEqual(len(result), 4)

    def test_predict_frame_tensor_input(self):
        frame_t = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        result = self.model.predict_frame(frame_t, _make_landmarks_6(1)[0])
        self.assertEqual(len(result), 4)

    def test_predict_frame_return_patches_true(self):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = self.model.predict_frame(frame, _make_landmarks_6(1)[0], return_patches=True)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[4].shape, (64, 64, 3))
        self.assertEqual(result[5].shape, (64, 64, 3))

    def test_predict_frame_with_headpose(self):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        headpose = np.array([60.0, 0.0, 0.0], dtype=np.float32)
        result = self.model.predict_frame(frame, _make_landmarks_6(1)[0], headpose=headpose)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
