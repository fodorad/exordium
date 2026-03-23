"""Tests for exordium.video.face.blink.densenet.BlinkDenseNet121Wrapper."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.face.blink.densenet import BlinkDenseNet121Wrapper


def _make_landmarks_478(B: int) -> np.ndarray:
    """Make (B, 478, 2) float32 landmarks."""
    landmarks = np.zeros((B, 478, 2), dtype=np.float32)
    for i in range(B):
        landmarks[i, 33] = [50, 100]
        landmarks[i, 133] = [90, 100]
        landmarks[i, 362] = [134, 100]
        landmarks[i, 263] = [174, 100]
    return landmarks


def _make_landmarks_6(B, left_x=80, left_y=112, right_x=144, right_y=112):
    """Make (B, 6, 2) float32 landmarks."""
    lmks = np.zeros((B, 6, 2), dtype=np.float32)
    for i in range(B):
        lmks[i, 0] = [left_x, left_y]
        lmks[i, 1] = [right_x, right_y]
        lmks[i, 2] = [112, 140]
        lmks[i, 3] = [112, 160]
        lmks[i, 4] = [40, 112]
        lmks[i, 5] = [184, 112]
    return lmks


class TestBlinkDenseNet121Wrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BlinkDenseNet121Wrapper(device_id=None)

    def test_call_single_returns_prob(self):
        img = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        probs = self.model(img)
        self.assertIsInstance(probs, torch.Tensor)
        self.assertEqual(probs.shape, (1,))

    def test_probs_in_range(self):
        img = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        probs = self.model(img)
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_batch_shape(self):
        imgs = torch.rand(4, 3, 64, 64, dtype=torch.float32)
        probs = self.model(imgs)
        self.assertEqual(probs.shape, (4,))

    def test_eyes_open_static_method(self):
        B = 4
        left_state = torch.tensor([0.1, 0.9, 0.1, 0.9])
        right_state = torch.tensor([0.1, 0.9, 0.9, 0.1])
        left_valid = torch.ones(B, dtype=torch.bool)
        right_valid = torch.ones(B, dtype=torch.bool)
        result = BlinkDenseNet121Wrapper.eyes_open(left_state, right_state, left_valid, right_valid)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (B,))
        self.assertTrue(result[0])  # both open
        self.assertFalse(result[1])  # both closed

    def test_predict_pipeline(self):
        B = 3
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        landmarks = _make_landmarks_478(B)
        results = self.model.predict_pipeline(frames, landmarks)
        self.assertIsInstance(results, (tuple, list))
        self.assertEqual(len(results), 4)

    def test_predict_frame(self):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        B = 1
        landmarks = _make_landmarks_478(B)
        result = self.model.predict_frame(frame, landmarks[0])
        self.assertIsNotNone(result)


class TestBlinkPredictPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BlinkDenseNet121Wrapper(device_id=None)

    def test_predict_pipeline_returns_4_arrays(self):
        B = 4
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        lmks = _make_landmarks_6(B)
        result = self.model.predict_pipeline(frames, lmks)
        self.assertEqual(len(result), 4)
        for arr in result:
            self.assertIsInstance(arr, torch.Tensor)
            self.assertEqual(arr.shape, (B,))

    def test_predict_pipeline_probs_in_range(self):
        B = 2
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        lmks = _make_landmarks_6(B)
        left_state, right_state, left_valid, right_valid = self.model.predict_pipeline(frames, lmks)
        for arr in [left_state, right_state]:
            self.assertTrue((arr >= 0.0).all())
            self.assertTrue((arr <= 1.0).all())

    def test_predict_frame_single(self):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        lmks = _make_landmarks_6(1)[0]  # (6, 2)
        result = self.model.predict_frame(frame, lmks)
        self.assertEqual(len(result), 4)

    def test_visualize_numpy_in_numpy_out(self):
        B = 2
        frames = np.random.randint(0, 255, (B, 112, 112, 3), dtype=np.uint8)
        left_state = np.array([0.3, 0.7])
        right_state = np.array([0.2, 0.8])
        left_valid = np.ones(B, dtype=bool)
        right_valid = np.ones(B, dtype=bool)
        result = BlinkDenseNet121Wrapper.visualize(
            frames, left_state, right_state, left_valid, right_valid
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], B)


class TestBlinkDenseNetCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BlinkDenseNet121Wrapper(device_id=None)

    def test_predict_pipeline_tensor_input(self):
        """torch.Tensor (B, 3, H, W) input → line 90: frames.permute(...).cpu().numpy()."""
        B = 2
        frames_t = torch.randint(0, 255, (B, 3, 224, 224), dtype=torch.uint8)
        lmks = _make_landmarks_6(B)
        result = self.model.predict_pipeline(frames_t, lmks)
        self.assertEqual(len(result), 4)
        for t in result:
            self.assertIsInstance(t, torch.Tensor)
            self.assertEqual(t.shape[0], B)

    def test_predict_pipeline_with_headpose(self):
        """headpose not None → lines 99-103: yaw-based masking of valid eyes."""
        B = 2
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        lmks = _make_landmarks_6(B)
        headpose = np.array([[60.0, 0.0, 0.0], [-60.0, 0.0, 0.0]], dtype=np.float32)
        left_state, right_state, left_valid, right_valid = self.model.predict_pipeline(
            frames, lmks, headpose
        )
        self.assertFalse(left_valid[0])
        self.assertFalse(right_valid[1])

    def test_predict_pipeline_empty_left_crop(self):
        """Left eye far outside right edge of frame → lines 141-142: left crop empty."""
        B = 1
        frames = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
        lmks = _make_landmarks_6(B, left_x=500, left_y=112, right_x=510, right_y=112)
        _, _, left_valid, right_valid = self.model.predict_pipeline(frames, lmks)
        self.assertFalse(left_valid[0])
        self.assertFalse(right_valid[0])

    def test_predict_frame_tensor_input(self):
        """torch.Tensor (3, H, W) frame → line 218: frame.permute(1,2,0).cpu().numpy()."""
        frame_t = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        lmks_2d = _make_landmarks_6(1)[0]  # (6, 2)
        result = self.model.predict_frame(frame_t, lmks_2d)
        self.assertEqual(len(result), 4)

    def test_predict_frame_return_patches_true(self):
        """return_patches=True → lines 235-254, 275: left/right patch extracted."""
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        lmks_2d = _make_landmarks_6(1)[0]  # (6, 2)
        result = self.model.predict_frame(frame, lmks_2d, return_patches=True)
        self.assertEqual(len(result), 6)
        left_patch, right_patch = result[4], result[5]
        self.assertEqual(left_patch.shape, (64, 64, 3))
        self.assertEqual(right_patch.shape, (64, 64, 3))

    def test_visualize_tensor_input_returns_tensor(self):
        """torch.Tensor frames → line 347: permute, line 398: return tensor."""
        B = 2
        frames_t = torch.randint(0, 255, (B, 3, 112, 112), dtype=torch.uint8)
        left_state = np.array([0.3, 0.7])
        right_state = np.array([0.2, 0.8])
        left_valid = np.ones(B, dtype=bool)
        right_valid = np.ones(B, dtype=bool)
        result = BlinkDenseNet121Wrapper.visualize(
            frames_t, left_state, right_state, left_valid, right_valid
        )
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], B)

    def test_visualize_with_landmarks(self):
        """landmarks not None → lines 376-387: draw circles at eye locations."""
        B = 2
        frames = np.random.randint(0, 255, (B, 112, 112, 3), dtype=np.uint8)
        left_state = np.array([0.3, 0.7])
        right_state = np.array([0.2, 0.8])
        left_valid = np.ones(B, dtype=bool)
        right_valid = np.ones(B, dtype=bool)
        lmks = _make_landmarks_6(B, left_x=30, left_y=30, right_x=80, right_y=30)
        result = BlinkDenseNet121Wrapper.visualize(
            frames, left_state, right_state, left_valid, right_valid, landmarks=lmks
        )
        self.assertIsInstance(result, np.ndarray)

    def test_visualize_with_output_path(self):
        """output_path not None → lines 390-395: save each frame."""
        B = 2
        frames = np.random.randint(0, 255, (B, 112, 112, 3), dtype=np.uint8)
        left_state = np.array([0.3, 0.7])
        right_state = np.array([0.2, 0.8])
        left_valid = np.ones(B, dtype=bool)
        right_valid = np.ones(B, dtype=bool)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "blink_vis.png"
            BlinkDenseNet121Wrapper.visualize(
                frames, left_state, right_state, left_valid, right_valid, output_path=str(out)
            )
            saved = list(Path(d).glob("*.png"))
            self.assertGreater(len(saved), 0)

    def test_visualize_with_invalid_eye_skips_text(self):
        """left_valid=False for frame → if left_eye_valid[i]: branch not taken."""
        B = 2
        frames = np.random.randint(0, 255, (B, 112, 112, 3), dtype=np.uint8)
        left_state = np.array([0.3, 0.7])
        right_state = np.array([0.2, 0.8])
        left_valid = np.array([False, True], dtype=bool)
        right_valid = np.array([True, False], dtype=bool)
        result = BlinkDenseNet121Wrapper.visualize(
            frames, left_state, right_state, left_valid, right_valid
        )
        self.assertEqual(result.shape[0], B)


if __name__ == "__main__":
    unittest.main()
