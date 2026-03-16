"""Tests for exordium.video.face.blink module."""

import unittest

import numpy as np
import torch


class TestBlinkWrapper(unittest.TestCase):
    """Tests for BlinkDenseNet121."""

    @classmethod
    def setUpClass(cls):
        from exordium.video.face.blink import BlinkDenseNet121

        cls.model = BlinkDenseNet121(device_id=None)

    def _make_frame(self, h=480, w=640):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def _make_landmarks(self, B=1):
        # MediaPipe FaceDetector: 6 keypoints (x, y)
        # Place eyes far enough apart (> 10px)
        lm = np.zeros((B, 6, 2), dtype=np.float32)
        lm[:, 0] = [[200, 200]]  # left eye
        lm[:, 1] = [[400, 200]]  # right eye (200px apart)
        lm[:, 2] = [[300, 280]]  # nose
        lm[:, 3] = [[300, 350]]  # mouth
        lm[:, 4] = [[120, 200]]  # left ear
        lm[:, 5] = [[480, 200]]  # right ear
        return lm

    def test_init(self):
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.yaw_threshold, 40.0)

    def test_call_with_tensor(self):
        samples = torch.rand(4, 3, 64, 64)
        probs = self.model(samples)
        self.assertEqual(probs.shape, (4,))
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))

    def test_predict_pipeline_output_shapes(self):
        B = 3
        frames = np.stack([self._make_frame() for _ in range(B)])
        landmarks = self._make_landmarks(B)
        left_state, right_state, left_valid, right_valid = self.model.predict_pipeline(
            frames, landmarks
        )
        self.assertEqual(left_state.shape, (B,))
        self.assertEqual(right_state.shape, (B,))
        self.assertEqual(left_valid.shape, (B,))
        self.assertEqual(right_valid.shape, (B,))

    def test_predict_pipeline_with_headpose(self):
        B = 3
        frames = np.stack([self._make_frame() for _ in range(B)])
        landmarks = self._make_landmarks(B)
        headpose = np.array(
            [[30.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-50.0, 0.0, 0.0]], dtype=np.float32
        )
        left_state, right_state, left_valid, right_valid = self.model.predict_pipeline(
            frames, landmarks, headpose
        )
        self.assertEqual(left_state.shape, (B,))
        # With yaw=50 turning left, right eye should be occluded
        self.assertFalse(right_valid[2])

    def test_predict_pipeline_invalid_eye_distance(self):
        B = 2
        frames = np.stack([self._make_frame() for _ in range(B)])
        landmarks = np.zeros((B, 6, 2), dtype=np.float32)
        # Eyes 5px apart — invalid (< 10)
        landmarks[:, 0] = [[100, 100]]
        landmarks[:, 1] = [[105, 100]]
        left_state, right_state, left_valid, right_valid = self.model.predict_pipeline(
            frames, landmarks
        )
        self.assertFalse(left_valid[0])
        self.assertFalse(right_valid[0])

    def test_predict_frame_no_return_patches(self):
        frame = self._make_frame()
        landmarks = self._make_landmarks(1)[0]
        result = self.model.predict_frame(frame, landmarks)
        self.assertEqual(len(result), 4)
        left_state, right_state, left_valid, right_valid = result
        self.assertIsInstance(left_state, float)
        self.assertIsInstance(right_state, float)

    def test_predict_frame_with_return_patches(self):
        frame = self._make_frame()
        landmarks = self._make_landmarks(1)[0]
        result = self.model.predict_frame(frame, landmarks, return_patches=True)
        self.assertEqual(len(result), 6)
        left_patch, right_patch = result[4], result[5]
        self.assertEqual(left_patch.shape, (64, 64, 3))
        self.assertEqual(right_patch.shape, (64, 64, 3))

    def test_predict_frame_with_headpose(self):
        frame = self._make_frame()
        landmarks = self._make_landmarks(1)[0]
        headpose = np.array([20.0, 0.0, 0.0])
        result = self.model.predict_frame(frame, landmarks, headpose=headpose)
        self.assertEqual(len(result), 4)

    def test_eyes_open_both_valid_open(self):
        left_state = np.array([0.1, 0.2])
        right_state = np.array([0.3, 0.4])
        left_valid = np.array([True, True])
        right_valid = np.array([True, True])
        result = self.model.eyes_open(left_state, right_state, left_valid, right_valid)
        self.assertTrue(np.all(result))

    def test_eyes_open_one_closed(self):
        left_state = np.array([0.8])  # closed
        right_state = np.array([0.2])  # open
        left_valid = np.array([True])
        right_valid = np.array([True])
        result = self.model.eyes_open(left_state, right_state, left_valid, right_valid)
        self.assertFalse(result[0])

    def test_eyes_open_both_invalid(self):
        left_state = np.array([0.1])
        right_state = np.array([0.1])
        left_valid = np.array([False])
        right_valid = np.array([False])
        result = self.model.eyes_open(left_state, right_state, left_valid, right_valid)
        self.assertFalse(result[0])

    def test_visualize_no_landmarks(self):
        B = 2
        frames = np.stack([self._make_frame(120, 160) for _ in range(B)])
        left_state = np.array([0.3, 0.7])
        right_state = np.array([0.2, 0.8])
        left_valid = np.array([True, True])
        right_valid = np.array([True, False])
        vis = self.model.visualize(frames, left_state, right_state, left_valid, right_valid)
        self.assertEqual(vis.shape, frames.shape)

    def test_visualize_with_landmarks(self):
        B = 2
        frames = np.stack([self._make_frame(480, 640) for _ in range(B)])
        landmarks = self._make_landmarks(B)
        left_state = np.array([0.2, 0.6])
        right_state = np.array([0.4, 0.8])
        left_valid = np.array([True, True])
        right_valid = np.array([True, True])
        vis = self.model.visualize(
            frames, left_state, right_state, left_valid, right_valid, landmarks=landmarks
        )
        self.assertEqual(vis.shape, frames.shape)


if __name__ == "__main__":
    unittest.main()
