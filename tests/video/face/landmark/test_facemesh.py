"""Tests for FaceMeshWrapper, rotate_landmarks, and visualize_landmarks."""

import gc
import unittest

import numpy as np
import torch

from exordium.video.core.detection import DetectionFactory, Track
from exordium.video.face.landmark.facemesh import (
    FaceMeshWrapper,
    rotate_landmarks,
    visualize_landmarks,
)
from tests.fixtures import IMAGE_FACE


def _make_detection(frame_id):
    """Create a DetectionFromImage with a face crop from IMAGE_FACE."""
    return DetectionFactory.create_detection(
        frame_id=frame_id,
        source=str(IMAGE_FACE),
        score=0.95,
        bb_xywh=np.array([20, 20, 150, 150], dtype=np.int32),
        landmarks=np.zeros((5, 2), dtype=np.int32),
    )


class TestFaceMeshWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FaceMeshWrapper()

    def test_call_with_image_path(self):
        result = self.model([IMAGE_FACE])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_output_shape(self):
        result = self.model([IMAGE_FACE])
        lmks = result[0]
        self.assertIsInstance(lmks, torch.Tensor)
        self.assertEqual(lmks.shape, (478, 2))

    def test_call_with_numpy_image(self):
        from exordium.video.core.io import image_to_np

        img = image_to_np(IMAGE_FACE, "RGB")
        result = self.model([img])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], torch.Tensor)

    def test_call_with_tensor_image(self):
        from exordium.video.core.io import image_to_tensor

        t = image_to_tensor(IMAGE_FACE)
        result = self.model([t])
        self.assertIsInstance(result, list)

    def test_rotate_landmarks_shape(self):
        lmks = np.random.rand(478, 2).astype(np.float32)
        R = np.eye(2, 3, dtype=np.float32)
        out = rotate_landmarks(lmks, R)
        self.assertEqual(out.shape, (478, 2))

    def test_visualize_landmarks_numpy_in_numpy_out(self):
        lmks = np.zeros((478, 2), dtype=np.float32)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = visualize_landmarks(img, lmks, show_indices=False)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_visualize_landmarks_tensor_in_tensor_out(self):
        lmks = np.zeros((478, 2), dtype=np.float32)
        img = torch.zeros(3, 100, 100, dtype=torch.uint8)
        out = visualize_landmarks(img, lmks, show_indices=False)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, 100, 100))

    def test_visualize_landmarks_invalid_shape_raises(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bad_lmks = np.zeros((5, 3), dtype=np.float32)
        with self.assertRaises(Exception):
            visualize_landmarks(img, bad_lmks)


class TestFaceMeshStaticMethods(unittest.TestCase):
    def test_to_numpy_rgb_hwc_tensor(self):
        """HWC tensor (shape[0] != 3) → line 112: return image.cpu().numpy()."""
        hwc = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
        result = FaceMeshWrapper._to_numpy_rgb(hwc)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (64, 64, 3))


class TestFaceMeshTrackToFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FaceMeshWrapper()

    def test_track_to_feature_returns_landmarks(self):
        """track_to_feature → lines 179-197: batch loop, landmark extraction."""
        track = Track(0)
        for i in range(2):
            track.add(_make_detection(i))

        result = self.model.track_to_feature(track)

        self.assertIsInstance(result, dict)
        self.assertIn("frame_ids", result)
        self.assertIn("features", result)
        frame_ids = result["frame_ids"]
        landmarks = result["features"]
        self.assertIsInstance(frame_ids, torch.Tensor)
        self.assertIsInstance(landmarks, torch.Tensor)
        if len(frame_ids) > 0:
            self.assertEqual(landmarks.ndim, 3)
            self.assertEqual(landmarks.shape[1], 478)
            self.assertEqual(landmarks.shape[2], 2)

    def test_track_to_feature_empty_track_returns_empty(self):
        """Empty track → concatenation returns empty (0, 478, 2) tensor."""
        track = Track(99)
        # No detections added
        result = self.model.track_to_feature(track)
        self.assertEqual(result["frame_ids"].shape[0], 0)
        self.assertEqual(result["features"].shape, (0, 478, 2))


class TestFaceMeshDel(unittest.TestCase):
    def test_del_closes_landmarker(self):
        """__del__ → lines 201-202: landmarker.close() called without error."""
        wrapper = FaceMeshWrapper()
        wrapper.__del__()

    def test_del_via_gc(self):
        """Delete object and force GC to call __del__."""
        wrapper = FaceMeshWrapper()
        del wrapper
        gc.collect()


if __name__ == "__main__":
    unittest.main()
