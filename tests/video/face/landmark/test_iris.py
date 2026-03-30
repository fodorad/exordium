"""Tests for exordium.video.face.landmark.iris.IrisWrapper and utility functions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.face.landmark.iris import (
    IrisWrapper,
    calculate_eye_aspect_ratio,
    calculate_eyelid_pupil_distances,
    calculate_iris_diameters,
    visualize_iris,
)


class TestIrisWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = IrisWrapper(device_id=None)
        cls.eye = torch.randint(0, 255, (3, 80, 120), dtype=torch.uint8)

    # --- __call__ ---

    def test_call_single_eye_patch_numpy(self):
        eye_patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        eye_lmks, iris_lmks = self.model(eye_patch)
        self.assertEqual(eye_lmks.shape, (1, 71, 2))
        self.assertEqual(iris_lmks.shape, (1, 5, 2))

    def test_call_batch_tensor(self):
        imgs = torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)
        eye_lmks, iris_lmks = self.model(imgs)
        self.assertEqual(eye_lmks.shape, (4, 71, 2))
        self.assertEqual(iris_lmks.shape, (4, 5, 2))

    def test_list_of_numpy_arrays(self):
        patches = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        ]
        eye_lmks, iris_lmks = self.model(patches)
        self.assertEqual(eye_lmks.shape, (2, 71, 2))
        self.assertEqual(iris_lmks.shape, (2, 5, 2))

    # --- eye_to_feature ---

    def test_returns_all_keys(self):
        result = self.model.eye_to_feature(self.eye)
        for key in (
            "eye_original",
            "eye",
            "eye_region_landmarks",
            "iris_landmarks",
            "iris_diameters",
            "eyelid_pupil_distances",
            "ear",
        ):
            self.assertIn(key, result)

    def test_all_values_are_tensors(self):
        result = self.model.eye_to_feature(self.eye)
        for key, val in result.items():
            self.assertIsInstance(val, torch.Tensor, msg=f'"{key}" is not a torch.Tensor')

    def test_eye_original_preserves_input(self):
        result = self.model.eye_to_feature(self.eye)
        self.assertIs(result["eye_original"], self.eye)
        self.assertEqual(result["eye_original"].shape, (3, 80, 120))

    def test_eye_is_64x64_uint8(self):
        result = self.model.eye_to_feature(self.eye)
        self.assertEqual(result["eye"].shape, (3, 64, 64))
        self.assertEqual(result["eye"].dtype, torch.uint8)

    def test_eye_region_landmarks_shape(self):
        result = self.model.eye_to_feature(self.eye)
        self.assertEqual(result["eye_region_landmarks"].shape, (71, 2))

    def test_iris_landmarks_shape(self):
        result = self.model.eye_to_feature(self.eye)
        self.assertEqual(result["iris_landmarks"].shape, (5, 2))

    def test_iris_diameters_shape(self):
        result = self.model.eye_to_feature(self.eye)
        self.assertEqual(result["iris_diameters"].shape, (2,))

    def test_eyelid_pupil_distances_shape(self):
        result = self.model.eye_to_feature(self.eye)
        self.assertEqual(result["eyelid_pupil_distances"].shape, (2,))

    def test_ear_is_scalar(self):
        result = self.model.eye_to_feature(self.eye)
        self.assertEqual(result["ear"].ndim, 0)


class TestIrisUtilsTypePreserving(unittest.TestCase):
    """Metric functions must be type-preserving: numpy→numpy, torch→torch."""

    def _np_iris(self):
        return np.random.rand(5, 2).astype(np.float32)

    def _np_eye(self):
        return np.random.rand(71, 2).astype(np.float32)

    def _t_iris(self):
        return torch.rand(5, 2)

    def _t_eye(self):
        return torch.rand(71, 2)

    def test_iris_diameters_numpy_in_numpy_out(self):
        out = calculate_iris_diameters(self._np_iris())
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2,))

    def test_iris_diameters_tensor_in_tensor_out(self):
        out = calculate_iris_diameters(self._t_iris())
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (2,))

    def test_eyelid_distances_numpy_in_numpy_out(self):
        out = calculate_eyelid_pupil_distances(self._np_iris(), self._np_eye())
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2,))

    def test_eyelid_distances_tensor_in_tensor_out(self):
        out = calculate_eyelid_pupil_distances(self._t_iris(), self._t_eye())
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (2,))

    def test_ear_numpy_in_float_out(self):
        out = calculate_eye_aspect_ratio(self._np_eye())
        self.assertIsInstance(out, float)

    def test_ear_tensor_in_tensor_out(self):
        out = calculate_eye_aspect_ratio(self._t_eye())
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 0)

    def test_ear_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            calculate_eye_aspect_ratio(np.zeros((4, 2), dtype=np.float32))

    def test_ear_16pt_numpy(self):
        lmks = np.zeros((16, 2), dtype=np.float32)
        lmks[0] = [10, 50]
        lmks[8] = [90, 50]
        lmks[11] = [50, 40]
        lmks[3] = [50, 60]
        lmks[12] = [50, 38]
        lmks[4] = [50, 62]
        lmks[13] = [70, 40]
        lmks[5] = [70, 60]
        result = calculate_eye_aspect_ratio(lmks)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_ear_16pt_tensor(self):
        lmks = torch.zeros(16, 2)
        lmks[0] = torch.tensor([10.0, 50.0])
        lmks[8] = torch.tensor([90.0, 50.0])
        lmks[11] = torch.tensor([50.0, 40.0])
        lmks[3] = torch.tensor([50.0, 60.0])
        lmks[12] = torch.tensor([50.0, 38.0])
        lmks[4] = torch.tensor([50.0, 62.0])
        lmks[13] = torch.tensor([70.0, 40.0])
        lmks[5] = torch.tensor([70.0, 60.0])
        result = calculate_eye_aspect_ratio(lmks)
        self.assertIsInstance(result, torch.Tensor)
        self.assertGreater(result.item(), 0.0)


class TestVisualizeIris(unittest.TestCase):
    def _np_img(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def _t_img(self):
        return torch.zeros(3, 100, 100, dtype=torch.uint8)

    def _eye_lmks(self):
        return np.zeros((71, 2), dtype=np.float32)

    def _iris_lmks(self):
        return np.zeros((5, 2), dtype=np.float32)

    def test_numpy_in_numpy_out(self):
        out = visualize_iris(self._np_img(), self._eye_lmks(), self._iris_lmks())
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_tensor_in_tensor_out(self):
        out = visualize_iris(self._t_img(), self._eye_lmks(), self._iris_lmks())
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, 100, 100))

    def test_invalid_landmarks_shape_raises(self):
        with self.assertRaises(Exception):
            visualize_iris(self._np_img(), np.zeros((71, 3), dtype=np.float32), self._iris_lmks())

    def test_invalid_iris_landmarks_shape_raises(self):
        with self.assertRaises(Exception):
            visualize_iris(self._np_img(), self._eye_lmks(), np.zeros((5, 3), dtype=np.float32))

    def test_show_indices_true(self):
        out = visualize_iris(self._np_img(), self._eye_lmks(), self._iris_lmks(), show_indices=True)
        self.assertIsInstance(out, np.ndarray)

    def test_output_path_saves_file(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "sub" / "iris_vis.jpg"
            visualize_iris(
                self._np_img(), self._eye_lmks(), self._iris_lmks(), output_path=out_path
            )
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
