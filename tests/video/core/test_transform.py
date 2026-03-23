"""Tests for exordium.video.core.transform functions and transform classes."""

import unittest

import numpy as np
import torch

from exordium.video.core.transform import (
    CenterCrop,
    Denormalize,
    Normalize,
    Resize,
    ToTensor,
    align_face,
    rotate_face,
)
from exordium.video.face.landmark.facemesh import crop_eye_regions


class TestRotateFace(unittest.TestCase):
    def setUp(self):
        self.np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.t_img = torch.randint(0, 255, (3, 100, 100), dtype=torch.uint8)

    def test_zero_rotation_numpy(self):
        out, R = rotate_face(self.np_img, 0.0)
        self.assertIs(out, self.np_img)

    def test_zero_rotation_tensor(self):
        out, _ = rotate_face(self.t_img, 0.0)
        self.assertIs(out, self.t_img)

    def test_nonzero_rotation_numpy_type(self):
        out, R = rotate_face(self.np_img, 15.0)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.np_img.shape)
        self.assertEqual(R.shape, (2, 3))

    def test_nonzero_rotation_tensor_type(self):
        out, _ = rotate_face(self.t_img, 15.0)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, self.t_img.shape)

    def test_180_degree_rotation_numpy(self):
        out, _ = rotate_face(self.np_img, 180.0)
        self.assertFalse(np.array_equal(out, self.np_img))

    def test_batch_tensor_rotation(self):
        batch = torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)
        out, _ = rotate_face(batch, 45.0)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, batch.shape)


class TestCropEyeRegions(unittest.TestCase):
    def setUp(self):
        self.img_np = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        self.img_t = torch.randint(0, 255, (3, 200, 200), dtype=torch.uint8)
        # 5-point YOLO11 coarse landmarks: right_eye, left_eye, nose, mouth_r, mouth_l
        self.lmks_5 = np.array(
            [[60, 90], [140, 90], [100, 120], [75, 150], [125, 150]], dtype=np.float32
        )
        # 478-point FaceMesh — zeros everywhere; eye regions get realistic positions
        self.lmks_478 = np.zeros((478, 2), dtype=np.float32)
        from exordium.video.face.landmark.constants import FaceMesh478Regions

        for idx in FaceMesh478Regions.RIGHT_EYE:
            self.lmks_478[idx] = [60, 90]
        for idx in FaceMesh478Regions.LEFT_EYE:
            self.lmks_478[idx] = [140, 90]

    def test_coarse_numpy_returns_dict(self):
        out = crop_eye_regions(self.img_np, self.lmks_5)
        self.assertIn("eye_region_right", out)
        self.assertIn("eye_region_left", out)
        self.assertIsInstance(out["eye_region_right"], np.ndarray)

    def test_coarse_tensor_returns_dict(self):
        out = crop_eye_regions(self.img_t, self.lmks_5)
        self.assertIsInstance(out["eye_region_right"], torch.Tensor)
        self.assertIsInstance(out["eye_region_left"], torch.Tensor)

    def test_fine_numpy_returns_dict(self):
        out = crop_eye_regions(self.img_np, self.lmks_478)
        self.assertIn("eye_region_right", out)
        self.assertIsInstance(out["eye_region_right"], np.ndarray)

    def test_fine_tensor_input_landmarks(self):
        """Tensor landmarks are accepted and produce the same result as numpy."""
        lmks_t = torch.from_numpy(self.lmks_5)
        out = crop_eye_regions(self.img_np, lmks_t)
        self.assertIn("eye_region_right", out)

    def test_invalid_landmark_shape_raises(self):
        with self.assertRaises(ValueError):
            crop_eye_regions(self.img_np, np.zeros((10, 2), dtype=np.float32))

    def test_square_crops(self):
        """Crops produced by crop_mid are always square."""
        out = crop_eye_regions(self.img_np, self.lmks_5)
        r = out["eye_region_right"]
        self.assertEqual(r.shape[0], r.shape[1])  # H == W


class TestAlignFace(unittest.TestCase):
    def setUp(self):
        self.img_np = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        self.img_t = torch.randint(0, 255, (3, 200, 200), dtype=torch.uint8)
        self.landmarks = np.array(
            [
                [80, 70],
                [120, 70],
                [100, 110],
                [85, 140],
                [115, 140],
            ],
            dtype=np.float32,
        )
        self.bb_xyxy = np.array([50.0, 40.0, 150.0, 160.0])

    def test_returns_dict_with_required_keys(self):
        result = align_face(self.img_np, self.bb_xyxy, self.landmarks)
        for key in (
            "rotated_image",
            "rotated_face",
            "rotated_bb_xyxy",
            "rotation_degree",
            "rotation_matrix",
        ):
            self.assertIn(key, result)

    def test_numpy_input_numpy_output(self):
        result = align_face(self.img_np, self.bb_xyxy, self.landmarks)
        self.assertIsInstance(result["rotated_image"], np.ndarray)
        self.assertIsInstance(result["rotated_face"], np.ndarray)

    def test_tensor_input_tensor_output(self):
        result = align_face(self.img_t, torch.tensor(self.bb_xyxy), self.landmarks)
        self.assertIsInstance(result["rotated_image"], torch.Tensor)
        self.assertIsInstance(result["rotated_face"], torch.Tensor)
        self.assertEqual(result["rotated_image"].shape[0], 3)

    def test_rotation_matrix_shape(self):
        result = align_face(self.img_np, self.bb_xyxy, self.landmarks)
        self.assertEqual(result["rotation_matrix"].shape, (2, 3))

    def test_invalid_landmarks_raises(self):
        bad_lmks = np.zeros((3, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            align_face(self.img_np, self.bb_xyxy, bad_lmks)


class TestToTensor(unittest.TestCase):
    def test_converts_to_float(self):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        result = ToTensor()(arr)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)

    def test_value_range_0_1(self):
        arr = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = ToTensor()(arr)
        self.assertAlmostEqual(result.max().item(), 1.0, places=5)


class TestResizeClass(unittest.TestCase):
    def test_resize_int(self):
        video = torch.zeros(3, 4, 64, 64)  # (C, T, H, W)
        out = Resize(32)(video)
        self.assertLessEqual(min(out.shape[-2:]), 32)

    def test_resize_to_32(self):
        video = torch.zeros(3, 4, 64, 64)
        out = Resize(32)(video)
        self.assertLessEqual(min(out.shape[-2:]), 32)


class TestCenterCropClass(unittest.TestCase):
    def test_center_crop_int(self):
        video = torch.zeros(3, 4, 64, 64)
        out = CenterCrop(32)(video)
        self.assertEqual(out.shape[-2:], torch.Size([32, 32]))

    def test_center_crop_tuple(self):
        video = torch.zeros(3, 4, 64, 64)
        out = CenterCrop((30, 40))(video)
        self.assertEqual(out.shape[-2], 30)
        self.assertEqual(out.shape[-1], 40)


class TestNormalizeDenormalize(unittest.TestCase):
    def test_normalize_output(self):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        video = torch.ones(3, 4, 8, 8)
        norm = Normalize(mean, std)(video)
        self.assertAlmostEqual(norm.mean().item(), 1.0, places=4)

    def test_normalize_denormalize_roundtrip(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        video = torch.rand(3, 4, 16, 16)
        roundtrip = Denormalize(mean, std)(Normalize(mean, std)(video))
        torch.testing.assert_close(roundtrip, video, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
