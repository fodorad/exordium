"""Tests for bounding-box utility functions in exordium.video.core.bb."""

import unittest

import numpy as np
import torch

from exordium.video.core.bb import (
    apply_10_crop,
    center_crop,
    crop_mid,
    crop_xyxy,
    iou_xywh,
    iou_xyxy,
    midwh2xywh,
    visualize_bb,
    xywh2midwh,
    xywh2xyxy,
    xyxy2full,
    xyxy2xywh,
)


class TestFormatConversions(unittest.TestCase):
    """Round-trip and correctness tests for bounding-box format conversions."""

    def _assert_close(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a.float(), b.float(), atol=1e-4, rtol=1e-4)
        else:
            np.testing.assert_allclose(a, b, atol=1e-4)

    def test_xywh2xyxy_numpy(self):
        bb = np.array([10.0, 20.0, 30.0, 40.0])
        out = xywh2xyxy(bb)
        self._assert_close(out, np.array([10.0, 20.0, 40.0, 60.0]))
        self.assertIsInstance(out, np.ndarray)

    def test_xywh2xyxy_tensor(self):
        bb = torch.tensor([10.0, 20.0, 30.0, 40.0])
        out = xywh2xyxy(bb)
        self._assert_close(out, torch.tensor([10.0, 20.0, 40.0, 60.0]))
        self.assertIsInstance(out, torch.Tensor)

    def test_xyxy2xywh_roundtrip_numpy(self):
        bb = np.array([10.0, 20.0, 40.0, 60.0])
        self._assert_close(xyxy2xywh(xywh2xyxy(xyxy2xywh(bb))), xyxy2xywh(bb))

    def test_xyxy2xywh_roundtrip_tensor(self):
        bb = torch.tensor([10.0, 20.0, 40.0, 60.0])
        self._assert_close(xyxy2xywh(xywh2xyxy(xyxy2xywh(bb))), xyxy2xywh(bb))

    def test_midwh2xywh_numpy(self):
        mid = np.array([25.0, 40.0, 30.0, 40.0])
        out = midwh2xywh(mid)
        self._assert_close(out, np.array([10.0, 20.0, 30.0, 40.0]))
        self.assertIsInstance(out, np.ndarray)

    def test_midwh2xywh_tensor(self):
        mid = torch.tensor([25.0, 40.0, 30.0, 40.0])
        out = midwh2xywh(mid)
        self.assertIsInstance(out, torch.Tensor)
        self._assert_close(out, torch.tensor([10.0, 20.0, 30.0, 40.0]))

    def test_xywh2midwh_roundtrip(self):
        bb = np.array([10.0, 20.0, 30.0, 40.0])
        self._assert_close(midwh2xywh(xywh2midwh(bb)), bb)

    def test_xyxy2full_numpy(self):
        bb = np.array([0.0, 0.0, 4.0, 6.0])
        corners = xyxy2full(bb)
        self.assertEqual(corners.shape, (4, 2))
        self.assertIsInstance(corners, np.ndarray)

    def test_xyxy2full_tensor(self):
        bb = torch.tensor([0.0, 0.0, 4.0, 6.0])
        corners = xyxy2full(bb)
        self.assertEqual(corners.shape, (4, 2))
        self.assertIsInstance(corners, torch.Tensor)


class TestIoU(unittest.TestCase):
    def test_perfect_overlap_xyxy(self):
        bb = np.array([0.0, 0.0, 10.0, 10.0])
        self.assertAlmostEqual(iou_xyxy(bb, bb), 1.0)

    def test_no_overlap_xyxy(self):
        a = np.array([0.0, 0.0, 5.0, 5.0])
        b = np.array([10.0, 10.0, 15.0, 15.0])
        self.assertAlmostEqual(iou_xyxy(a, b), 0.0)

    def test_perfect_overlap_xywh(self):
        bb = np.array([0.0, 0.0, 10.0, 10.0])
        self.assertAlmostEqual(iou_xywh(bb, bb), 1.0)

    def test_partial_overlap(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([5.0, 5.0, 10.0, 10.0])
        iou = iou_xywh(a, b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


class TestCropFunctions(unittest.TestCase):
    def setUp(self):
        self.np_img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        self.t_img = torch.randint(0, 255, (3, 100, 80), dtype=torch.uint8)

    def test_crop_mid_numpy_type(self):
        mid = np.array([40, 40])
        out = crop_mid(self.np_img, mid, 30)
        self.assertIsInstance(out, np.ndarray)

    def test_crop_mid_tensor_type(self):
        mid = torch.tensor([40, 40])
        out = crop_mid(self.t_img, mid, 30)
        self.assertIsInstance(out, torch.Tensor)

    def test_crop_xyxy_numpy(self):
        bb = np.array([10, 10, 50, 60])
        out = crop_xyxy(self.np_img, bb)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape[0], 50)
        self.assertEqual(out.shape[1], 40)

    def test_crop_xyxy_tensor(self):
        bb = torch.tensor([10, 10, 50, 60])
        out = crop_xyxy(self.t_img, bb)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[1], 50)
        self.assertEqual(out.shape[2], 40)

    def test_center_crop_numpy(self):
        out = center_crop(self.np_img, (60, 50))
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (60, 50, 3))

    def test_center_crop_tensor(self):
        out = center_crop(self.t_img, (60, 50))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, 60, 50))

    def test_apply_10_crop_numpy(self):
        crops = apply_10_crop(self.np_img, (40, 40))
        self.assertEqual(len(crops), 10)
        for c in crops:
            self.assertIsInstance(c, np.ndarray)
            self.assertEqual(c.shape, (40, 40, 3))

    def test_apply_10_crop_tensor(self):
        crops = apply_10_crop(self.t_img, (40, 40))
        self.assertEqual(len(crops), 10)
        for c in crops:
            self.assertIsInstance(c, torch.Tensor)
            self.assertEqual(c.shape, (3, 40, 40))


class TestVisualizeBb(unittest.TestCase):
    def setUp(self):
        self.np_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        self.t_rgb = torch.zeros(3, 100, 100, dtype=torch.uint8)
        self.bb = np.array([10.0, 10.0, 50.0, 50.0])

    def test_numpy_in_numpy_out(self):
        out = visualize_bb(self.np_bgr, self.bb, 0.9)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_tensor_in_tensor_out(self):
        out = visualize_bb(self.t_rgb, self.bb, 0.9)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, 100, 100))

    def test_tensor_bb_accepted(self):
        bb_t = torch.tensor([10.0, 10.0, 50.0, 50.0])
        out = visualize_bb(self.np_bgr, bb_t, 0.9)
        self.assertIsInstance(out, np.ndarray)

    def test_wrong_bb_shape_raises(self):
        with self.assertRaises(ValueError):
            visualize_bb(self.np_bgr, np.array([1.0, 2.0, 3.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
