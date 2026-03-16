"""Tests for exordium.video.core.bb module (merged from test_bb.py and test_bb_extended.py)."""

import unittest

import numpy as np

from exordium.video.core.bb import (
    apply_10_crop,
    center_crop,
    crop_mid,
    crop_xyxy,
    iou_xywh,
    iou_xyxy,
    midwh2xywh,
    xywh2midwh,
    xywh2xyxy,
    xyxy2full,
    xyxy2xywh,
)


class TestBoundingBoxConversion(unittest.TestCase):
    def test_xyxy2xywh(self):
        xyxy = np.array([10, 20, 50, 60])
        xywh = xyxy2xywh(xyxy)
        self.assertEqual(xywh.shape, (4,))
        self.assertTrue(np.array_equal(xywh, np.array([10, 20, 40, 40])))

        xyxy = np.array([[10, 20, 50, 60], [30, 40, 70, 80]])
        xywh = xyxy2xywh(xyxy)
        self.assertEqual(xywh.shape, (2, 4))
        self.assertTrue(np.array_equal(xywh, np.array([[10, 20, 40, 40], [30, 40, 40, 40]])))

    def test_xywh2xyxy(self):
        xywh = np.array([10, 20, 30, 40])
        xyxy = xywh2xyxy(xywh)
        self.assertEqual(xyxy.shape, (4,))
        self.assertTrue(np.array_equal(xyxy, np.array([10, 20, 40, 60])))

        xywh = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        xyxy = xywh2xyxy(xywh)
        self.assertEqual(xyxy.shape, (2, 4))
        self.assertTrue(np.array_equal(xyxy, np.array([[10, 20, 40, 60], [50, 60, 120, 140]])))

    def test_midwh2xywh(self):
        midwh = np.array([50, 60, 30, 40])
        xywh = midwh2xywh(midwh)
        self.assertEqual(xywh.shape, (4,))
        self.assertTrue(np.array_equal(xywh, np.array([35, 40, 30, 40])))

        midwh = np.array([[50, 60, 30, 40], [100, 120, 50, 60]])
        xywh = midwh2xywh(midwh)
        self.assertEqual(xywh.shape, (2, 4))
        self.assertTrue(np.array_equal(xywh, np.array([[35, 40, 30, 40], [75, 90, 50, 60]])))

    def test_xywh2midwh(self):
        xywh = np.array([10, 20, 30, 40])
        midwh = xywh2midwh(xywh)
        self.assertEqual(midwh.shape, (4,))
        self.assertTrue(np.array_equal(midwh, np.array([25, 40, 30, 40])))

        xywh = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        midwh = xywh2midwh(xywh)
        self.assertEqual(midwh.shape, (2, 4))
        self.assertTrue(np.array_equal(midwh, np.array([[25, 40, 30, 40], [85, 100, 70, 80]])))

    def test_iou_xywh(self):
        bb1 = np.array([10, 10, 50, 70])
        bb2 = np.array([30, 40, 70, 80])
        iou = iou_xywh(bb1, bb2)
        self.assertAlmostEqual(iou, 0.1519, places=4)

        bb3 = np.array([10, 10, 70, 110])
        bb4 = np.array([20, 40, 20, 40])
        iou = iou_xywh(bb3, bb4)
        self.assertAlmostEqual(iou, 0.1039, places=4)

        bb5 = np.array([10, 20, 40, 40])
        bb6 = np.array([50, 70, 40, 40])
        iou = iou_xywh(bb5, bb6)
        self.assertEqual(iou, 0.0)

        bb7 = np.array([10, 20, 40, 40])
        bb8 = np.array([10, 20, 40, 40])
        iou = iou_xywh(bb7, bb8)
        self.assertEqual(iou, 1.0)

    def test_iou_xyxy(self):
        bb1 = np.array([10, 10, 60, 80])
        bb2 = np.array([30, 40, 100, 120])
        iou = iou_xyxy(bb1, bb2)
        self.assertAlmostEqual(iou, 0.1519, places=4)

        bb3 = np.array([10, 10, 80, 120])
        bb4 = np.array([20, 40, 40, 80])
        iou = iou_xyxy(bb3, bb4)
        self.assertAlmostEqual(iou, 0.1039, places=4)

        bb5 = np.array([10, 20, 40, 60])
        bb6 = np.array([50, 70, 90, 110])
        iou = iou_xyxy(bb5, bb6)
        self.assertEqual(iou, 0.0)

        bb7 = np.array([10, 20, 40, 40])
        bb8 = np.array([10, 20, 40, 40])
        iou = iou_xyxy(bb7, bb8)
        self.assertEqual(iou, 1.0)


class TestCropMid(unittest.TestCase):
    def test_basic_crop(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        mid = np.array([100, 100])
        result = crop_mid(image, mid, bb_size=50)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[2], 3)

    def test_crop_size_matches(self):
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        mid = np.array([100, 100])
        result = crop_mid(image, mid, bb_size=40)
        self.assertEqual(result.shape[0], 40)
        self.assertEqual(result.shape[1], 40)

    def test_clamps_to_image_boundary(self):
        image = np.ones((100, 100, 3), dtype=np.uint8)
        mid = np.array([5, 5])
        result = crop_mid(image, mid, bb_size=50)
        self.assertIsNotNone(result)

    def test_mid_at_corner(self):
        image = np.ones((100, 100, 3), dtype=np.uint8)
        mid = np.array([0, 0])
        result = crop_mid(image, mid, bb_size=30)
        self.assertIsNotNone(result)


class TestCropXyxy(unittest.TestCase):
    def test_basic_crop(self):
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        bb_xyxy = np.array([50, 50, 150, 150])
        result = crop_xyxy(image, bb_xyxy)
        self.assertEqual(result.shape, (100, 100, 3))

    def test_clamps_to_boundary(self):
        image = np.ones((100, 100, 3), dtype=np.uint8)
        bb_xyxy = np.array([-10, -10, 200, 200])
        result = crop_xyxy(image, bb_xyxy)
        self.assertIsNotNone(result)


class TestCenterCrop(unittest.TestCase):
    def test_output_size(self):
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = center_crop(image, (100, 100))
        self.assertEqual(result.shape, (100, 100, 3))

    def test_non_square(self):
        image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        result = center_crop(image, (80, 120))
        self.assertEqual(result.shape, (80, 120, 3))

    def test_preserves_channels(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = center_crop(image, (50, 50))
        self.assertEqual(result.shape[2], 3)


class TestApply10Crop(unittest.TestCase):
    def test_returns_10_crops(self):
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        crops = apply_10_crop(image, (112, 112))
        self.assertEqual(crops.shape[0], 10)

    def test_crop_size(self):
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        crops = apply_10_crop(image, (100, 100))
        self.assertEqual(crops.shape[1], 100)
        self.assertEqual(crops.shape[2], 100)

    def test_channels_preserved(self):
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        crops = apply_10_crop(image, (112, 112))
        self.assertEqual(crops.shape[3], 3)

    def test_rectangular_image(self):
        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        crops = apply_10_crop(image, (100, 150))
        self.assertEqual(crops.shape[0], 10)
        self.assertEqual(crops.shape[1], 100)
        self.assertEqual(crops.shape[2], 150)


class TestXyxy2Full(unittest.TestCase):
    def test_output_shape(self):
        bb = np.array([10, 20, 50, 60])
        result = xyxy2full(bb)
        self.assertEqual(result.shape, (4, 2))

    def test_corners_correct(self):
        bb = np.array([10, 20, 50, 60])
        result = xyxy2full(bb)
        # Top-left
        np.testing.assert_array_equal(result[0], [10, 20])
        # Top-right
        np.testing.assert_array_equal(result[1], [50, 20])
        # Bottom-left
        np.testing.assert_array_equal(result[2], [10, 60])
        # Bottom-right
        np.testing.assert_array_equal(result[3], [50, 60])


if __name__ == "__main__":
    unittest.main()
