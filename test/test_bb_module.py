import unittest
import numpy as np
from exordium.video.bb import xyxy2xywh, xywh2xyxy, midwh2xywh, xywh2midwh, iou_xywh, iou_xyxy


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


if __name__ == '__main__':
    unittest.main()