import unittest
import numpy as np
from exordium.video.bb import xyxy2xywh, xywh2xyxy, midwh2xywh, xywh2midwh, iou_xywh, iou_xyxy


class TestBoundingBoxConversion(unittest.TestCase):

    def test_xyxy2xywh(self):
        # single_box
        xyxy = np.array([10, 20, 50, 60])
        xywh = xyxy2xywh(xyxy)
        self.assertEqual(xywh.shape, (4,))
        self.assertTrue(np.array_equal(xywh, np.array([10, 20, 40, 40])))

        # multiple_boxes
        xyxy = np.array([[10, 20, 50, 60], [30, 40, 70, 80]])
        xywh = xyxy2xywh(xyxy)
        self.assertEqual(xywh.shape, (2, 4))
        self.assertTrue(np.array_equal(xywh, np.array([[10, 20, 40, 40], [30, 40, 40, 40]])))

    def test_xywh2xyxy(self):
        # single_box
        xywh = np.array([10, 20, 30, 40])
        xyxy = xywh2xyxy(xywh)
        self.assertEqual(xyxy.shape, (4,))
        self.assertTrue(np.array_equal(xyxy, np.array([10, 20, 40, 60])))

        # multiple_boxes
        xywh = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        xyxy = xywh2xyxy(xywh)
        self.assertEqual(xyxy.shape, (2, 4))
        self.assertTrue(np.array_equal(xyxy, np.array([[10, 20, 40, 60], [50, 60, 120, 140]])))

    def test_midwh2xywh(self):
        # single_box
        midwh = np.array([50, 60, 30, 40])
        xywh = midwh2xywh(midwh)
        self.assertEqual(xywh.shape, (4,))
        self.assertTrue(np.array_equal(xywh, np.array([35, 40, 30, 40])))

        # multiple_boxes
        midwh = np.array([[50, 60, 30, 40], [100, 120, 50, 60]])
        xywh = midwh2xywh(midwh)
        self.assertEqual(xywh.shape, (2, 4))
        self.assertTrue(np.array_equal(xywh, np.array([[35, 40, 30, 40], [75, 90, 50, 60]])))

    def test_xywh2midwh(self):
        # single box
        xywh = np.array([10, 20, 30, 40])
        midwh = xywh2midwh(xywh)
        self.assertEqual(midwh.shape, (4,))
        self.assertTrue(np.array_equal(midwh, np.array([25, 40, 30, 40])))

        # multiple_boxes
        xywh = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        midwh = xywh2midwh(xywh)
        self.assertEqual(midwh.shape, (2, 4))
        self.assertTrue(np.array_equal(midwh, np.array([[25, 40, 30, 40], [85, 100, 70, 80]])))

    def test_iou_xywh(self):
        # Test case with overlapping bounding boxes
        bb1 = np.array([10, 10, 50, 70])  # xywh format
        bb2 = np.array([30, 40, 70, 80])  # xywh format
        iou = iou_xywh(bb1, bb2)
        self.assertAlmostEqual(iou, 0.1519, places=4)

        # Test case with one bounding box fully contained within the other
        bb3 = np.array([10, 10, 70, 110])  # xywh format
        bb4 = np.array([20, 40, 20, 40])  # xywh format
        iou = iou_xywh(bb3, bb4)
        self.assertAlmostEqual(iou, 0.1039, places=4)

        # Test case with non-overlapping bounding boxes
        bb5 = np.array([10, 20, 40, 40])  # xywh format
        bb6 = np.array([50, 70, 40, 40])  # xywh format
        iou = iou_xywh(bb5, bb6)
        self.assertEqual(iou, 0.0)

        # Test case with identical bounding boxes
        bb7 = np.array([10, 20, 40, 40])  # xywh format
        bb8 = np.array([10, 20, 40, 40])  # xywh format
        iou = iou_xywh(bb7, bb8)
        self.assertEqual(iou, 1.0)

    def test_iou_xyxy(self):
        # Test case with overlapping bounding boxes
        bb1 = np.array([10, 10, 60, 80])  # xyxy format
        bb2 = np.array([30, 40, 100, 120])  # xyxy format
        iou = iou_xyxy(bb1, bb2)
        self.assertAlmostEqual(iou, 0.1519, places=4)

        # Test case with one bounding box fully contained within the other
        bb3 = np.array([10, 10, 80, 120])  # xyxy format
        bb4 = np.array([20, 40, 40, 80])  # xyxy format
        iou = iou_xyxy(bb3, bb4)
        self.assertAlmostEqual(iou, 0.1039, places=4)

        # Test case with non-overlapping bounding boxes
        bb5 = np.array([10, 20, 40, 60])  # xyxy format
        bb6 = np.array([50, 70, 90, 110])  # xyxy format
        iou = iou_xyxy(bb5, bb6)
        self.assertEqual(iou, 0.0)

        # Test case with identical bounding boxes
        bb7 = np.array([10, 20, 40, 40])  # xywh format
        bb8 = np.array([10, 20, 40, 40])  # xywh format
        iou = iou_xyxy(bb7, bb8)
        self.assertEqual(iou, 1.0)

if __name__ == '__main__':
    unittest.main()
