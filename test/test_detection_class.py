import unittest
from pathlib import Path
import cv2
import numpy as np
from exordium.video.detection import Detection

RESOLUTION = '720p'
FRAME_PATH = f'data/processed/frames/multispeaker_{RESOLUTION}/000000.png'
VIDEO_PATH = f'data/videos/multispeaker_{RESOLUTION}.mp4'
assert Path(FRAME_PATH).exists() and Path(VIDEO_PATH).exists()


class DetectionTestCase(unittest.TestCase):

    def setUp(self):
        self.detection_frame = Detection(
            frame_id=0,
            frame_path=FRAME_PATH,
            score=0.9,
            bb_xywh=np.array([100, 100, 200, 200]),
            bb_xyxy=np.array([100, 100, 300, 300]),
            landmarks=np.array([1, 2, 3, 4])
        )
        self.detection_video = Detection(
            frame_id=0,
            frame_path=VIDEO_PATH,
            score=0.9,
            bb_xywh=np.array([100, 100, 200, 200]),
            bb_xyxy=np.array([100, 100, 300, 300]),
            landmarks=np.array([1, 2, 3, 4])
        )

    def test_crop_from_frame(self):
        cropped_image = self.detection_frame.crop()
        self.assertIsInstance(cropped_image, np.ndarray)

    def test_crop_from_video(self):
        cropped_image = self.detection_video.crop()
        self.assertIsInstance(cropped_image, np.ndarray)

    def test_frame_center_from_frame(self):
        frame_center = self.detection_frame.frame_center()
        frame = cv2.imread(self.detection_frame.frame_path)
        height, width, _ = frame.shape
        self.assertIsInstance(frame_center, np.ndarray)
        self.assertEqual(frame_center.shape, (2,))
        self.assertEqual(frame_center[0], width // 2)
        self.assertEqual(frame_center[1], height // 2)

    def test_frame_center_from_video(self):
        frame_center = self.detection_video.frame_center()
        frame = cv2.imread(self.detection_frame.frame_path)
        height, width, _ = frame.shape
        self.assertIsInstance(frame_center, np.ndarray)
        self.assertEqual(frame_center.shape, (2,))
        self.assertEqual(frame_center[0], width // 2)
        self.assertEqual(frame_center[1], height // 2)

    def test_equal(self):
        detection1 = Detection(
            frame_id=1,
            frame_path='path/to/frame1.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            bb_xyxy=np.array([150, 150, 400, 400]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        detection2 = Detection(
            frame_id=1,
            frame_path='path/to/frame1.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            bb_xyxy=np.array([150, 150, 400, 400]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        detection3 = Detection(
            frame_id=1,
            frame_path='path/to/frame1.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 300]),
            bb_xyxy=np.array([150, 150, 400, 400]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        self.assertEqual(detection1, detection2)
        self.assertNotEqual(detection1, detection3)
        self.assertNotEqual(detection2, detection3)

if __name__ == '__main__':
    unittest.main()