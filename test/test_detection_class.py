import unittest
import numpy as np
from decord import VideoReader
from exordium import EXAMPLE_VIDEO_PATH, DATA_DIR
from exordium.video.detection import DetectionFromImage, DetectionFromVideo


class DetectionTestCase(unittest.TestCase):

    def setUp(self):
        self.vr = VideoReader(str(EXAMPLE_VIDEO_PATH))
        self.detection_video = DetectionFromVideo(
            frame_id=0,
            source=str(EXAMPLE_VIDEO_PATH),
            score=0.9,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.zeros((5,2))
        )
        self.detections_from_video = [
            DetectionFromVideo(
                frame_id=frame_id,
                source=str(EXAMPLE_VIDEO_PATH),
                score=0.9,
                bb_xywh=np.array([100, 100, 200, 200]),
                landmarks=np.zeros((5,2))
            ) for frame_id in range(10)
        ]
        self.detection_image = DetectionFromImage(
            frame_id=0,
            source=str(DATA_DIR / 'processed' / 'example_multispeaker' / 'frames' / '000000.png'),
            score=0.9,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.zeros((5,2))
        )

    def test_crop_from_frame(self):
        cropped_image = self.detection_image.bb_crop()
        self.assertIsInstance(cropped_image, np.ndarray)

    def test_crop_from_video(self):
        cropped_image = self.detection_video.bb_crop()
        self.assertIsInstance(cropped_image, np.ndarray)

    def test_crop_multiple_from_video(self):
        for detection in self.detections_from_video:
            cropped_image = detection.bb_crop(self.vr)
            self.assertIsInstance(cropped_image, np.ndarray)

    def test_frame_center_from_frame(self):
        frame_center = self.detection_image.frame_center()
        frame = self.detection_image.frame()
        height, width = frame.shape[:2]
        self.assertIsInstance(frame_center, np.ndarray)
        self.assertEqual(frame_center.shape, (2,))
        self.assertEqual(frame_center[0], width // 2)
        self.assertEqual(frame_center[1], height // 2)

    def test_frame_center_from_video(self):
        frame_center = self.detection_video.frame_center(self.vr)
        frame = self.detection_video.frame(self.vr)
        height, width = frame.shape[:2]
        self.assertIsInstance(frame_center, np.ndarray)
        self.assertEqual(frame_center.shape, (2,))
        self.assertEqual(frame_center[0], width // 2)
        self.assertEqual(frame_center[1], height // 2)

    def test_equal(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source='path/to/frame1.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        detection2 = DetectionFromImage(
            frame_id=1,
            source='path/to/frame1.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        detection3 = DetectionFromImage(
            frame_id=1,
            source='path/to/frame1.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 300]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        self.assertEqual(detection1, detection2)
        self.assertNotEqual(detection1, detection3)
        self.assertNotEqual(detection2, detection3)


if __name__ == '__main__':
    unittest.main()