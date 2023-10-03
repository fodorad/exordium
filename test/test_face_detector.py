import unittest
import os
import cv2
import numpy as np
from pathlib import Path
from exordium import DATA_DIR, EXAMPLE_VIDEO_PATH
from exordium.video.facedetector import RetinaFaceDetector, FrameDetections, VideoDetections

FRAME_PATH = DATA_DIR / 'processed' / 'example_multispeaker' / 'frames' / '000000.png'
FRAME_DIR = DATA_DIR / 'processed' / 'example_multispeaker' / 'frames'


class FaceDetectorTestCase(unittest.TestCase):

    def setUp(self):
        self.face_detector = RetinaFaceDetector()

    def test_retinaface_detect_image_path(self):
        frame_detections = self.face_detector.detect_image_path(FRAME_PATH)
        self.assertIsInstance(frame_detections, FrameDetections)
        self.assertEqual(len(frame_detections), 3)

    def test_retinaface_detect_image_no_face(self):
        image = np.zeros(shape=(200,300,3))
        frame_detections = self.face_detector.detect_image(image)
        self.assertIsInstance(frame_detections, FrameDetections)
        self.assertEqual(len(frame_detections), 0)

    def test_retinaface_detect_frame_dir(self):
        n_frames = len(list(FRAME_DIR.iterdir()))
        video_detections = self.face_detector.detect_frame_dir(FRAME_DIR)
        self.assertIsInstance(video_detections, VideoDetections)
        # last frame of the example video is black
        self.assertEqual(len(video_detections), n_frames - 1)

    def test_retinaface_detect_video(self):
        cap = cv2.VideoCapture(str(EXAMPLE_VIDEO_PATH))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_detections = self.face_detector.detect_video(EXAMPLE_VIDEO_PATH)
        self.assertIsInstance(video_detections, VideoDetections)
        # last frame of the example video is black
        self.assertEqual(len(video_detections), n_frames - 1)

    def test_retinaface_detect_video_with_output(self):
        output_path = DATA_DIR / 'test.vdet'
        if Path(output_path).exists(): os.remove(output_path)
        video_detections = self.face_detector.detect_video(EXAMPLE_VIDEO_PATH, output_path=output_path)
        self.assertIsInstance(video_detections, VideoDetections)
        loaded_video_detections = self.face_detector.detect_video(EXAMPLE_VIDEO_PATH, output_path=output_path)
        self.assertEqual(video_detections, loaded_video_detections)
        os.remove(output_path)


if __name__ == '__main__':
    unittest.main()