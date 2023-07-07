import unittest
from pathlib import Path
import os
import cv2
import numpy as np
from exordium.video.detection import FaceDetector, FrameDetections, VideoDetections

RESOLUTION = '720p'
FRAME_PATH = f'data/processed/frames/multispeaker_{RESOLUTION}/000000.png'
FRAME_DIR = f'data/processed/frames/9KAqOrdiZ4I.001'
VIDEO_PATH = f'data/videos/9KAqOrdiZ4I.001.mp4'
assert Path(FRAME_PATH).exists() and Path(FRAME_DIR).is_dir() and Path(VIDEO_PATH).exists()


class FaceDetectorTestCase(unittest.TestCase):

    def setUp(self):
        self.face_detector = FaceDetector(verbose=False)

    def test_detect_image(self):
        frame_detections = self.face_detector.detect_image(FRAME_PATH)
        self.assertIsInstance(frame_detections, FrameDetections)
        self.assertEqual(len(frame_detections), 3)

    def test_detect_image_with_no_face(self):
        image = np.zeros(shape=(200,300,3))
        frame_detections = self.face_detector.detect_image(image)
        self.assertIsInstance(frame_detections, FrameDetections)
        self.assertEqual(len(frame_detections), 0)

    def test_iterate_folder(self):
        n_frames = len(list(Path(FRAME_DIR).iterdir()))
        video_detections = self.face_detector.iterate_folder(FRAME_DIR)
        self.assertIsInstance(video_detections, VideoDetections)
        self.assertEqual(len(video_detections), n_frames)

    def test_detect_video(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_detections = self.face_detector.detect_video(VIDEO_PATH)
        self.assertIsInstance(video_detections, VideoDetections)
        self.assertEqual(len(video_detections), n_frames)

    def test_detect_video_with_output(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        output_path = 'test.vdet'
        if Path(output_path).exists(): os.remove(output_path)
        video_detections = self.face_detector.detect_video(VIDEO_PATH, output_path=output_path)
        self.assertIsInstance(video_detections, VideoDetections)
        self.assertEqual(len(video_detections), n_frames)
        loaded_video_detections = self.face_detector.detect_video(VIDEO_PATH, output_path=output_path)
        self.assertEqual(video_detections, loaded_video_detections)
        os.remove(output_path)

if __name__ == '__main__':
    unittest.main()
