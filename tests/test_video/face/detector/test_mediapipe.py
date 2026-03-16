"""Tests for exordium.video.face.detector.mediapipe module."""

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from exordium.video.core.detection import FrameDetections, VideoDetections
from exordium.video.face import MediaPipeFaceDetector, align_face
from tests.fixtures import IMAGE_EMMA


class TestAlignFacePure(unittest.TestCase):
    """Tests for align_face and crop_eye_keep_ratio that need no model."""

    def test_align_face_invalid_landmarks_raises(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        bb_xyxy = np.array([0, 0, 100, 100])
        bad_landmarks = np.array([[10, 10], [20, 10]])  # wrong shape
        with self.assertRaises(Exception):
            align_face(image, bb_xyxy, bad_landmarks)


def _make_face_video(image_path: Path, output_path: Path, num_frames: int = 5) -> None:
    """Write a short MP4 from a single image for testing video detection."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    # Resize to 480p to keep file small
    scale = 480 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 10.0, (w, h))
    for _ in range(num_frames):
        writer.write(img)
    writer.release()


class FaceDetectorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.face_detector = MediaPipeFaceDetector()
        cls.TMP_DIR = Path(tempfile.mkdtemp())
        cls.face_video = cls.TMP_DIR / "face.mp4"
        _make_face_video(IMAGE_EMMA, cls.face_video, num_frames=5)

    @classmethod
    def tearDownClass(cls):
        if cls.TMP_DIR.exists():
            shutil.rmtree(cls.TMP_DIR)

    def test_mediapipe_detect_image_path(self):
        frame_detections = self.face_detector.detect_image_path(IMAGE_EMMA)
        self.assertIsInstance(frame_detections, FrameDetections)
        self.assertGreater(len(frame_detections), 0)

    def test_mediapipe_detect_image_no_face(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        frame_detections = self.face_detector.detect_image(image)
        self.assertIsInstance(frame_detections, FrameDetections)
        self.assertEqual(len(frame_detections), 0)

    def test_mediapipe_detect_video(self):
        video_detections = self.face_detector.detect_video(self.face_video)
        self.assertIsInstance(video_detections, VideoDetections)
        self.assertGreater(len(video_detections), 0)

    def test_mediapipe_detect_video_with_output(self):
        output_path = self.TMP_DIR / "test.vdet"
        video_detections = self.face_detector.detect_video(self.face_video, output_path=output_path)
        self.assertIsInstance(video_detections, VideoDetections)
        self.assertTrue(output_path.exists())
        loaded_video_detections = self.face_detector.detect_video(
            self.face_video, output_path=output_path
        )
        self.assertEqual(video_detections, loaded_video_detections)

    def test_mediapipe_detect_frames(self):
        """Test detecting faces from a directory of frame images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            img = cv2.imread(str(IMAGE_EMMA))
            for i in range(3):
                cv2.imwrite(str(frame_dir / f"{i:06d}.jpg"), img)
            video_detections = self.face_detector.detect_frame_dir(frame_dir)
        self.assertIsInstance(video_detections, VideoDetections)
        self.assertGreater(len(video_detections), 0)

    def test_mediapipe_detect_video_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.face_detector.detect_video("/nonexistent/path/video.mp4")

    def test_align_face(self):
        frame_detections = self.face_detector.detect_image_path(IMAGE_EMMA)
        det = frame_detections[0]
        output_aligned = align_face(det.frame(), det.bb_xyxy, det.landmarks)
        self.assertEqual(output_aligned["rotated_image"].ndim, 3)
        self.assertEqual(output_aligned["rotated_face"].ndim, 3)
        self.assertEqual(output_aligned["rotated_bb_xyxy"].shape, (4,))
        self.assertIsInstance(output_aligned["rotation_degree"], float)
        self.assertEqual(output_aligned["rotation_matrix"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
