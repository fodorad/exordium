"""Tests for exordium.video.face.detector.base: FaceDetector utility methods."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from exordium.video.core.detection import FrameDetections
from exordium.video.face.detector.base import FaceDetector as BaseDetector
from tests.fixtures import IMAGE_FACE


def _make_detector():
    """Instantiate YoloFace11Detector with ultralytics mocked out."""
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = MagicMock()
    with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
        from exordium.video.face.detector.yolo11 import YoloFace11Detector

        det = YoloFace11Detector.__new__(YoloFace11Detector)
        det.batch_size = 16
        det.verbose = False
        det.conf = 0.25
        det.device = torch.device("cpu")
        det.model = MagicMock()
    return det


class TestBuildFrameDetectionsTensorInputs(unittest.TestCase):
    def test_torch_tensor_landmarks_rounded(self):
        """run_detector returns tensors; _build_frame_detections calls .round().long() directly."""
        bb_xyxy = torch.tensor([10.0, 20.0, 90.0, 100.0], dtype=torch.float32)
        landmarks_tensor = torch.tensor(
            [[30.6, 40.3], [60.2, 70.8], [50.0, 55.0], [35.0, 80.0], [65.0, 80.0]],
            dtype=torch.float32,
        )
        score = 0.92
        fd = BaseDetector._build_frame_detections(
            [(bb_xyxy, landmarks_tensor, score)],
            frame_id=0,
            source=str(IMAGE_FACE),
        )
        self.assertIsInstance(fd, FrameDetections)
        self.assertEqual(len(fd), 1)

    def test_empty_detections_returns_empty_frame_detections(self):
        """Empty detection list produces a FrameDetections with zero entries."""
        fd = BaseDetector._build_frame_detections([], frame_id=0, source="test")
        self.assertIsInstance(fd, FrameDetections)
        self.assertEqual(len(fd), 0)


def _make_mock_result(n_boxes: int = 0):
    """Return a MagicMock mimicking an Ultralytics Results object."""
    r = MagicMock()
    if n_boxes == 0:
        r.boxes = None
    else:
        r.boxes.xyxy = torch.zeros(n_boxes, 4, dtype=torch.float32)
        r.boxes.conf = torch.ones(n_boxes, dtype=torch.float32)
        r.boxes.__len__ = MagicMock(return_value=n_boxes)
        r.keypoints.xy = torch.zeros(n_boxes, 5, 2, dtype=torch.float32)
    return r


class TestDetectImageAcceptsNumpyAndTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = _make_detector()
        cls.detector.model.predict.return_value = [_make_mock_result(0)]

    def test_detect_image_numpy_input(self):
        """detect_image should accept (H, W, 3) uint8 numpy arrays."""
        import numpy as np

        cls = self.__class__
        cls.detector.model.predict.return_value = [_make_mock_result(0)]
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = cls.detector.detect_image(img, image_path=str(IMAGE_FACE))
        self.assertIsInstance(result, FrameDetections)

    def test_detect_image_tensor_input(self):
        """detect_image should accept (3, H, W) uint8 torch tensors."""
        cls = self.__class__
        cls.detector.model.predict.return_value = [_make_mock_result(0)]
        img = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
        result = cls.detector.detect_image(img, image_path=str(IMAGE_FACE))
        self.assertIsInstance(result, FrameDetections)


class TestDetectFrameDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = _make_detector()
        cls.detector.model.predict.return_value = [_make_mock_result(0)]

    def test_detect_frame_dir_returns_video_detections(self):
        """detect_frame_dir → iterate frame files as tensors, detect, add."""
        from exordium.video.core.detection import VideoDetections

        self.detector.model.predict.return_value = [_make_mock_result(0)] * 3
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                shutil.copy(IMAGE_FACE, Path(tmpdir) / f"{i:06d}.jpg")
            result = self.detector.detect_frame_dir(tmpdir)

        self.assertIsInstance(result, VideoDetections)


if __name__ == "__main__":
    unittest.main()
