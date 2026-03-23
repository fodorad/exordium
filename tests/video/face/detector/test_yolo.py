"""Tests for YoloFaceV8Detector."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError

from exordium.video.core.detection import FrameDetections, VideoDetections
from exordium.video.face.detector.yolo import _DEFAULT_MODEL, YoloFaceV8Detector
from tests.fixtures import IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT


def _make_mock_result(n_boxes: int = 1, with_keypoints: bool = True) -> MagicMock:
    """Build a minimal mock of an Ultralytics Results object."""
    result = MagicMock()
    if n_boxes == 0:
        result.boxes = None
    else:
        result.boxes = MagicMock()
        result.boxes.xyxy = torch.rand(n_boxes, 4).clamp(0, 1) * 100
        result.boxes.conf = torch.rand(n_boxes)
        result.boxes.__len__ = MagicMock(return_value=n_boxes)
        if with_keypoints:
            result.keypoints = MagicMock()
            result.keypoints.xy = torch.rand(n_boxes, 5, 2) * 100
        else:
            result.keypoints = None
    return result


def _make_detector() -> YoloFaceV8Detector:
    """Instantiate YoloFaceV8Detector with ultralytics mocked out."""
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = MagicMock()
    with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
        det = YoloFaceV8Detector(device_id=None, conf=0.3, batch_size=4, verbose=False)
    det.model = MagicMock()
    return det


class TestYoloModelAvailability(unittest.TestCase):
    def test_model_exists_on_huggingface(self):
        """Check that the default YOLO model repo is reachable on HuggingFace."""
        try:
            info = model_info(_DEFAULT_MODEL)
            self.assertEqual(info.modelId, _DEFAULT_MODEL)
        except RepositoryNotFoundError:
            self.fail(f"Model '{_DEFAULT_MODEL}' not found on HuggingFace.")
        except Exception as e:
            self.fail(f"Could not reach HuggingFace to verify model: {e}")


class TestParseResults(unittest.TestCase):
    """_parse_results: tensor-native path (formerly _parse_results_tensor)."""

    def test_no_boxes_none_returns_empty(self):
        """boxes=None hits the `is not None` short-circuit."""
        out = YoloFaceV8Detector._parse_results([_make_mock_result(n_boxes=0)])
        self.assertEqual(out, [[]])

    def test_no_boxes_empty_object_returns_empty(self):
        """Non-None but empty Boxes (len==0) hits the `len(r.boxes)` falsy branch."""
        result = MagicMock()
        result.boxes = MagicMock()
        result.boxes.__len__ = MagicMock(return_value=0)
        out = YoloFaceV8Detector._parse_results([result])
        self.assertEqual(out, [[]])

    def test_boxes_with_keypoints_are_tensors(self):
        """Bounding boxes and landmarks must be torch.Tensor."""
        out = YoloFaceV8Detector._parse_results([_make_mock_result(n_boxes=2)])
        self.assertEqual(len(out[0]), 2)
        bb, lm, score = out[0][0]
        self.assertIsInstance(bb, torch.Tensor)
        self.assertEqual(bb.shape, (4,))
        self.assertIsInstance(lm, torch.Tensor)
        self.assertEqual(lm.shape, (5, 2))
        self.assertIsInstance(score, float)

    def test_boxes_no_keypoints_zero_tensor(self):
        """When keypoints is None, landmarks should be a zero tensor of shape (5, 2)."""
        out = YoloFaceV8Detector._parse_results(
            [_make_mock_result(n_boxes=1, with_keypoints=False)]
        )
        _, lm, _ = out[0][0]
        self.assertIsInstance(lm, torch.Tensor)
        self.assertEqual(lm.shape, (5, 2))
        self.assertTrue(torch.all(lm == 0))

    def test_multiple_images(self):
        r1 = _make_mock_result(n_boxes=1)
        r2 = _make_mock_result(n_boxes=3)
        out = YoloFaceV8Detector._parse_results([r1, r2])
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 1)
        self.assertEqual(len(out[1]), 3)


class TestRunDetector(unittest.TestCase):
    """run_detector: tensor input with internal RGB→BGR flip."""

    def setUp(self):
        self.detector = _make_detector()

    def test_calls_predict_and_returns_list(self):
        self.detector.model.predict.return_value = [_make_mock_result(n_boxes=1)]
        img = torch.zeros((3, 80, 80), dtype=torch.uint8)
        results = self.detector.run_detector([img])
        self.assertEqual(len(results), 1)
        self.detector.model.predict.assert_called_once()

    def test_bgr_flip_applied(self):
        """R channel (index 0 in CHW tensor) should appear at index 2 in HWC BGR numpy."""
        self.detector.model.predict.return_value = [_make_mock_result(n_boxes=0)]
        img = torch.zeros((3, 10, 10), dtype=torch.uint8)
        img[0, :, :] = 200  # R channel = 200
        self.detector.run_detector([img])
        bgr_arg = self.detector.model.predict.call_args[0][0][0]
        # After permute(1,2,0) and [:,:,::-1], R ends up at channel index 2
        self.assertEqual(int(bgr_arg[0, 0, 2]), 200)

    def test_no_detections_returns_empty_per_image(self):
        self.detector.model.predict.return_value = [_make_mock_result(n_boxes=0)]
        results = self.detector.run_detector([torch.zeros((3, 50, 50), dtype=torch.uint8)])
        self.assertEqual(results, [[]])

    def test_batch_of_images(self):
        self.detector.model.predict.return_value = [
            _make_mock_result(n_boxes=1),
            _make_mock_result(n_boxes=0),
            _make_mock_result(n_boxes=2),
        ]
        imgs = [torch.zeros((3, 40, 40), dtype=torch.uint8)] * 3
        results = self.detector.run_detector(imgs)
        self.assertEqual(len(results), 3)


class TestDetectVideo(unittest.TestCase):
    """detect_video: tensor-native video path."""

    def setUp(self):
        self.detector = _make_detector()

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.detector.detect_video("/nonexistent/path/clip.mp4")

    def test_returns_video_detections(self):
        def predict_side_effect(batch, **kwargs):
            return [_make_mock_result(n_boxes=1)] * len(batch)

        self.detector.model.predict.side_effect = predict_side_effect
        with tempfile.TemporaryDirectory() as d:
            result = self.detector.detect_video(
                VIDEO_MULTISPEAKER_SHORT,
                output_path=Path(d) / "out.vdet",
            )
        self.assertIsInstance(result, VideoDetections)
        self.assertGreater(len(result), 0)

    def test_detect_image_goes_through_run_detector(self):
        self.detector.model.predict.return_value = [_make_mock_result(n_boxes=1)]
        img = torch.zeros((3, 100, 100), dtype=torch.uint8)
        result = self.detector.detect_image(img, image_path=IMAGE_FACE)
        self.assertIsInstance(result, FrameDetections)


if __name__ == "__main__":
    unittest.main()
