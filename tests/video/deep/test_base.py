"""Tests for VisualModelWrapper base class via ClipWrapper."""

import pathlib
import shutil
import tempfile
import unittest

import torch

from exordium.video.core.detection import DetectionFactory
from exordium.video.core.io import clear_video_cache
from exordium.video.deep.base import VisualModelWrapper
from exordium.video.deep.clip import ClipWrapper
from tests.fixtures import (
    IMAGE_FACE,
    PRETRAINED,
    TEST_CLIP_MODEL,
    VIDEO_FI_PROTAGONIST,
    ModelTestCase,
)


class TestVisualModelWrapperBase(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ClipWrapper(model_name=TEST_CLIP_MODEL, device_id=None, pretrained=PRETRAINED)

    def test_dir_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                dest = pathlib.Path(tmp) / f"{i:06d}.jpg"
                shutil.copy(IMAGE_FACE, dest)
            img_paths = sorted(pathlib.Path(tmp).glob("*.jpg"))
            result = self.model.dir_to_feature(img_paths, batch_size=2)
            self.assertIsInstance(result, dict)
            self.assertIn("frame_ids", result)
            self.assertIn("features", result)
            feats = result["features"]
            ids = result["frame_ids"]
            self.assertIsInstance(feats, torch.Tensor)
            self.assertEqual(feats.ndim, 2)
            self.assertEqual(len(ids), feats.shape[0])


class TestCropsForBatchedDecode(unittest.TestCase):
    """``_crops_for`` batches the decode; it must not change a single pixel.

    The per-detection ``crop()`` path is the reference. These compare against it
    directly rather than against extracted features, because batching alters float
    accumulation in the model (~1e-7) and would mask a real pixel-level defect.
    """

    def setUp(self):
        clear_video_cache()

    def tearDown(self):
        clear_video_cache()

    def _video_det(self, frame_id: int):
        return DetectionFactory.create_detection(
            frame_id=frame_id,
            source=str(VIDEO_FI_PROTAGONIST),
            score=0.9,
            bb_xywh=torch.tensor([400, 150, 220, 220], dtype=torch.long),
            landmarks=torch.zeros((5, 2), dtype=torch.long),
        )

    def _image_det(self, frame_id: int):
        return DetectionFactory.create_detection(
            frame_id=frame_id,
            source=str(IMAGE_FACE),
            score=0.9,
            bb_xywh=torch.tensor([50, 50, 60, 60], dtype=torch.long),
            landmarks=torch.zeros((5, 2), dtype=torch.long),
        )

    def _assert_matches_reference(self, detections):
        clear_video_cache()
        expected = [d.crop(square=True, extra_space=1.5) for d in detections]
        clear_video_cache()
        actual = VisualModelWrapper._crops_for(detections)
        self.assertEqual(len(actual), len(expected))
        for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
            self.assertTrue(torch.equal(got, want), f"crop {i} differs")

    def test_sequential_track(self):
        self._assert_matches_reference([self._video_det(i) for i in range(0, 40, 4)])

    def test_out_of_order_track(self):
        # A backward seek makes the decoder restart from the preceding keyframe, so
        # this is both the slow case and the one most likely to return a stale frame.
        self._assert_matches_reference([self._video_det(i) for i in (30, 2, 17, 2, 39, 0)])

    def test_non_video_detections_fall_back(self):
        self._assert_matches_reference([self._image_det(i) for i in range(3)])

    def test_mixed_sources_stay_aligned(self):
        # Video detections are decoded grouped by source and scattered back into place;
        # a misplaced crop here would silently pair a face with the wrong frame.
        self._assert_matches_reference(
            [
                self._image_det(0),
                self._video_det(11),
                self._image_det(1),
                self._video_det(3),
                self._video_det(25),
            ]
        )

    def test_empty_input(self):
        self.assertEqual(VisualModelWrapper._crops_for([]), [])


if __name__ == "__main__":
    unittest.main()
