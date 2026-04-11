"""Tests for MarlinWrapper video-clip feature extractor."""

import math
import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import torch

from exordium.utils.padding import fill_gaps_with_repeat, repeat_pad_time_dim
from exordium.video.deep.marlin import (
    _DEFAULT_MARLIN_MODEL,
    _FEATURE_DIMS,
    _HF_REPO_IDS,
    _TIME_DIM,
    MarlinWrapper,
    _frames_to_clips,
)
from tests.fixtures import IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT, hf_repo_exists


class TestModuleConstants(unittest.TestCase):
    def test_constants_consistent(self):
        self.assertEqual(set(_HF_REPO_IDS.keys()), set(_FEATURE_DIMS.keys()))
        self.assertIn(_DEFAULT_MARLIN_MODEL, _HF_REPO_IDS)

    def test_clip_frames_is_16(self):
        self.assertEqual(_TIME_DIM, 16)


class TestFramesToClips(unittest.TestCase):
    def test_exact_16_frames(self):
        frames = torch.randint(0, 255, (16, 3, 64, 64), dtype=torch.uint8)
        clips, starts = _frames_to_clips(frames)
        self.assertEqual(clips.shape, (1, 3, 16, 64, 64))
        self.assertEqual(starts, [0])

    def test_short_input_padded(self):
        frames = torch.randint(0, 255, (5, 3, 64, 64), dtype=torch.uint8)
        clips, starts = _frames_to_clips(frames)
        self.assertEqual(clips.shape, (1, 3, 16, 64, 64))
        self.assertEqual(starts, [0])

    def test_multiple_clips_with_stride(self):
        frames = torch.randint(0, 255, (32, 3, 64, 64), dtype=torch.uint8)
        clips, starts = _frames_to_clips(frames, stride=8)
        # starts: 0, 8, 16
        self.assertEqual(clips.shape[0], 3)
        self.assertEqual(clips.shape[1:], (3, 16, 64, 64))
        self.assertEqual(starts, [0, 8, 16])

    def test_non_overlapping_clips(self):
        frames = torch.randint(0, 255, (48, 3, 64, 64), dtype=torch.uint8)
        clips, starts = _frames_to_clips(frames, stride=16)
        self.assertEqual(clips.shape[0], 3)
        self.assertEqual(starts, [0, 16, 32])

    def test_single_frame_padded(self):
        frames = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
        clips, starts = _frames_to_clips(frames)
        self.assertEqual(clips.shape, (1, 3, 16, 64, 64))


class TestRepeatPadTimeDim(unittest.TestCase):
    def test_no_padding_needed(self):
        x = torch.tensor([[1, 2], [3, 4]])
        result = repeat_pad_time_dim(x, 2)
        self.assertTrue(torch.equal(result, x))

    def test_longer_than_target(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        result = repeat_pad_time_dim(x, 2)
        self.assertTrue(torch.equal(result, x))

    def test_padding(self):
        x = torch.tensor([[1, 2], [3, 4]])
        result = repeat_pad_time_dim(x, 5)
        self.assertEqual(result.shape, (5, 2))
        # Last 3 rows should be copies of [3, 4]
        for i in range(2, 5):
            self.assertTrue(torch.equal(result[i], torch.tensor([3, 4])))

    def test_zero_frames_raises(self):
        with self.assertRaises(ValueError):
            repeat_pad_time_dim(torch.empty(0, 3), 5)

    def test_high_dim_tensor(self):
        x = torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
        result = repeat_pad_time_dim(x, 16)
        self.assertEqual(result.shape, (16, 3, 64, 64))
        self.assertTrue(torch.equal(result[-1], x[-1]))


class TestFillGapsWithRepeat(unittest.TestCase):
    def test_no_gaps(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = fill_gaps_with_repeat(x)
        self.assertTrue(torch.equal(result, x))

    def test_leading_zeros_backfilled(self):
        x = torch.zeros(4, 2)
        x[2] = torch.tensor([1.0, 2.0])
        result = fill_gaps_with_repeat(x)
        self.assertTrue(torch.equal(result[0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result[1], torch.tensor([1.0, 2.0])))

    def test_forward_fill(self):
        x = torch.zeros(4, 2)
        x[0] = torch.tensor([1.0, 2.0])
        x[3] = torch.tensor([5.0, 6.0])
        result = fill_gaps_with_repeat(x)
        self.assertTrue(torch.equal(result[1], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result[2], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result[3], torch.tensor([5.0, 6.0])))

    def test_all_zeros_raises(self):
        with self.assertRaises(ValueError):
            fill_gaps_with_repeat(torch.zeros(3, 2))

    def test_empty_tensor_returns_empty(self):
        x = torch.empty(0, 2)
        result = fill_gaps_with_repeat(x)
        self.assertEqual(result.shape, (0, 2))

    def test_explicit_valid_mask(self):
        x = torch.ones(4, 2)
        mask = torch.tensor([True, False, False, True])
        result = fill_gaps_with_repeat(x, valid_mask=mask)
        self.assertEqual(result.shape, (4, 2))


class TestMarlinWrapperInit(unittest.TestCase):
    def test_invalid_model_name_raises(self):
        with self.assertRaises(ValueError):
            MarlinWrapper(model_name="xlarge")


class TestMarlinWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MarlinWrapper(model_name="base", device_id=None)

    def test_feature_dim_attribute(self):
        self.assertEqual(self.model.feature_dim, _FEATURE_DIMS["base"])

    def test_from_uint8_tensor(self):
        clip = torch.randint(0, 255, (1, 3, 16, 112, 112), dtype=torch.uint8)
        out = self.model(clip)
        self.assertEqual(out.shape, (1, _FEATURE_DIMS["base"]))

    def test_from_float_tensor(self):
        clip = torch.rand(1, 3, 16, 224, 224)
        with torch.inference_mode():
            out = self.model.inference(clip.to(self.model.device))
        self.assertEqual(out.shape, (1, _FEATURE_DIMS["base"]))

    def test_batch_of_clips(self):
        clips = torch.randint(0, 255, (2, 3, 16, 224, 224), dtype=torch.uint8)
        out = self.model(clips)
        self.assertEqual(out.shape, (2, _FEATURE_DIMS["base"]))

    def test_single_clip_no_batch_dim(self):
        clip = torch.randint(0, 255, (3, 16, 112, 112), dtype=torch.uint8)
        out = self.model(clip)
        self.assertEqual(out.shape, (1, _FEATURE_DIMS["base"]))

    def test_from_numpy_array(self):
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        out = self.model(clip)
        self.assertEqual(out.shape, (1, _FEATURE_DIMS["base"]))

    def test_from_numpy_batch(self):
        clips = np.random.randint(0, 255, (2, 16, 64, 64, 3), dtype=np.uint8)
        out = self.model(clips)
        self.assertEqual(out.shape, (2, _FEATURE_DIMS["base"]))

    def test_preprocess_output_shape(self):
        clip = torch.randint(0, 255, (2, 3, 16, 128, 128), dtype=torch.uint8)
        preprocessed = self.model.preprocess(clip)
        self.assertEqual(preprocessed.shape, (2, 3, 16, 224, 224))
        self.assertEqual(preprocessed.dtype, torch.float32)

    def test_preprocess_value_range(self):
        clip = torch.randint(0, 255, (1, 3, 16, 224, 224), dtype=torch.uint8)
        preprocessed = self.model.preprocess(clip)
        self.assertGreaterEqual(preprocessed.min().item(), 0.0)
        self.assertLessEqual(preprocessed.max().item(), 1.0)

    def test_invalid_input_raises(self):
        with self.assertRaises(TypeError):
            self.model(["not", "a", "tensor"])

    def test_wrong_ndim_raises(self):
        with self.assertRaises(ValueError):
            self.model(torch.randint(0, 255, (16, 224, 224), dtype=torch.uint8))

    def test_wrong_channels_raises(self):
        with self.assertRaises(ValueError):
            self.model(torch.randint(0, 255, (1, 1, 16, 224, 224), dtype=torch.uint8))

    def test_dir_to_feature_face_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(20):
                shutil.copy(IMAGE_FACE, pathlib.Path(tmp) / f"{i:06d}.jpg")
            result = self.model.dir_to_feature(
                sorted(pathlib.Path(tmp).glob("*.jpg")),
                face_crops=True,
            )
            self.assertIn("features", result)
            self.assertIn("frame_ids", result)
            self.assertEqual(result["features"].ndim, 2)
            self.assertEqual(result["features"].shape[1], _FEATURE_DIMS["base"])
            self.assertGreater(result["features"].shape[0], 0)

    def test_dir_to_feature_empty(self):
        result = self.model.dir_to_feature([])
        self.assertEqual(result["features"].shape, (0, _FEATURE_DIMS["base"]))
        self.assertEqual(result["frame_ids"].shape, (0,))

    def test_video_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                output_path=pathlib.Path(tmp_dir) / "marlin.st",
            )
            self.assertIn("features", result)
            self.assertIn("frame_ids", result)
            self.assertIn("mask", result)
            self.assertEqual(result["features"].shape[1], _FEATURE_DIMS["base"])
            self.assertEqual(result["features"].shape[0], result["frame_ids"].shape[0])
            self.assertEqual(result["features"].shape[0], result["mask"].shape[0])
            self.assertEqual(result["mask"].dtype, torch.bool)
            # At least some windows should have detections
            self.assertTrue(result["mask"].any())

    def test_video_to_feature_window_grid(self):
        """Verify that output covers the full video timeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from exordium.video.core.io import get_video_metadata

            meta = get_video_metadata(VIDEO_MULTISPEAKER_SHORT)
            num_frames = meta["num_frames"]
            stride = _TIME_DIM
            expected_windows = math.ceil(num_frames / stride)

            result = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                stride=stride,
                output_path=pathlib.Path(tmp_dir) / "marlin_grid.st",
            )
            self.assertEqual(result["features"].shape[0], expected_windows)
            self.assertEqual(result["frame_ids"][0].item(), 0)
            # Frame IDs should be evenly spaced
            diffs = result["frame_ids"][1:] - result["frame_ids"][:-1]
            self.assertTrue((diffs == stride).all())

    def test_video_to_feature_zero_features_are_masked(self):
        """Windows with mask=False should have zero feature vectors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                output_path=pathlib.Path(tmp_dir) / "marlin_mask.st",
            )
            unmasked = ~result["mask"]
            if unmasked.any():
                zero_feats = result["features"][unmasked]
                self.assertTrue((zero_feats == 0).all())


class TestMarlinWeightAvailability(unittest.TestCase):
    def test_marlin_base_repo(self):
        self.assertTrue(hf_repo_exists("ControlNet/marlin_vit_base_ytf"))


if __name__ == "__main__":
    unittest.main()
