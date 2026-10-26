"""Tests for the Gaze-LLE gaze-target wrapper."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.core.detection import DetectionFromImage, Track
from exordium.video.face.gaze.gazelle import (
    DEFAULT_VARIANT,
    GAZELLE_VARIANTS,
    GazelleOutput,
    GazelleWrapper,
    boxes_from_tracks,
)
from tests.fixtures import (
    IMAGE_MULTISPEAKER,
    PRETRAINED,
    VIDEO_MULTISPEAKER_SHORT,
    ModelTestCase,
    hf_file_exists,
)

BOX_A = (0.1, 0.1, 0.3, 0.4)
BOX_B = (0.6, 0.2, 0.8, 0.5)


def _frames(batch: int = 2, h: int = 90, w: int = 160) -> torch.Tensor:
    return torch.randint(0, 255, (batch, 3, h, w), dtype=torch.uint8)


class TestGazelleWrapper(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GazelleWrapper("vitb14_inout", device_id=None, pretrained=PRETRAINED)

    def test_output_shapes_and_range(self):
        out = self.model(_frames(2), [[BOX_A, BOX_B], [BOX_A, BOX_B]])
        self.assertEqual(out.heatmap.shape, (2, 2, 64, 64))
        self.assertEqual(out.valid.shape, (2, 2))
        self.assertEqual(out.inout.shape, (2, 2))
        self.assertGreaterEqual(float(out.heatmap.min()), 0.0)
        self.assertLessEqual(float(out.heatmap.max()), 1.0)
        self.assertGreaterEqual(float(out.inout.min()), 0.0)
        self.assertLessEqual(float(out.inout.max()), 1.0)

    def test_variable_people_are_padded(self):
        out = self.model(_frames(2), [[BOX_A, BOX_B], [BOX_A]])
        self.assertEqual(out.heatmap.shape[:2], (2, 2))
        self.assertEqual(out.valid.tolist(), [[True, True], [True, False]])
        self.assertEqual(float(out.heatmap[1, 1].abs().sum()), 0.0)
        self.assertEqual(float(out.inout[1, 1]), 0.0)

    def test_frame_without_people(self):
        out = self.model(_frames(2), [[BOX_A], []])
        self.assertEqual(out.valid.tolist(), [[True], [False]])

    def test_no_people_at_all(self):
        out = self.model(_frames(1), [[]])
        self.assertEqual(out.heatmap.shape, (1, 0, 64, 64))

    def test_none_box_allowed(self):
        out = self.model(_frames(1), [[None]])
        self.assertEqual(out.heatmap.shape, (1, 1, 64, 64))

    def test_non_square_inputs(self):
        for h, w in [(60, 200), (300, 120)]:
            out = self.model(_frames(1, h, w), [[BOX_A]])
            self.assertEqual(out.heatmap.shape, (1, 1, 64, 64))

    def test_accepts_numpy_and_path(self):
        arr = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
        self.assertEqual(self.model(arr, [[BOX_A]]).heatmap.shape[0], 1)
        self.assertEqual(self.model(IMAGE_MULTISPEAKER, [[BOX_A]]).heatmap.shape[0], 1)

    def test_box_count_must_match_frames(self):
        with self.assertRaises(ValueError):
            self.model(_frames(2), [[BOX_A]])

    def test_out_of_range_box_is_clipped(self):
        out = self.model(_frames(1), [[(-0.5, -0.5, 1.5, 1.5)]])
        self.assertEqual(out.heatmap.shape, (1, 1, 64, 64))

    def test_deterministic(self):
        frames = _frames(1)
        a = self.model(frames, [[BOX_A]])
        b = self.model(frames, [[BOX_A]])
        self.assertTrue(torch.equal(a.heatmap, b.heatmap))

    def test_people_in_one_frame_do_not_leak_across_frames(self):
        frames = _frames(2)
        both = self.model(frames, [[BOX_A], [BOX_B]])
        alone = self.model(frames[1:], [[BOX_B]])
        self.assertTrue(torch.allclose(both.heatmap[1], alone.heatmap[0], atol=1e-4))

    def test_head_box_changes_prediction(self):
        frames = _frames(1)
        a = self.model(frames, [[BOX_A]])
        b = self.model(frames, [[BOX_B]])
        self.assertFalse(torch.allclose(a.heatmap, b.heatmap))

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            GazelleWrapper("vitx", pretrained=False)

    def test_variants_registry(self):
        self.assertIn(DEFAULT_VARIANT, GAZELLE_VARIANTS)
        self.assertEqual(len(GAZELLE_VARIANTS), 6)


def _boxes(frame_ids: range, people: int = 2) -> dict[int, dict[int, tuple]]:
    return {f: {0: BOX_A, 1: BOX_B}.copy() if people == 2 else {0: BOX_A} for f in frame_ids}


class TestGazelleVideo(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GazelleWrapper("vitb14_inout", device_id=None, pretrained=PRETRAINED)

    def test_rows_per_frame_and_person(self):
        out = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, _boxes(range(6)), batch_size=4)
        self.assertEqual(out["frame_ids"].tolist(), [f for f in range(6) for _ in (0, 1)])
        self.assertEqual(out["person_ids"].tolist(), [0, 1] * 6)
        self.assertEqual(out["gaze_points"].shape, (12, 2))
        self.assertEqual(out["inout"].shape, (12,))
        self.assertNotIn("heatmaps", out)

    def test_frames_without_boxes_are_skipped(self):
        boxes = {2: {0: BOX_A}, 5: {0: BOX_A}}
        out = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, boxes, batch_size=2)
        self.assertEqual(out["frame_ids"].tolist(), [2, 5])

    def test_empty_boxes_give_empty_result(self):
        out = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, {})
        self.assertEqual(out["frame_ids"].numel(), 0)
        self.assertEqual(out["gaze_points"].shape, (0, 2))

    def test_callable_boxes(self):
        out = self.model.video_to_gaze(
            VIDEO_MULTISPEAKER_SHORT, lambda f: {7: BOX_A} if f < 3 else {}, batch_size=2
        )
        self.assertEqual(out["frame_ids"].tolist(), [0, 1, 2])
        self.assertEqual(out["person_ids"].tolist(), [7, 7, 7])

    def test_fps_subsampling_keeps_original_frame_ids(self):
        out = self.model.video_to_gaze(
            VIDEO_MULTISPEAKER_SHORT, _boxes(range(30), people=1), fps=10, batch_size=4
        )
        ids = out["frame_ids"].tolist()
        self.assertIn(len(ids), (10, 11))  # clip is 29.97 fps, 1.001 s long
        self.assertEqual(ids[:3], [0, 3, 6])

    def test_store_heatmaps(self):
        out = self.model.video_to_gaze(
            VIDEO_MULTISPEAKER_SHORT, _boxes(range(3)), store_heatmaps=True
        )
        self.assertEqual(out["heatmaps"].shape, (6, 64, 64))

    def test_reducers(self):
        def reducer(heatmaps, frame_ids, person_ids):
            return heatmaps.flatten(1).sum(1)

        out = self.model.video_to_gaze(
            VIDEO_MULTISPEAKER_SHORT, _boxes(range(3)), reducers={"total": reducer}
        )
        self.assertEqual(out["total"].shape, (6,))

    def test_cached_result_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gaze.st"
            first = self.model.video_to_gaze(
                VIDEO_MULTISPEAKER_SHORT, _boxes(range(3)), output_path=path
            )
            self.assertTrue(path.exists())
            second = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, {}, output_path=path)
            self.assertTrue(torch.equal(first["gaze_points"], second["gaze_points"]))

    def test_interrupted_run_resumes_from_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks = Path(tmp) / "chunks"
            kwargs = dict(batch_size=2, chunk_frames=4, chunk_dir=chunks)
            full = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, _boxes(range(8)), **kwargs)
            saved = sorted(chunks.glob("*.st"))
            self.assertEqual(len(saved), 2)
            saved[1].unlink()  # simulate an interruption after the first chunk
            kept = saved[0].stat().st_mtime_ns
            resumed = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, _boxes(range(8)), **kwargs)
            self.assertEqual(saved[0].stat().st_mtime_ns, kept)
            self.assertTrue(torch.equal(full["gaze_points"], resumed["gaze_points"]))


class TestBoxesFromTracks(unittest.TestCase):
    def test_converts_to_normalized_and_expands(self):
        track = Track(3)
        for frame_id in (0, 2):
            track.add(
                DetectionFromImage(
                    frame_id=frame_id,
                    score=0.9,
                    bb_xywh=torch.tensor([100, 50, 200, 100]),
                    landmarks=torch.zeros(5, 2, dtype=torch.long),
                    source="x.png",
                )
            )
        boxes = boxes_from_tracks({3: track}, width=1000, height=500, scale=1.0)
        self.assertEqual(sorted(boxes), [0, 2])
        x1, y1, x2, y2 = boxes[0][3]
        self.assertAlmostEqual(x1, 0.1)
        self.assertAlmostEqual(y2, 0.3)
        wide = boxes_from_tracks({3: track}, width=1000, height=500, scale=2.0)[0][3]
        self.assertLess(wide[0], x1)


class TestGazelleNoInout(unittest.TestCase):
    def test_inout_is_none(self):
        model = GazelleWrapper("vitb14", pretrained=False)
        out = model(_frames(1), [[BOX_A]])
        self.assertIsNone(out.inout)
        self.assertEqual(out.heatmap.shape, (1, 1, 64, 64))


class TestGazelleOutput(unittest.TestCase):
    @staticmethod
    def _output() -> GazelleOutput:
        heatmap = torch.zeros(1, 2, 64, 64)
        heatmap[0, 0, 10, 40] = 1.0
        return GazelleOutput(
            heatmap=heatmap,
            valid=torch.tensor([[True, False]]),
            inout=torch.tensor([[0.9, 0.0]]),
        )

    def test_gaze_point(self):
        point = self._output().gaze_point()
        self.assertEqual(point.shape, (1, 2, 2))
        self.assertAlmostEqual(float(point[0, 0, 0]), 40.5 / 64)
        self.assertAlmostEqual(float(point[0, 0, 1]), 10.5 / 64)

    def test_mass_in_box(self):
        out = self._output()
        mass = out.mass_in_box((0.5, 0.0, 1.0, 0.5))
        self.assertEqual(mass.shape, (1, 2))
        self.assertAlmostEqual(float(mass[0, 0]), 1.0, places=5)
        self.assertEqual(float(mass[0, 1]), 0.0)


class TestGazelleWeightAvailability(unittest.TestCase):
    def test_all_checkpoints_mirrored(self):
        for spec in GAZELLE_VARIANTS.values():
            self.assertTrue(
                hf_file_exists("fodorad/exordium-weights", spec.weight_file),
                f"{spec.weight_file} not found in fodorad/exordium-weights",
            )


if __name__ == "__main__":
    unittest.main()
