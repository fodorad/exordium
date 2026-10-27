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
    boxes_from_table,
    boxes_from_tracks,
    gaze_table,
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

    def test_default_and_non_positive_fps_use_every_frame(self):
        boxes = _boxes(range(30), people=1)
        for fps in (None, -1, 0):
            out = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, boxes, fps=fps, batch_size=10)
            self.assertEqual(out["frame_ids"].tolist(), list(range(30)))

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

    def test_changed_boxes_do_not_reuse_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(batch_size=2, chunk_frames=4, chunk_dir=Path(tmp))
            self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, _boxes(range(4)), **kwargs)
            moved = {f: {0: BOX_B, 1: BOX_A} for f in range(4)}
            out = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, moved, **kwargs)
            self.assertEqual(len(list(Path(tmp).glob("*.st"))), 2)
            fresh = self.model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, moved)
            self.assertTrue(torch.equal(out["gaze_points"], fresh["gaze_points"]))


class TestGazeTable(unittest.TestCase):
    @staticmethod
    def _result() -> dict[str, torch.Tensor]:
        return {
            "frame_ids": torch.tensor([0, 0, 3]),
            "person_ids": torch.tensor([1, 2, 1]),
            "gaze_points": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "inout": torch.tensor([0.9, 0.1, 0.8]),
            "heatmaps": torch.zeros(3, 64, 64),
            "mass": torch.tensor([[0.0, 0.5], [0.2, 0.0], [0.0, 0.7]]),
            "total": torch.tensor([1.0, 2.0, 3.0]),
        }

    def test_one_row_per_frame_and_person(self):
        table = gaze_table(self._result())
        self.assertEqual(len(table), 3)
        self.assertEqual(table["frame_id"].tolist(), [0, 0, 3])
        self.assertEqual(table["person_id"].tolist(), [1, 2, 1])
        self.assertAlmostEqual(table["gaze_x"][2], 0.5)
        self.assertAlmostEqual(table["gaze_y"][2], 0.6)
        self.assertAlmostEqual(table["inout"][1], 0.1, places=5)

    def test_heatmaps_are_left_out(self):
        self.assertNotIn("heatmaps", gaze_table(self._result()).columns)

    def test_time_column_from_fps(self):
        table = gaze_table(self._result(), fps=30.0)
        self.assertAlmostEqual(table["time_s"][2], 0.1)
        self.assertNotIn("time_s", gaze_table(self._result()).columns)

    def test_matrix_columns_are_expanded_and_labelled(self):
        table = gaze_table(self._result(), labels={"mass": ["p1", "p2"]})
        self.assertAlmostEqual(table["mass_p1"][1], 0.2, places=5)
        self.assertAlmostEqual(table["mass_p2"][2], 0.7, places=5)
        self.assertIn("mass_0", gaze_table(self._result()).columns)

    def test_scalar_reducer_column_is_kept(self):
        self.assertEqual(gaze_table(self._result())["total"].tolist(), [1.0, 2.0, 3.0])

    def test_empty_result(self):
        table = gaze_table(
            {
                "frame_ids": torch.zeros(0, dtype=torch.long),
                "person_ids": torch.zeros(0, dtype=torch.long),
                "gaze_points": torch.zeros(0, 2),
            }
        )
        self.assertEqual(len(table), 0)
        self.assertIn("gaze_x", table.columns)


class TestBoxesFromTable(unittest.TestCase):
    ROWS = (
        "frame_id,person_id,x1,y1,x2,y2\n"
        "0,7,100,50,300,150\n"
        "0,9,500,100,700,300\n"
        "2,7,110,50,310,150\n"
    )

    def _csv(self, tmp: str, text: str | None = None) -> Path:
        path = Path(tmp) / "boxes.csv"
        path.write_text(self.ROWS if text is None else text)
        return path

    def test_reads_a_csv_in_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            boxes = boxes_from_table(self._csv(tmp), width=1000, height=500)
        self.assertEqual(sorted(boxes), [0, 2])
        self.assertEqual(sorted(boxes[0]), [7, 9])
        x1, y1, x2, y2 = boxes[0][7]
        self.assertAlmostEqual(x1, 0.1)
        self.assertAlmostEqual(y1, 0.1)
        self.assertAlmostEqual(x2, 0.3)
        self.assertAlmostEqual(y2, 0.3)

    def test_accepts_a_dataframe_and_expands(self):
        import pandas as pd

        table = pd.DataFrame(
            {"frame_id": [0], "person_id": [1], "x1": [400], "y1": [200], "x2": [600], "y2": [300]}
        )
        plain = boxes_from_table(table, 1000, 500)[0][1]
        wide = boxes_from_table(table, 1000, 500, scale=2.0)[0][1]
        self.assertLess(wide[0], plain[0])
        self.assertGreater(wide[3], plain[3])

    def test_xywh_format(self):
        import pandas as pd

        table = pd.DataFrame(
            {"frame_id": [0], "person_id": [1], "x": [100], "y": [50], "w": [200], "h": [100]}
        )
        box = boxes_from_table(table, 1000, 500, box_format="xywh")[0][1]
        self.assertAlmostEqual(box[2], 0.3)
        self.assertAlmostEqual(box[3], 0.3)

    def test_normalized_boxes_pass_through(self):
        import pandas as pd

        table = pd.DataFrame(
            {"frame_id": [4], "person_id": [1], "x1": [0.1], "y1": [0.2], "x2": [0.3], "y2": [0.4]}
        )
        box = boxes_from_table(table, 1000, 500, normalized=True)[4][1]
        self.assertEqual(box, (0.1, 0.2, 0.3, 0.4))

    def test_missing_column_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._csv(tmp, "frame_id,person_id,x1,y1,x2\n0,1,1,2,3\n")
            with self.assertRaisesRegex(ValueError, "y2"):
                boxes_from_table(path, 1000, 500)

    def test_duplicate_person_in_a_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._csv(tmp, self.ROWS + "0,7,1,1,2,2\n")
            with self.assertRaisesRegex(ValueError, "twice"):
                boxes_from_table(path, 1000, 500)

    def test_unknown_box_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                boxes_from_table(self._csv(tmp), 1000, 500, box_format="cxcywh")

    def test_feeds_video_to_gaze(self):
        model = GazelleWrapper("vitb14_inout", pretrained=PRETRAINED)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._csv(
                tmp, "frame_id,person_id,x1,y1,x2,y2\n0,3,100,50,300,150\n1,3,100,50,300,150\n"
            )
            boxes = boxes_from_table(path, 1280, 720)
        out = model.video_to_gaze(VIDEO_MULTISPEAKER_SHORT, boxes)
        self.assertEqual(out["frame_ids"].tolist(), [0, 1])
        self.assertEqual(out["person_ids"].tolist(), [3, 3])


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
