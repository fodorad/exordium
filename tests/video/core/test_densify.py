"""Tests for scattering sparse per-detection features onto a dense frame grid."""

import unittest

import torch

from exordium.video.core.densify import densify


def _sparse(frame_ids: list[int], dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse (frame_ids, features) pair whose rows are identifiable.

    Row *i* is filled with the value ``frame_ids[i]``, so a test can assert a
    feature landed on the right timestep by reading the value back.
    """
    ids = torch.tensor(frame_ids, dtype=torch.long)
    features = torch.stack([torch.full((dim,), float(f)) for f in frame_ids])
    return ids, features


class TestDensifyShapeAndContract(unittest.TestCase):
    """The output must match MARLIN's frame_ids/features/mask contract."""

    def test_returns_the_three_contract_keys(self):
        ids, feats = _sparse([0, 2])
        out = densify(ids, feats, end_frame_id=4)
        self.assertEqual(set(out), {"frame_ids", "features", "mask"})

    def test_dense_shapes_follow_the_window(self):
        ids, feats = _sparse([1, 3])
        out = densify(ids, feats, start_frame_id=0, end_frame_id=10)
        self.assertEqual(out["features"].shape, (10, 4))
        self.assertEqual(out["mask"].shape, (10,))
        self.assertEqual(out["frame_ids"].shape, (10,))

    def test_frame_ids_are_absolute_not_window_offsets(self):
        # A window starting at 100 must report frames 100..104, not 0..4 — otherwise
        # the caller cannot align the window back to the source video.
        ids, feats = _sparse([101])
        out = densify(ids, feats, start_frame_id=100, end_frame_id=105)
        self.assertEqual(out["frame_ids"].tolist(), [100, 101, 102, 103, 104])

    def test_mask_dtype_is_bool(self):
        ids, feats = _sparse([0])
        self.assertEqual(densify(ids, feats, end_frame_id=2)["mask"].dtype, torch.bool)

    def test_feature_dtype_is_preserved(self):
        ids = torch.tensor([0], dtype=torch.long)
        feats = torch.ones(1, 4, dtype=torch.float64)
        self.assertEqual(densify(ids, feats, end_frame_id=3)["features"].dtype, torch.float64)


class TestDensifyPlacement(unittest.TestCase):
    """Features must land on their own frame, and nowhere else."""

    def test_features_land_on_their_own_frame(self):
        ids, feats = _sparse([0, 2, 5])
        out = densify(ids, feats, end_frame_id=6)
        for frame_id in (0, 2, 5):
            self.assertTrue(
                torch.equal(out["features"][frame_id], torch.full((4,), float(frame_id)))
            )

    def test_gaps_are_zero_filled_and_masked_false(self):
        ids, feats = _sparse([0, 2])
        out = densify(ids, feats, end_frame_id=4)
        self.assertEqual(out["mask"].tolist(), [True, False, True, False])
        for gap in (1, 3):
            self.assertTrue(torch.equal(out["features"][gap], torch.zeros(4)))

    def test_unsorted_frame_ids_still_land_correctly(self):
        # track_to_feature batches detections, so ordering is not guaranteed.
        ids, feats = _sparse([5, 0, 2])
        out = densify(ids, feats, end_frame_id=6)
        self.assertEqual(out["mask"].tolist(), [True, False, True, False, False, True])
        self.assertTrue(torch.equal(out["features"][5], torch.full((4,), 5.0)))

    def test_offset_window_places_features_relative_to_start(self):
        ids, feats = _sparse([102])
        out = densify(ids, feats, start_frame_id=100, end_frame_id=104)
        # Frame 102 is offset 2 within the window.
        self.assertEqual(out["mask"].tolist(), [False, False, True, False])
        self.assertTrue(torch.equal(out["features"][2], torch.full((4,), 102.0)))

    def test_fully_dense_track_masks_everything_true(self):
        ids, feats = _sparse([0, 1, 2])
        out = densify(ids, feats, end_frame_id=3)
        self.assertTrue(bool(out["mask"].all()))
        self.assertTrue(torch.equal(out["features"], feats))


class TestDensifyWindowClipping(unittest.TestCase):
    """A long track legitimately overruns a short window."""

    def test_detections_outside_the_window_are_dropped(self):
        ids, feats = _sparse([0, 5, 10])
        out = densify(ids, feats, start_frame_id=4, end_frame_id=7)
        self.assertEqual(out["features"].shape, (3, 4))
        self.assertEqual(out["mask"].tolist(), [False, True, False])  # only frame 5 survives
        self.assertTrue(torch.equal(out["features"][1], torch.full((4,), 5.0)))

    def test_window_entirely_outside_the_track_is_all_fill(self):
        ids, feats = _sparse([0, 1])
        out = densify(ids, feats, start_frame_id=50, end_frame_id=53)
        self.assertFalse(bool(out["mask"].any()))
        self.assertTrue(torch.equal(out["features"], torch.zeros(3, 4)))

    def test_end_frame_id_is_exclusive(self):
        # Half-open [start, end): a detection exactly at end_frame_id is outside.
        ids, feats = _sparse([3])
        out = densify(ids, feats, start_frame_id=0, end_frame_id=3)
        self.assertEqual(out["features"].shape[0], 3)
        self.assertFalse(bool(out["mask"].any()))

    def test_strict_raises_on_out_of_range_frame_id(self):
        ids, feats = _sparse([0, 99])
        with self.assertRaises(ValueError):
            densify(ids, feats, end_frame_id=10, strict=True)

    def test_strict_passes_when_all_ids_fit(self):
        ids, feats = _sparse([0, 2])
        out = densify(ids, feats, end_frame_id=10, strict=True)
        self.assertEqual(int(out["mask"].sum()), 2)


class TestDensifyFill(unittest.TestCase):
    """The fill value marks absence; it must be controllable."""

    def test_scalar_fill_is_broadcast(self):
        ids, feats = _sparse([1])
        out = densify(ids, feats, end_frame_id=3, fill=-1.0)
        self.assertTrue(torch.equal(out["features"][0], torch.full((4,), -1.0)))
        self.assertTrue(torch.equal(out["features"][2], torch.full((4,), -1.0)))

    def test_vector_fill_is_used_per_gap(self):
        ids, feats = _sparse([1])
        fill = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = densify(ids, feats, end_frame_id=3, fill=fill)
        self.assertTrue(torch.equal(out["features"][0], fill))
        self.assertTrue(torch.equal(out["features"][2], fill))

    def test_fill_does_not_overwrite_real_features(self):
        ids, feats = _sparse([1])
        out = densify(ids, feats, end_frame_id=3, fill=-1.0)
        self.assertTrue(torch.equal(out["features"][1], torch.full((4,), 1.0)))

    def test_vector_fill_rows_are_independent(self):
        # expand() returns a read-only broadcast view; if it were not cloned, writing
        # one row would write them all. Prove the rows are genuinely separate.
        ids, feats = _sparse([0])
        out = densify(ids, feats, end_frame_id=3, fill=torch.zeros(4))
        out["features"][1] = 7.0
        self.assertTrue(torch.equal(out["features"][2], torch.zeros(4)))

    def test_wrong_shape_fill_tensor_raises(self):
        ids, feats = _sparse([0], dim=4)
        with self.assertRaises(ValueError):
            densify(ids, feats, end_frame_id=3, fill=torch.zeros(7))


class TestDensifyEdgeCases(unittest.TestCase):
    def test_empty_track_yields_all_fill_and_empty_mask(self):
        ids = torch.tensor([], dtype=torch.long)
        feats = torch.zeros((0, 4))
        out = densify(ids, feats, end_frame_id=5)
        self.assertEqual(out["features"].shape, (5, 4))
        self.assertFalse(bool(out["mask"].any()))

    def test_empty_track_without_end_frame_id_yields_empty_grid(self):
        ids = torch.tensor([], dtype=torch.long)
        feats = torch.zeros((0, 4))
        out = densify(ids, feats)
        self.assertEqual(out["features"].shape, (0, 4))

    def test_end_frame_id_defaults_to_last_detection_plus_one(self):
        ids, feats = _sparse([0, 3])
        out = densify(ids, feats)
        self.assertEqual(out["features"].shape[0], 4)
        self.assertEqual(out["mask"].tolist(), [True, False, False, True])

    def test_zero_length_window_is_allowed(self):
        ids, feats = _sparse([0])
        out = densify(ids, feats, start_frame_id=2, end_frame_id=2)
        self.assertEqual(out["features"].shape, (0, 4))
        self.assertEqual(out["mask"].shape, (0,))

    def test_end_before_start_raises(self):
        ids, feats = _sparse([0])
        with self.assertRaises(ValueError):
            densify(ids, feats, start_frame_id=5, end_frame_id=2)

    def test_non_2d_features_raises(self):
        ids = torch.tensor([0], dtype=torch.long)
        with self.assertRaises(ValueError):
            densify(ids, torch.zeros(3), end_frame_id=2)

    def test_length_mismatch_raises(self):
        ids = torch.tensor([0, 1], dtype=torch.long)
        with self.assertRaises(ValueError):
            densify(ids, torch.zeros((3, 4)), end_frame_id=2)


if __name__ == "__main__":
    unittest.main()
