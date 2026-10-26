"""Tests for StreamingTracker: per-frame tracking with a provisional protagonist."""

import math
import random
import unittest

import torch

from exordium.video.core.detection import (
    DetectionFactory,
    FaceIdTracker,
    FrameDetections,
    IouTracker,
    VideoDetections,
)
from exordium.video.core.streaming import StreamingTracker, StreamUpdate
from tests.fixtures import IMAGE_FACE


def _det(frame_id: int, x: int, y: int, size: int = 80, score: float = 0.95):
    return DetectionFactory.create_detection(
        frame_id=frame_id,
        source=str(IMAGE_FACE),
        score=score,
        bb_xywh=torch.tensor([x, y, size, size], dtype=torch.long),
        landmarks=torch.zeros((5, 2), dtype=torch.long),
    )


def _frame(frame_id: int, faces: list[tuple[int, int, int]]) -> FrameDetections:
    """One frame with a face per ``(x, y, size)``."""
    fd = FrameDetections()
    for x, y, size in faces:
        fd.add(_det(frame_id, x, y, size))
    return fd


# Two well-separated faces: A (left, big) and B (right, small).
A = (10, 10, 120)
B = (600, 400, 40)


def _push_all(streamer: StreamingTracker, frames: list[FrameDetections]) -> list[StreamUpdate]:
    return [streamer.push(fd) for fd in frames]


class TestAssignments(unittest.TestCase):
    def test_assigns_consistent_track_ids(self):
        streamer = StreamingTracker(IouTracker())
        first = streamer.push(_frame(0, [A, B]))
        second = streamer.push(_frame(1, [A, B]))
        self.assertEqual(first.frame_id, 0)
        self.assertEqual([t for t, _ in first.assignments], [t for t, _ in second.assignments])
        self.assertEqual(len({t for t, _ in first.assignments}), 2)

    def test_low_score_detections_are_ignored(self):
        streamer = StreamingTracker(IouTracker(), min_score=0.7)
        fd = FrameDetections()
        fd.add(_det(0, 10, 10, score=0.3))
        update = streamer.push(fd)
        self.assertEqual(update.assignments, [])
        self.assertIsNone(update.protagonist_id)

    def test_empty_frame_needs_explicit_id(self):
        streamer = StreamingTracker(IouTracker())
        with self.assertRaises(ValueError):
            streamer.push(FrameDetections())
        update = streamer.push(FrameDetections(), frame_id=5)
        self.assertEqual(update.frame_id, 5)

    def test_frame_ids_must_increase(self):
        streamer = StreamingTracker(IouTracker())
        streamer.push(_frame(3, [A]))
        with self.assertRaises(ValueError):
            streamer.push(_frame(3, [A]))
        with self.assertRaises(ValueError):
            streamer.push(_frame(2, [A]))


class TestProtagonist(unittest.TestCase):
    def test_no_tracks_no_protagonist(self):
        update = StreamingTracker(IouTracker()).push(FrameDetections(), frame_id=0)
        self.assertIsNone(update.protagonist_id)
        self.assertFalse(update.stable)
        self.assertEqual(update.scores, {})

    def test_largest_longest_face_leads(self):
        streamer = StreamingTracker(IouTracker())
        updates = _push_all(streamer, [_frame(i, [A, B]) for i in range(10)])
        last = updates[-1]
        self.assertEqual(last.protagonist_id, last.assignments[0][0])
        self.assertAlmostEqual(last.scores[last.protagonist_id], 10 * 120)
        self.assertEqual(last.protagonist_detection.bb_xywh[2], 120)

    def test_margin_is_leader_over_runner_up(self):
        streamer = StreamingTracker(IouTracker())
        last = _push_all(streamer, [_frame(i, [A, B]) for i in range(10)])[-1]
        self.assertAlmostEqual(last.margin, (10 * 120) / (10 * 40))

    def test_single_track_has_infinite_margin(self):
        update = StreamingTracker(IouTracker()).push(_frame(0, [A]))
        self.assertTrue(math.isinf(update.margin))

    def test_protagonist_detection_is_none_when_absent_this_frame(self):
        streamer = StreamingTracker(IouTracker())
        streamer.push(_frame(0, [A]))
        update = streamer.push(FrameDetections(), frame_id=1)
        self.assertIsNotNone(update.protagonist_id)
        self.assertIsNone(update.protagonist_detection)

    def test_hysteresis_delays_a_switch(self):
        # A leads for 10 frames, then only B (same size) appears: B needs more than
        # 1.1x A's score before it takes over.
        streamer = StreamingTracker(IouTracker(), switch_ratio=1.1, forget_after=1000)
        updates = _push_all(streamer, [_frame(i, [A]) for i in range(10)])
        leader = updates[-1].protagonist_id
        same_size_b = (600, 400, 120)
        later = _push_all(streamer, [_frame(i, [same_size_b]) for i in range(10, 22)])
        self.assertEqual(later[9].protagonist_id, leader)  # 10 vs 10 frames: tie
        self.assertEqual(later[-1].protagonist_id, later[-1].assignments[0][0])  # 12 > 1.1 * 10

    def test_switch_ratio_one_follows_the_argmax(self):
        streamer = StreamingTracker(IouTracker(), switch_ratio=1.0, forget_after=1000)
        _push_all(streamer, [_frame(i, [A]) for i in range(5)])
        later = _push_all(streamer, [_frame(i, [(600, 400, 120)]) for i in range(5, 11)])
        self.assertEqual(later[-1].protagonist_id, later[-1].assignments[0][0])

    def test_decay_lets_a_newcomer_overtake_an_old_leader(self):
        keep = StreamingTracker(IouTracker(), decay=1.0, switch_ratio=1.0, forget_after=10_000)
        fade = StreamingTracker(IouTracker(), decay=0.9, switch_ratio=1.0, forget_after=10_000)
        frames = [_frame(i, [A]) for i in range(40)]
        frames += [_frame(i, [(600, 400, 120)]) for i in range(40, 60)]
        old = _push_all(keep, frames)[-1]
        new = _push_all(fade, frames)[-1]
        self.assertNotEqual(old.protagonist_id, old.assignments[0][0])
        self.assertEqual(new.protagonist_id, new.assignments[0][0])


class TestStability(unittest.TestCase):
    def _streamer(self) -> StreamingTracker:
        return StreamingTracker(
            IouTracker(), min_track_frames=5, min_stable_frames=8, min_margin=2.0
        )

    def test_becomes_stable_after_enough_frames(self):
        updates = _push_all(self._streamer(), [_frame(i, [A, B]) for i in range(12)])
        flags = [u.stable for u in updates]
        self.assertFalse(flags[0])
        self.assertTrue(flags[-1])
        first_stable = flags.index(True)
        self.assertGreaterEqual(updates[first_stable].stable_for, 8)
        self.assertTrue(all(flags[first_stable:]))

    def test_close_race_is_not_stable(self):
        streamer = self._streamer()
        updates = _push_all(streamer, [_frame(i, [A, (600, 400, 110)]) for i in range(30)])
        self.assertFalse(updates[-1].stable)  # margin ~1.09 < 2.0

    def test_leader_change_resets_stability(self):
        streamer = StreamingTracker(
            IouTracker(),
            switch_ratio=1.0,
            min_track_frames=3,
            min_stable_frames=5,
            min_margin=1.0,
            forget_after=10_000,
        )
        _push_all(streamer, [_frame(i, [A]) for i in range(10)])
        updates = _push_all(streamer, [_frame(i, [(600, 400, 120)]) for i in range(10, 25)])
        switch = next(
            i for i, u in enumerate(updates) if u.protagonist_id != updates[0].protagonist_id
        )
        self.assertEqual(updates[switch].stable_for, 0)
        self.assertFalse(updates[switch].stable)


class TestMemory(unittest.TestCase):
    def test_window_bounds_detections_but_not_scores(self):
        streamer = StreamingTracker(IouTracker(), window=20)
        last = _push_all(streamer, [_frame(i, [A]) for i in range(100)])[-1]
        track = next(iter(streamer.tracker.tracks.values()))
        self.assertEqual(len(track), 20)
        self.assertEqual(track.last_detection().frame_id, 99)
        self.assertAlmostEqual(last.scores[last.protagonist_id], 100 * 120)

    def test_absent_tracks_are_forgotten(self):
        streamer = StreamingTracker(IouTracker(), forget_after=10)
        _push_all(streamer, [_frame(i, [A, B]) for i in range(5)])
        updates = _push_all(streamer, [_frame(i, [A]) for i in range(5, 30)])
        self.assertEqual(len(streamer.tracker.tracks), 1)
        self.assertEqual(len(updates[-1].scores), 1)

    def test_forgetting_the_leader_promotes_another(self):
        streamer = StreamingTracker(IouTracker(), forget_after=5, switch_ratio=1.0)
        _push_all(streamer, [_frame(i, [A, B]) for i in range(10)])
        updates = _push_all(streamer, [_frame(i, [B]) for i in range(10, 30)])
        self.assertEqual(updates[-1].protagonist_id, updates[-1].assignments[0][0])

    def test_reset_clears_everything(self):
        streamer = StreamingTracker(IouTracker())
        _push_all(streamer, [_frame(i, [A, B]) for i in range(5)])
        streamer.reset()
        update = streamer.push(_frame(0, [B]))  # frame ids may restart
        self.assertEqual(update.assignments[0][0], 0)
        self.assertEqual(len(update.scores), 1)


class TestFaceIdBackend(unittest.TestCase):
    @staticmethod
    def _encoder(crops: torch.Tensor) -> torch.Tensor:
        """Hand-written stand-in for a face encoder: normalised mean colour."""
        return torch.nn.functional.normalize(crops.float().mean(dim=(2, 3)), dim=1)

    def test_tracks_with_identity_embeddings_and_forgets_them(self):
        streamer = StreamingTracker(
            FaceIdTracker(self._encoder, embedding_threshold=1.01), forget_after=4
        )
        _push_all(streamer, [_frame(i, [A, B]) for i in range(6)])
        self.assertEqual(len(streamer.tracker.tracks), 2)
        later = _push_all(streamer, [_frame(i, [A]) for i in range(6, 14)])
        self.assertEqual(len(streamer.tracker.tracks), 1)
        self.assertEqual(set(streamer.tracker._track_embeddings), set(streamer.tracker.tracks))
        self.assertEqual(later[-1].protagonist_id, later[-1].assignments[0][0])


class TestOfflineEquivalence(unittest.TestCase):
    @staticmethod
    def _random_walk_video(n_frames: int, seed: int) -> VideoDetections:
        rng = random.Random(seed)
        positions = [[40.0, 40.0, 90], [500.0, 300.0, 70], [900.0, 80.0, 50]]
        vd = VideoDetections()
        for frame_id in range(n_frames):
            fd = FrameDetections()
            for idx, (x, y, size) in enumerate(positions):
                if rng.random() < 0.15 and idx != 0:  # others flicker out
                    continue
                positions[idx][0] = x + rng.uniform(-6, 6)
                positions[idx][1] = y + rng.uniform(-6, 6)
                fd.add(_det(frame_id, int(positions[idx][0]), int(positions[idx][1]), size))
            vd.add(fd)
        return vd

    def test_tracks_and_protagonist_match_the_offline_path(self):
        vd = self._random_walk_video(120, seed=3)
        offline = IouTracker().label(vd)
        streamer = StreamingTracker(
            IouTracker(), window=10_000, forget_after=10_000, switch_ratio=1.0
        )
        for fd in vd:
            update = streamer.push(fd)
        self.assertEqual(sorted(offline.tracks), sorted(streamer.tracker.tracks))
        for track_id, track in offline.tracks.items():
            self.assertEqual(track.frame_ids(), streamer.tracker.tracks[track_id].frame_ids())
        expected = max(offline.tracks.values(), key=lambda t: len(t) * t.bb_size(extra_percent=0.0))
        self.assertEqual(update.protagonist_id, expected.track_id)


if __name__ == "__main__":
    unittest.main()
