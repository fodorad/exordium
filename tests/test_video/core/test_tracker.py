"""Tests for exordium.video.core.tracker module (merged from test_tracker.py and test_track.py)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exordium.video.core.detection import (
    DetectionFactory,
    DetectionFromImage,
    FrameDetections,
    Track,
    VideoDetections,
)
from exordium.video.core.tracker import IouTracker


class TrackTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detection1 = DetectionFromImage(
            frame_id=1,
            source="video/000001.png",
            score=0.9,
            bb_xywh=np.array([10, 20, 50, 60]),
            landmarks=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        )
        cls.detection2 = DetectionFromImage(
            frame_id=2,
            source="video/000002.png",
            score=0.8,
            bb_xywh=np.array([20, 30, 60, 70]),
            landmarks=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        )
        cls.detection3 = DetectionFromImage(
            frame_id=3,
            source="video/000003.png",
            score=0.7,
            bb_xywh=np.array([30, 40, 70, 80]),
            landmarks=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        )
        cls.TMP_DIR = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        if cls.TMP_DIR.exists():
            shutil.rmtree(cls.TMP_DIR)

    def test_track_creation(self):
        track = Track(1, self.detection1)
        self.assertEqual(track.track_id, 1)
        self.assertEqual(len(track.detections), 1)
        self.assertEqual(track.detections[0], self.detection1)

    def test_frame_ids(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        frame_ids = track.frame_ids()
        self.assertListEqual(frame_ids, [1, 2, 3])

    def test_get_detection(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        detection = track.get_detection(2)
        self.assertEqual(detection, self.detection2)

    def test_add_detection(self):
        track1 = Track(1, self.detection1)
        track2 = Track(2, self.detection2)
        track1.add(self.detection3)
        self.assertEqual(len(track1.detections), 2)
        self.assertEqual(track1.detections[1], self.detection3)
        track1.merge(track2)
        self.assertEqual(len(track1.detections), 3)
        self.assertEqual(track1.detections[1], self.detection2)

    def test_track_frame_distance(self):
        track1 = Track(1, self.detection1)
        track2 = Track(2, self.detection2)
        distance = track1.frame_distance(track2)
        self.assertEqual(distance, 1)
        track3 = Track(3, self.detection3)
        distance = track1.frame_distance(track3)
        self.assertEqual(distance, 2)

    def test_last_detection(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        last_detection = track.last_detection()
        self.assertEqual(last_detection, self.detection3)

    def test_first_detection(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        first_detection = track.first_detection()
        self.assertEqual(first_detection, self.detection1)

    def test_center(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        center = track.center()
        expected_center = [50.0, 65.0]
        self.assertListEqual(center.tolist(), expected_center)

    def test_bb_size(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        bb_size = track.bb_size(extra_percent=0.2)
        self.assertEqual(bb_size, 96)

    def test_len(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        self.assertEqual(len(track), 3)

    def test_iterator(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        detections = [detection for _, detection in enumerate(track)]
        self.assertEqual(len(detections), 3)
        self.assertEqual(detections[0], self.detection1)
        self.assertEqual(detections[1], self.detection2)
        self.assertEqual(detections[2], self.detection3)

    def test_save_and_load(self):
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        output_file = self.TMP_DIR / "output.csv"
        track.save(output_file)
        loaded_detections = Track(track_id=1).load(output_file)
        self.assertEqual(len(loaded_detections), 3)
        self.assertEqual(loaded_detections[0], self.detection1)
        self.assertEqual(loaded_detections[1], self.detection2)
        self.assertEqual(loaded_detections[2], self.detection3)


def _build_synthetic_vdet(num_frames: int = 30) -> VideoDetections:
    """Build synthetic VideoDetections with 3 well-separated faces across frames.

    Faces are placed at fixed positions in a 720×1280 frame so that IoU between
    consecutive detections of the same face is 1.0, and IoU between different
    faces is 0.  Order is [left, right, center] so track IDs are 0, 1, 2 and
    the center face gets track_id=2 (closest to the frame centre (640, 360)).
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    landmarks = np.array([[10, 10], [20, 10], [15, 20], [10, 25], [20, 25]])
    # bb_xywh: [x, y, w, h] → mid = [x + w/2, y + h/2]
    face_bbs = [
        np.array([100, 200, 100, 120]),  # track 0: left  — mid=(150, 260)
        np.array([1050, 200, 100, 120]),  # track 1: right — mid=(1100, 260)
        np.array([590, 200, 100, 120]),  # track 2: center — mid=(640, 260)
    ]
    vdet = VideoDetections()
    for frame_id in range(num_frames):
        fd = FrameDetections()
        for bb in face_bbs:
            fd.detections.append(
                DetectionFactory.create_detection(
                    frame_id=frame_id,
                    source=frame,
                    score=0.9,
                    bb_xywh=bb.copy(),
                    landmarks=landmarks.copy(),
                )
            )
        vdet.add(fd)
    return vdet


class TrackerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vdet = _build_synthetic_vdet(num_frames=30)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_tracker_creation(self):
        tracker = IouTracker()
        self.assertEqual(tracker.new_track_id, 0)
        self.assertEqual(len(tracker.tracks), 0)
        self.assertEqual(len(tracker.selected_tracks), 0)

    def test_label_tracks_iou(self):
        tracker = IouTracker()
        tracker.label(self.vdet)
        self.assertEqual(len(tracker.tracks), 3)

    def test_interpolate(self):
        tracker = IouTracker()
        vdet_part1 = VideoDetections()
        vdet_part1.detections = self.vdet.detections[:10]
        vdet_part2 = VideoDetections()
        vdet_part2.detections = self.vdet.detections[20:]
        vdet = VideoDetections()
        vdet.merge(vdet_part1)
        vdet.merge(vdet_part2)
        tracker.label(vdet, max_lost=15)
        self.assertEqual(len(tracker.tracks), 3)

    def test_merge_iou(self):
        tracker = IouTracker()
        vdet_part1 = VideoDetections()
        vdet_part1.detections = self.vdet.detections[:10]
        vdet_part2 = VideoDetections()
        vdet_part2.detections = self.vdet.detections[20:]
        vdet = VideoDetections()
        vdet.merge(vdet_part1)
        vdet.merge(vdet_part2)
        tracker.label(vdet, max_lost=5)
        self.assertEqual(len(tracker.tracks), 6)
        tracker.merge()
        self.assertEqual(len(tracker.tracks), 3)

    def test_merge_zero_tracks_returns_self(self):
        tracker = IouTracker()
        result = tracker.merge()
        self.assertIs(result, tracker)

    def test_select_long_track(self):
        tracker = IouTracker()
        tracker.label(self.vdet)
        del tracker.tracks[0].detections[19:]
        tracker.select_long_tracks(min_length=20)
        self.assertEqual(len(tracker.selected_tracks), 2)

    def test_select_top_long_track(self):
        tracker = IouTracker()
        tracker.label(self.vdet)
        del tracker.tracks[0].detections[20:]
        del tracker.tracks[1].detections[:20]
        tracker.select_topk_long_tracks(top_k=1)
        self.assertEqual(len(tracker.selected_tracks), 1)

    def test_select_topk_biggest_bb_tracks(self):
        tracker = IouTracker()
        tracker.label(self.vdet)
        tracker.select_topk_biggest_bb_tracks(top_k=1)
        self.assertEqual(len(tracker.selected_tracks), 1)

    def test_select_center_track(self):
        tracker = IouTracker()
        tracker.label(self.vdet)
        center_track = tracker.get_center_track()
        self.assertEqual(center_track.track_id, 2)

    def test_get_center_track_empty_returns_none(self):
        tracker = IouTracker()
        result = tracker.get_center_track()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
