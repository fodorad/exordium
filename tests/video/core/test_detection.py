"""Tests for detection data structures: Detection, FrameDetections, VideoDetections, IouTracker."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from exordium.video.core.detection import (
    DetectionFactory,
    FrameDetections,
    IouTracker,
    Track,
    VideoDetections,
    _arr_equal,
    _to_bgr_numpy,
    _to_list,
    add_detections_to_frame,
    save_detections_to_video,
    save_track_target_to_images,
    save_track_with_context_to_video,
    visualize_detection,
    visualize_detection_crop,
)
from tests.fixtures import IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT


def _make_fd(frame_id: int, x: int, y: int, w: int = 60, h: int = 80) -> FrameDetections:
    """Create a FrameDetections with a single detection at (x, y, w, h)."""
    fd = FrameDetections()
    fd.add_dict(
        {
            "frame_id": frame_id,
            "source": str(IMAGE_FACE),
            "score": 0.95,
            "bb_xywh": torch.tensor([x, y, w, h], dtype=torch.long),
            "landmarks": torch.zeros((5, 2), dtype=torch.long),
        }
    )
    return fd


def _np_det(frame_id, x=50, y=50, w=60, h=80, score=0.95):
    return DetectionFactory.create_detection(
        frame_id=frame_id,
        source=str(IMAGE_FACE),
        score=score,
        bb_xywh=torch.tensor([x, y, w, h], dtype=torch.long),
        landmarks=torch.zeros((5, 2), dtype=torch.long),
    )


def _tensor_det(frame_id, frame_t, x=50, y=50, w=60, h=80):
    return DetectionFactory.create_detection(
        frame_id=frame_id,
        source=frame_t,
        score=0.9,
        bb_xywh=torch.tensor([x, y, w, h], dtype=torch.int32),
        landmarks=torch.zeros((5, 2), dtype=torch.int32),
    )


def _video_det(frame_id):
    return DetectionFactory.create_detection(
        frame_id=frame_id,
        source=str(VIDEO_MULTISPEAKER_SHORT),
        score=0.9,
        bb_xywh=torch.tensor([50, 50, 60, 60], dtype=torch.long),
        landmarks=torch.zeros((5, 2), dtype=torch.long),
    )


def _fd(frame_id, x=50, y=50, w=60, h=80, score=0.95):
    fd = FrameDetections()
    fd.add(_np_det(frame_id, x, y, w, h, score))
    return fd


def _numpy_frame_det(frame_id, np_frame, x=10, y=10, w=30, h=30, score=0.9):
    return DetectionFactory.create_detection(
        frame_id=frame_id,
        source=np_frame,
        score=score,
        bb_xywh=torch.tensor([x, y, w, h], dtype=torch.long),
        landmarks=torch.zeros((5, 2), dtype=torch.long),
    )


def _det(frame_id, x=50, y=50, w=80, h=80, score=0.95):
    return DetectionFactory.create_detection(
        frame_id=frame_id,
        source=str(IMAGE_FACE),
        score=score,
        bb_xywh=torch.tensor([x, y, w, h], dtype=torch.long),
        landmarks=torch.zeros((5, 2), dtype=torch.long),
    )


class TestDetectionFromImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd = _make_fd(frame_id=0, x=50, y=50)
        cls.det = list(fd)[0]

    def test_frame_returns_tensor(self):
        frame = self.det.frame()
        self.assertIsInstance(frame, torch.Tensor)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[0], 3)

    def test_crop_returns_tensor(self):
        crop = self.det.crop()
        self.assertIsInstance(crop, torch.Tensor)
        self.assertEqual(crop.ndim, 3)
        self.assertEqual(crop.shape[0], 3)

    def test_crop_square_with_extra_space(self):
        crop = self.det.crop(square=True, extra_space=1.5)
        self.assertIsInstance(crop, torch.Tensor)
        self.assertEqual(crop.ndim, 3)

    def test_score_in_range(self):
        self.assertGreater(self.det.score, 0.0)
        self.assertLessEqual(self.det.score, 1.0)

    def test_bb_xywh_shape(self):
        bb = self.det.bb_xywh
        self.assertEqual(len(bb), 4)


class TestDetectionFactory(unittest.TestCase):
    def test_from_image_path(self):
        fd = _make_fd(0, 10, 10)
        det = list(fd)[0]
        self.assertEqual(det.source, str(IMAGE_FACE))

    def test_factory_from_numpy_source(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        fd = FrameDetections()
        fd.add_dict(
            {
                "frame_id": 0,
                "source": arr,
                "score": 0.8,
                "bb_xywh": torch.tensor([10, 10, 30, 30], dtype=torch.long),
                "landmarks": torch.zeros((5, 2), dtype=torch.long),
            }
        )
        det = list(fd)[0]
        self.assertIsNotNone(det)
        crop = det.crop()
        self.assertIsInstance(crop, torch.Tensor)

    def test_factory_from_torch_tensor_source(self):
        t = torch.zeros(3, 100, 100, dtype=torch.uint8)
        fd = FrameDetections()
        fd.add_dict(
            {
                "frame_id": 0,
                "source": t,
                "score": 0.9,
                "bb_xywh": torch.tensor([10.0, 10.0, 30.0, 30.0]),
                "landmarks": torch.zeros((5, 2), dtype=torch.int64),
            }
        )
        det = list(fd)[0]
        crop = det.crop()
        self.assertIsInstance(crop, torch.Tensor)


class TestFrameDetections(unittest.TestCase):
    def test_len(self):
        fd = _make_fd(0, 10, 10)
        self.assertEqual(len(fd), 1)

    def test_iter(self):
        fd = _make_fd(0, 10, 10)
        dets = list(fd)
        self.assertEqual(len(dets), 1)

    def test_empty_frame_detections(self):
        fd = FrameDetections()
        self.assertEqual(len(fd), 0)

    def test_csv_round_trip(self):
        fd = _make_fd(0, 20, 30)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)
        try:
            fd.save(csv_path)
            fd2 = FrameDetections().load(csv_path)
            self.assertEqual(len(fd2), len(fd))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_multiple_detections(self):
        fd = FrameDetections()
        for i in range(3):
            fd.add_dict(
                {
                    "frame_id": 0,
                    "source": str(IMAGE_FACE),
                    "score": 0.9,
                    "bb_xywh": torch.tensor([i * 10, i * 10, 20, 20], dtype=torch.long),
                    "landmarks": torch.zeros((5, 2), dtype=torch.long),
                }
            )
        self.assertEqual(len(fd), 3)


class TestVideoDetections(unittest.TestCase):
    def test_add_and_len(self):
        vd = VideoDetections()
        vd.add(_make_fd(0, 10, 10))
        vd.add(_make_fd(1, 20, 20))
        self.assertEqual(len(vd), 2)

    def test_iter(self):
        vd = VideoDetections()
        vd.add(_make_fd(0, 10, 10))
        frames = list(vd)
        self.assertEqual(len(frames), 1)

    def test_csv_round_trip(self):
        vd = VideoDetections()
        for i in range(4):
            vd.add(_make_fd(i, i * 5, i * 5))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)
        try:
            vd.save(csv_path)
            vd2 = VideoDetections().load(csv_path)
            self.assertEqual(len(vd2), len(vd))
        finally:
            csv_path.unlink(missing_ok=True)


class TestIouTracker(unittest.TestCase):
    def _make_vd_sequence(self, n_frames: int, x_start: int = 10) -> VideoDetections:
        """Build a VideoDetections with a single slowly moving face across frames."""
        vd = VideoDetections()
        for i in range(n_frames):
            fd = _make_fd(frame_id=i, x=x_start + i * 2, y=50)
            vd.add(fd)
        return vd

    def test_single_track_created(self):
        vd = self._make_vd_sequence(6)
        tracker = IouTracker()
        tracker.label(vd)
        tracks = tracker.selected_tracks
        self.assertEqual(len(tracks), 1)

    def test_track_has_correct_length(self):
        vd = self._make_vd_sequence(6)
        tracker = IouTracker()
        tracker.label(vd)
        tracks = tracker.selected_tracks
        first_track = next(iter(tracks.values()))
        self.assertEqual(len(first_track), 6)

    def test_two_separate_tracks(self):
        """A large gap (no overlap) should split into two tracks."""
        vd = VideoDetections()
        for i in range(4):
            vd.add(_make_fd(frame_id=i, x=10, y=10))
        for i in range(4, 8):
            vd.add(_make_fd(frame_id=i, x=500, y=500))
        tracker = IouTracker(iou_threshold=0.3)
        tracker.label(vd)
        tracks = tracker.selected_tracks
        self.assertGreaterEqual(len(tracks), 2)


class TestToList(unittest.TestCase):
    def test_tensor_2d(self):
        t = torch.tensor([[1, 2], [3, 4]])
        self.assertEqual(_to_list(t), [1, 2, 3, 4])

    def test_tensor(self):
        t = torch.tensor([5, 6, 7])
        self.assertEqual(_to_list(t), [5, 6, 7])


class TestArrEqual(unittest.TestCase):
    def test_tensor_equal(self):
        a = torch.tensor([1, 2, 3])
        self.assertTrue(_arr_equal(a, a.clone()))

    def test_tensor_not_equal(self):
        self.assertFalse(_arr_equal(torch.tensor([1, 2]), torch.tensor([1, 3])))

    def test_both_tensor(self):
        self.assertTrue(_arr_equal(torch.tensor([4, 5]), torch.tensor([4, 5])))

    def test_tensor_not_equal_values(self):
        self.assertFalse(_arr_equal(torch.tensor([1, 2]), torch.tensor([1, 3])))


class TestDetectionFromVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det = _video_det(0)

    def test_frame_returns_rgb_tensor(self):
        frame = self.det.frame()
        self.assertIsInstance(frame, torch.Tensor)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[0], 3)

    def test_frame_center_shape(self):
        center = self.det.frame_center()
        self.assertIsInstance(center, torch.Tensor)
        self.assertEqual(center.shape, (2,))

    def test_crop_tensor(self):
        crop = self.det.crop()
        self.assertIsInstance(crop, torch.Tensor)

    def test_crop_square_with_extra_space(self):
        crop = self.det.crop(square=True, extra_space=1.5)
        self.assertIsInstance(crop, torch.Tensor)


class TestDetectionFromTorchTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame_t = torch.randint(0, 255, (3, 200, 200), dtype=torch.uint8)
        cls.det = _tensor_det(0, cls.frame_t, x=50, y=50, w=60, h=80)

    def test_frame_is_same_tensor(self):
        self.assertIs(self.det.frame(), self.frame_t)

    def test_frame_center(self):
        center = self.det.frame_center()
        self.assertEqual(center[0].item(), 100)
        self.assertEqual(center[1].item(), 100)

    def test_crop(self):
        crop = self.det.crop()
        self.assertIsInstance(crop, torch.Tensor)

    def test_crop_square_with_extra_space(self):
        crop = self.det.crop(square=True, extra_space=1.5)
        self.assertIsInstance(crop, torch.Tensor)


class TestDetectionFactoryErrors(unittest.TestCase):
    def test_missing_key_raises_key_error(self):
        with self.assertRaises(KeyError):
            DetectionFactory.create_detection(frame_id=0, source=str(IMAGE_FACE))

    def test_unsupported_extension_raises_value_error(self):
        with self.assertRaises(ValueError):
            DetectionFactory.create_detection(
                frame_id=0,
                source="/some/file.xyz",
                score=0.9,
                bb_xywh=torch.tensor([0, 0, 10, 10], dtype=torch.long),
                landmarks=torch.zeros((5, 2), dtype=torch.long),
            )

    def test_unsupported_source_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            DetectionFactory.create_detection(
                frame_id=0,
                source=12345,
                score=0.9,
                bb_xywh=torch.tensor([0, 0, 10, 10], dtype=torch.long),
                landmarks=torch.zeros((5, 2), dtype=torch.long),
            )


class TestFrameDetectionsProperties(unittest.TestCase):
    def test_frame_id_property(self):
        fd = _fd(7)
        self.assertEqual(fd.frame_id, 7)

    def test_source_property(self):
        fd = _fd(0)
        self.assertEqual(fd.source, str(IMAGE_FACE))

    def test_eq_true(self):
        self.assertEqual(_fd(0), _fd(0))

    def test_eq_false_different_type(self):
        self.assertNotEqual(_fd(0), "not a FrameDetections")

    def test_get_detection_biggest_bb(self):
        fd = FrameDetections()
        fd.add(_np_det(0, w=20, h=20))
        fd.add(_np_det(0, w=80, h=80))
        biggest = fd.get_detection_with_biggest_bb()
        self.assertGreaterEqual(int(biggest.bb_xywh[2]), 80)

    def test_get_detection_highest_score(self):
        fd = FrameDetections()
        fd.add(_np_det(0, score=0.5))
        fd.add(_np_det(0, score=0.99))
        best = fd.get_detection_with_highest_score()
        self.assertAlmostEqual(best.score, 0.99)


class TestVideoDetectionsMergeAndEq(unittest.TestCase):
    def test_merge(self):
        vd1 = VideoDetections()
        vd1.add(_fd(0))
        vd2 = VideoDetections()
        vd2.add(_fd(1))
        vd1.merge(vd2)
        self.assertEqual(len(vd1), 2)

    def test_eq_false_different_type(self):
        vd = VideoDetections()
        self.assertNotEqual(vd, "other")


class TestTrackSampleAndGetDetection(unittest.TestCase):
    def _track(self, n=10):
        t = Track()
        for i in range(n):
            t.add(_np_det(i))
        return t

    def test_sample_when_fewer_than_n(self):
        t = self._track(3)
        result = t.sample(10)
        self.assertEqual(len(result), 3)

    def test_sample_exactly_n(self):
        t = self._track(10)
        result = t.sample(5)
        self.assertEqual(len(result), 5)

    def test_get_detection_by_frame(self):
        t = self._track(5)
        d = t.get_detection(3)
        self.assertEqual(d.frame_id, 3)

    def test_str_repr_contains_id(self):
        t = Track(42)
        t.add(_np_det(0))
        s = str(t)
        self.assertIn("42", s)


class TestIouTrackerMergeRuleAndCenterTrack(unittest.TestCase):
    def test_merge_rule_should_merge_same_position(self):
        tracker = IouTracker(max_lost=10, iou_threshold=0.1)
        t1 = Track(0)
        t1.add(_np_det(0, x=50, y=50, w=80, h=80))
        t2 = Track(1)
        t2.add(_np_det(5, x=50, y=50, w=80, h=80))
        should, keep, drop = tracker.merge_rule(t1, t2)
        self.assertTrue(should)

    def test_merge_rule_no_merge_far_apart(self):
        tracker = IouTracker(max_lost=10, iou_threshold=0.5)
        t1 = Track(0)
        t1.add(_np_det(0, x=10, y=10, w=20, h=20))
        t2 = Track(1)
        t2.add(_np_det(5, x=500, y=500, w=20, h=20))
        should, _, _ = tracker.merge_rule(t1, t2)
        self.assertFalse(should)

    def test_merge_with_single_track_is_noop(self):
        tracker = IouTracker()
        vd = VideoDetections()
        vd.add(_fd(0))
        tracker.label(vd)
        result = tracker.merge()
        self.assertIs(result, tracker)

    def test_get_center_track_returns_track(self):
        tracker = IouTracker()
        vd = VideoDetections()
        for i in range(5):
            vd.add(_fd(i, x=50 + i, y=50))
        tracker.label(vd)
        center = tracker.get_center_track()
        self.assertIsInstance(center, Track)

    def test_get_center_track_empty_returns_none(self):
        tracker = IouTracker()
        result = tracker.get_center_track()
        self.assertIsNone(result)


class TestAddDetectionsToFrameTensor(unittest.TestCase):
    def test_with_rgb_tensor(self):
        fd = _fd(0)
        frame_t = torch.zeros(3, 100, 100, dtype=torch.uint8)
        result = add_detections_to_frame(fd, frame_t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 3)


class TestSaveDetectionsToVideo(unittest.TestCase):
    def test_writes_annotated_frames(self):
        vd = VideoDetections()
        vd.add(_fd(0))

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            out_dir = Path(tmpdir) / "out"

            frame_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_dir / "000000.png"), frame_img)

            save_detections_to_video(vd, frame_dir, out_dir)
            saved = list(out_dir.glob("*.png"))
            self.assertGreater(len(saved), 0)

    def test_writes_unannotated_frame_for_missing_detection(self):
        vd = VideoDetections()  # no detections for frame 0

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            out_dir = Path(tmpdir) / "out"

            frame_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_dir / "000000.png"), frame_img)

            save_detections_to_video(vd, frame_dir, out_dir)
            saved = list(out_dir.glob("*.png"))
            self.assertEqual(len(saved), 1)


class TestSaveTrackTargetToImages(unittest.TestCase):
    def test_saves_all_detections(self):
        track = Track(0)
        for i in range(3):
            track.add(_np_det(i, x=100, y=100, w=60, h=60))
        with tempfile.TemporaryDirectory() as tmpdir:
            save_track_target_to_images(track, tmpdir)
            self.assertEqual(len(list(Path(tmpdir).glob("*.png"))), 3)

    def test_sample_every_n(self):
        track = Track(0)
        for i in range(4):
            track.add(_np_det(i, x=100, y=100, w=60, h=60))
        with tempfile.TemporaryDirectory() as tmpdir:
            save_track_target_to_images(track, tmpdir, sample_every_n=2)
            self.assertEqual(len(list(Path(tmpdir).glob("*.png"))), 2)

    def test_bb_size_minus_one_skips_resize(self):
        track = Track(0)
        track.add(_np_det(0, x=100, y=100, w=60, h=60))
        with tempfile.TemporaryDirectory() as tmpdir:
            save_track_target_to_images(track, tmpdir, bb_size=-1)
            self.assertEqual(len(list(Path(tmpdir).glob("*.png"))), 1)


class TestSaveTrackWithContextToVideo(unittest.TestCase):
    def test_writes_context_frames(self):
        track = Track(0)
        track.add(_np_det(0, x=30, y=30, w=40, h=40))

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            out_dir = Path(tmpdir) / "out"

            frame_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_dir / "000000.png"), frame_img)

            save_track_with_context_to_video(track, frame_dir, out_dir)
            saved = list(out_dir.glob("*.png"))
            self.assertGreater(len(saved), 0)


class TestVisualizeDetection(unittest.TestCase):
    def test_returns_ndarray(self):
        det = _np_det(0, x=100, y=100, w=80, h=80)
        result = visualize_detection(det)
        self.assertIsInstance(result, np.ndarray)

    def test_with_output_path(self):
        det = _np_det(0, x=100, y=100, w=80, h=80)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "viz.png"
            visualize_detection(det, output_path=p)
            self.assertTrue(p.exists())

    def test_with_show_indices(self):
        det = _np_det(0, x=100, y=100, w=80, h=80)
        result = visualize_detection(det, show_indices=True)
        self.assertIsInstance(result, np.ndarray)

    def test_with_tensor_landmarks(self):
        frame_t = torch.randint(0, 255, (3, 200, 200), dtype=torch.uint8)
        det = DetectionFactory.create_detection(
            frame_id=0,
            source=frame_t,
            score=0.9,
            bb_xywh=torch.tensor([50, 50, 60, 60]),
            landmarks=torch.zeros((5, 2), dtype=torch.int32),
        )
        result = visualize_detection(det)
        self.assertIsInstance(result, np.ndarray)


class TestToBgrNumpyNonChwTensor(unittest.TestCase):
    def test_non_chw_tensor_converted(self):
        """A 2D tensor (not C,H,W) should go through the else branch."""
        t = torch.zeros(100, 100, dtype=torch.uint8)
        result = _to_bgr_numpy(t)
        self.assertIsInstance(result, np.ndarray)

    def test_hwc_tensor_converted_to_bgr(self):
        """Tensor with shape (H, W, C) is not C,H,W so hits else branch."""
        t = torch.zeros(100, 100, 3, dtype=torch.uint8)
        result = _to_bgr_numpy(t)
        self.assertIsInstance(result, np.ndarray)


class TestDetectionBbCenter(unittest.TestCase):
    def test_bb_center_shape(self):
        det = _np_det(0, x=100, y=100, w=60, h=80)
        center = det.bb_center
        self.assertEqual(len(center), 2)

    def test_bb_center_approximate(self):
        det = _np_det(0, x=100, y=100, w=60, h=60)
        center = det.bb_center
        self.assertAlmostEqual(float(center[0]), 130.0, delta=2)


class TestDetectionEqNonDetection(unittest.TestCase):
    def test_eq_with_non_detection_returns_false(self):
        det = _np_det(0)
        self.assertNotEqual(det, "not a Detection")
        self.assertNotEqual(det, 42)
        self.assertNotEqual(det, None)


class TestDetectionFromNpMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cls.det = _numpy_frame_det(0, cls.np_frame, x=10, y=10, w=30, h=30)

    def test_frame_center(self):
        center = self.det.frame_center()
        self.assertIsInstance(center, torch.Tensor)
        self.assertEqual(center.shape, (2,))
        self.assertEqual(center[0].item(), 100)

    def test_crop_square_with_extra_space(self):
        crop = self.det.crop(square=True, extra_space=1.5)
        self.assertIsInstance(crop, torch.Tensor)


class TestVideoDetectionsGetitemAndEq(unittest.TestCase):
    def test_getitem(self):
        vd = VideoDetections()
        vd.add(_fd(0))
        vd.add(_fd(1))
        fd = vd[0]
        self.assertIsInstance(fd, FrameDetections)

    def test_eq_two_equal_instances(self):
        vd1 = VideoDetections()
        vd1.add(_fd(0))
        vd2 = VideoDetections()
        vd2.add(_fd(0))
        self.assertEqual(vd1, vd2)


class TestVideoDetectionsLoadMultiplePerFrame(unittest.TestCase):
    """Test that loading a CSV with multiple detections per frame works."""

    def test_load_two_detections_same_frame(self):
        vd = VideoDetections()
        vd.add(_fd(0))
        vd.detections[0].add(_np_det(0, x=200, y=200, w=60, h=80))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            p = Path(f.name)
        try:
            vd.save(p)
            loaded = VideoDetections().load(p)
            total = sum(len(fd) for fd in loaded)
            self.assertGreaterEqual(total, 2)
        finally:
            p.unlink(missing_ok=True)


class TestTrackFrameDistanceOverlap(unittest.TestCase):
    def test_overlapping_tracks_return_zero(self):
        """Tracks that overlap in time should have frame_distance = 0."""
        t1 = Track(0)
        t1.add(_np_det(0))
        t1.add(_np_det(5))

        t2 = Track(1)
        t2.add(_np_det(3))  # overlaps with t1

        dist = t1.frame_distance(t2)
        self.assertEqual(dist, 0)


class TestTrackerScoreFilter(unittest.TestCase):
    def test_low_score_detections_skipped(self):
        """Detections below min_score should not create tracks."""
        vd = VideoDetections()
        fd = FrameDetections()
        fd.add(_np_det(0, score=0.3))  # score below default 0.7
        vd.add(fd)

        tracker = IouTracker()
        tracker.label(vd, min_score=0.7)
        self.assertEqual(len(tracker.tracks), 0)


class TestTrackerMergeMultipleTracks(unittest.TestCase):
    def test_merge_two_tracks_at_same_position(self):
        """Two tracks at same position that meet merge_rule should be merged."""
        tracker = IouTracker(max_lost=100, iou_threshold=0.0)

        vd = VideoDetections()
        for i in range(3):
            fd = FrameDetections()
            fd.add(_np_det(i, x=50, y=50, w=80, h=80))
            vd.add(fd)

        tracker.label(vd)
        initial_count = len(tracker.tracks)
        tracker.merge()
        self.assertLessEqual(len(tracker.tracks), initial_count)


class TestSaveTrackTargetWithVideo(unittest.TestCase):
    def test_save_with_save_video_true(self):
        """save_track_target_to_images with save_video=True creates a .mp4."""
        track = Track(0)
        for i in range(3):
            track.add(_np_det(i, x=100, y=100, w=60, h=60))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "crops"
            save_track_target_to_images(track, out_dir, save_video=True)
            mp4 = out_dir.parent / f"{out_dir.stem}.mp4"
            self.assertTrue(mp4.exists())


class TestSaveTrackWithContextSaveVideo(unittest.TestCase):
    def test_context_with_save_video_and_missing_frame(self):
        """save_track_with_context_to_video: frame not in track is skipped."""
        track = Track(0)
        track.add(_np_det(1, x=30, y=30, w=40, h=40))  # only frame 1

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            out_dir = Path(tmpdir) / "out"

            for i in range(3):
                frame_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(frame_dir / f"{i:06d}.png"), frame_img)

            save_track_with_context_to_video(track, frame_dir, out_dir, save_video=True)
            mp4 = out_dir.parent / f"{out_dir.stem}.mp4"
            self.assertTrue(mp4.exists())
            saved = list(out_dir.glob("*.png"))
            self.assertEqual(len(saved), 1)


class TestTrackerMergeLoop(unittest.TestCase):
    def test_merge_combines_two_overlapping_tracks(self):
        """Force two tracks with overlapping bbs into tracker and call merge."""
        tracker = IouTracker(max_lost=-1, iou_threshold=0.0)

        t1 = Track(0)
        for i in range(4):
            t1.add(_det(i, x=50, y=50, w=80, h=80))

        t2 = Track(1)
        t2.add(_det(6, x=50, y=50, w=80, h=80))

        tracker.tracks = {0: t1, 1: t2}
        tracker.new_track_id = 2

        initial_count = len(tracker.tracks)
        tracker.merge()

        self.assertLess(len(tracker.tracks), initial_count)
        merged = next(iter(tracker.tracks.values()))
        self.assertEqual(len(merged), 5)

    def test_merge_skips_blacklisted_tracks(self):
        """After a track is merged (added to blacklist), it is skipped."""
        tracker = IouTracker(max_lost=-1, iou_threshold=0.0)

        t0 = Track(0)
        t0.add(_det(0, x=50, y=50, w=80, h=80))

        t1 = Track(1)
        t1.add(_det(5, x=50, y=50, w=80, h=80))

        t2 = Track(2)
        t2.add(_det(10, x=50, y=50, w=80, h=80))

        tracker.tracks = {0: t0, 1: t1, 2: t2}
        tracker.new_track_id = 3

        tracker.merge()
        self.assertLessEqual(len(tracker.tracks), 3)

    def test_merge_returns_self(self):
        tracker = IouTracker()
        t0 = Track(0)
        t0.add(_det(0))
        t1 = Track(1)
        t1.add(_det(5, x=200, y=200))  # far apart — no merge
        tracker.tracks = {0: t0, 1: t1}
        result = tracker.merge()
        self.assertIs(result, tracker)


class TestTrack(unittest.TestCase):
    def _make_track(self, n=5):
        t = Track()
        for i in range(n):
            fd = _make_fd(i, 10 + i * 2, 20)
            t.add(list(fd)[0])
        return t

    def test_len(self):
        self.assertEqual(len(self._make_track(5)), 5)

    def test_frame_ids(self):
        t = self._make_track(3)
        self.assertEqual(t.frame_ids(), [0, 1, 2])

    def test_first_last_detection(self):
        t = self._make_track(4)
        self.assertEqual(t.first_detection().frame_id, 0)
        self.assertEqual(t.last_detection().frame_id, 3)

    def test_iter(self):
        t = self._make_track(3)
        self.assertEqual(len(list(t)), 3)

    def test_getitem(self):
        t = self._make_track(3)
        self.assertIsNotNone(t[0])

    def test_center_shape(self):
        t = self._make_track(3)
        c = t.center()
        self.assertEqual(len(c), 2)

    def test_bb_size_positive(self):
        t = self._make_track(3)
        self.assertGreater(t.bb_size(), 0)

    def test_save_load_roundtrip(self):
        t = self._make_track(3)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            p = Path(f.name)
        try:
            t.save(p)
            t2 = Track().load(p)
            self.assertEqual(len(t2), len(t))
        finally:
            p.unlink(missing_ok=True)

    def test_merge_two_tracks(self):
        t1 = self._make_track(3)
        t2 = Track()
        fd = _make_fd(10, 50, 50)
        t2.add(list(fd)[0])
        merged = t1.merge(t2)
        self.assertEqual(len(merged), 4)

    def test_is_started_earlier(self):
        t1 = self._make_track(3)
        t2 = Track()
        t2.add(list(_make_fd(5, 10, 10))[0])
        self.assertTrue(t1.is_started_earlier(t2))

    def test_frame_distance(self):
        t1 = self._make_track(3)  # last frame=2
        t2 = Track()
        t2.add(list(_make_fd(5, 10, 10))[0])  # first frame=5
        self.assertEqual(t1.frame_distance(t2), 3)


class TestTrackerMergeAndSelect(unittest.TestCase):
    def _make_vd(self, n_frames, x_start=10):
        vd = VideoDetections()
        for i in range(n_frames):
            vd.add(_make_fd(i, x_start + i * 2, 50))
        return vd

    def test_tracker_merge(self):
        vd = self._make_vd(6)
        tracker = IouTracker()
        tracker.label(vd)
        tracker.merge()
        self.assertGreaterEqual(len(tracker.selected_tracks), 1)

    def test_select_long_tracks_returns_subset(self):
        vd = self._make_vd(10)
        tracker = IouTracker()
        tracker.label(vd)
        tracker.select_long_tracks(min_length=5)
        for track in tracker.selected_tracks.values():
            self.assertGreaterEqual(len(track), 5)

    def test_select_topk_long_tracks(self):
        vd = self._make_vd(8)
        tracker = IouTracker()
        tracker.label(vd)
        tracker.select_topk_long_tracks(top_k=1)
        self.assertLessEqual(len(tracker.selected_tracks), 1)

    def test_select_topk_biggest_bb(self):
        vd = self._make_vd(6)
        tracker = IouTracker()
        tracker.label(vd)
        tracker.select_topk_biggest_bb_tracks(top_k=1)
        self.assertLessEqual(len(tracker.selected_tracks), 1)


class TestAddDetectionsToFrame(unittest.TestCase):
    def test_returns_ndarray(self):
        fd = _make_fd(0, 10, 10)
        result = add_detections_to_frame(fd)
        self.assertIsInstance(result, np.ndarray)

    def test_with_frame_numpy(self):
        fd = _make_fd(0, 10, 10)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = add_detections_to_frame(fd, frame)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[2], 3)


class TestCropOrigin(unittest.TestCase):
    """Unit tests for Detection._crop_origin — pure geometry, no I/O."""

    def _make_det(self, x, y, w, h):
        return DetectionFactory.create_detection(
            frame_id=0,
            source=str(IMAGE_FACE),
            score=0.9,
            bb_xywh=torch.tensor([x, y, w, h], dtype=torch.long),
            landmarks=torch.zeros((5, 2), dtype=torch.long),
        )

    def test_non_square_origin_equals_scaled_topleft(self):
        """For square=False, extra_space=1.0 origin matches the BB top-left."""
        det = self._make_det(x=100, y=100, w=60, h=60)
        # cx=130, cy=130, half_w=half_h=30, x1=int(130-30)=100
        x1, y1 = det._crop_origin(square=False, extra_space=1.0, frame_w=400, frame_h=400)
        self.assertEqual(x1, 100)
        self.assertEqual(y1, 100)

    def test_square_tall_bb_extends_left_of_xmin(self):
        """square=True + h>w: crop extends further left than bb_xyxy[0]."""
        # BB: x=100, y=100, w=60, h=100 → side=100, cx=130, cy=150
        # x1 = int(130 - 50) = 80  (< bb_xyxy[0]=100)
        det = self._make_det(x=100, y=100, w=60, h=100)
        x1, y1 = det._crop_origin(square=True, extra_space=1.0, frame_w=400, frame_h=400)
        self.assertEqual(x1, 80)
        self.assertEqual(y1, 100)

    def test_clamped_to_zero_near_edge(self):
        det = self._make_det(x=5, y=5, w=60, h=80)
        x1, y1 = det._crop_origin(square=True, extra_space=1.0, frame_w=400, frame_h=400)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)

    def test_extra_space_scales_sides(self):
        """extra_space=2.0 doubles the side for square mode."""
        det = self._make_det(x=100, y=100, w=60, h=60)
        # square: side=60*2=120, half=60, cx=130, x1=int(130-60)=70
        x1, _ = det._crop_origin(square=True, extra_space=2.0, frame_w=400, frame_h=400)
        self.assertEqual(x1, 70)


class TestCrop(unittest.TestCase):
    """Tests for Detection.crop() — loads frame and crops."""

    def test_crop_returns_tensor(self):
        det = _np_det(0, x=50, y=50, w=60, h=80)
        crop = det.crop()
        self.assertIsInstance(crop, torch.Tensor)
        self.assertEqual(crop.ndim, 3)
        self.assertEqual(crop.shape[0], 3)

    def test_crop_square_returns_tensor(self):
        det = _np_det(0, x=50, y=50, w=60, h=80)
        crop = det.crop(square=True)
        self.assertIsInstance(crop, torch.Tensor)

    def test_crop_square_with_extra_space(self):
        det = _np_det(0, x=50, y=50, w=60, h=80)
        crop = det.crop(square=True, extra_space=1.5)
        self.assertIsInstance(crop, torch.Tensor)

    def test_crop_non_square_with_extra_space(self):
        det = _np_det(0, x=50, y=50, w=60, h=80)
        crop = det.crop(extra_space=1.2)
        self.assertIsInstance(crop, torch.Tensor)

    def test_crop_square_is_squarish(self):
        """Square crop height and width should be equal (or differ by ≤1 due to clamping)."""
        det = _np_det(0, x=100, y=100, w=60, h=80)
        crop = det.crop(square=True)
        self.assertAlmostEqual(crop.shape[1], crop.shape[2], delta=2)

    def test_torch_tensor_source(self):
        frame_t = torch.randint(0, 255, (3, 200, 200), dtype=torch.uint8)
        det = DetectionFactory.create_detection(
            frame_id=0,
            source=frame_t,
            score=0.9,
            bb_xywh=torch.tensor([50, 50, 60, 80]),
            landmarks=torch.zeros((5, 2), dtype=torch.int32),
        )
        crop = det.crop(square=True)
        self.assertIsInstance(crop, torch.Tensor)


class TestCropLandmarks(unittest.TestCase):
    """Tests for Detection.crop_landmarks()."""

    def test_shape_and_dtype(self):
        det = _np_det(0, x=50, y=50, w=60, h=80)
        lm = det.crop_landmarks()
        self.assertIsInstance(lm, torch.Tensor)
        self.assertEqual(lm.shape, (5, 2))
        self.assertEqual(lm.dtype, torch.long)

    def test_correct_offset_non_square(self):
        """crop_landmarks(square=False) shifts by the non-square crop origin."""
        raw_lm = torch.tensor(
            [[120, 130], [140, 130], [130, 150], [120, 160], [140, 160]], dtype=torch.long
        )
        det = DetectionFactory.create_detection(
            frame_id=0,
            source=str(IMAGE_FACE),
            score=0.9,
            bb_xywh=torch.tensor([100, 100, 60, 80], dtype=torch.long),
            landmarks=raw_lm,
        )
        _, h, w = det.frame().shape
        x1, y1 = det._crop_origin(square=False, extra_space=1.0, frame_w=w, frame_h=h)
        lm_crop = det.crop_landmarks(square=False)
        torch.testing.assert_close(lm_crop[:, 0], raw_lm[:, 0] - x1)
        torch.testing.assert_close(lm_crop[:, 1], raw_lm[:, 1] - y1)

    def test_correct_offset_square(self):
        """crop_landmarks(square=True) shifts by the square-crop origin."""
        raw_lm = torch.tensor(
            [[120, 130], [140, 130], [130, 150], [120, 160], [140, 160]], dtype=torch.long
        )
        det = DetectionFactory.create_detection(
            frame_id=0,
            source=str(IMAGE_FACE),
            score=0.9,
            bb_xywh=torch.tensor([100, 100, 60, 100], dtype=torch.long),
            landmarks=raw_lm,
        )
        _, h, w = det.frame().shape
        x1, y1 = det._crop_origin(square=True, extra_space=1.0, frame_w=w, frame_h=h)
        lm_crop = det.crop_landmarks(square=True)
        torch.testing.assert_close(lm_crop[:, 0], raw_lm[:, 0] - x1)
        torch.testing.assert_close(lm_crop[:, 1], raw_lm[:, 1] - y1)

    def test_extra_space_origin_applied(self):
        """extra_space shifts the origin further left, changing landmark positions."""
        det = _np_det(0, x=80, y=80, w=50, h=90)
        lm_tight = det.crop_landmarks(square=True, extra_space=1.0)
        lm_wide = det.crop_landmarks(square=True, extra_space=1.5)
        # wider crop → origin shifts further left (smaller x1) → subtracting a
        # smaller x1 gives LARGER crop-local x coordinates.
        self.assertLessEqual(int(lm_tight[0, 0]), int(lm_wide[0, 0]))

    def test_with_tensor_landmarks(self):
        frame_t = torch.randint(0, 255, (3, 200, 200), dtype=torch.uint8)
        det = DetectionFactory.create_detection(
            frame_id=0,
            source=frame_t,
            score=0.9,
            bb_xywh=torch.tensor([50, 50, 60, 80]),
            landmarks=torch.ones((5, 2), dtype=torch.int32) * 80,
        )
        lm_crop = det.crop_landmarks(square=True)
        # landmarks stored as torch.int32 → crop_landmarks preserves dtype
        self.assertIsInstance(lm_crop, torch.Tensor)
        self.assertEqual(lm_crop.dtype, torch.int32)
        self.assertEqual(tuple(lm_crop.shape), (5, 2))


class TestVisualizeDetectionCrop(unittest.TestCase):
    """Tests for the visualize_detection_crop() standalone function."""

    def test_returns_ndarray(self):
        det = _np_det(0, x=100, y=100, w=80, h=80)
        result = visualize_detection_crop(det)
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_matches_crop(self):
        """Result spatial size should match det.crop(square, extra_space)."""
        det = _np_det(0, x=100, y=100, w=80, h=80)
        crop = det.crop(square=True)
        result = visualize_detection_crop(det, square=True)
        self.assertEqual(result.shape[0], crop.shape[1])
        self.assertEqual(result.shape[1], crop.shape[2])

    def test_non_square_mode(self):
        det = _np_det(0, x=100, y=100, w=80, h=80)
        result = visualize_detection_crop(det, square=False)
        self.assertIsInstance(result, np.ndarray)

    def test_with_extra_space(self):
        det = _np_det(0, x=100, y=100, w=80, h=80)
        result = visualize_detection_crop(det, square=True, extra_space=1.5)
        self.assertIsInstance(result, np.ndarray)

    def test_with_output_path(self):
        det = _np_det(0, x=100, y=100, w=80, h=80)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "crop_viz.png"
            visualize_detection_crop(det, output_path=p)
            self.assertTrue(p.exists())

    def test_with_show_indices(self):
        det = _np_det(0, x=100, y=100, w=80, h=80)
        result = visualize_detection_crop(det, show_indices=True)
        self.assertIsInstance(result, np.ndarray)

    def test_with_tensor_source(self):
        frame_t = torch.randint(0, 255, (3, 300, 300), dtype=torch.uint8)
        det = DetectionFactory.create_detection(
            frame_id=0,
            source=frame_t,
            score=0.9,
            bb_xywh=torch.tensor([50, 50, 80, 80]),
            landmarks=torch.tensor(
                [[90, 90], [110, 90], [100, 110], [90, 120], [110, 120]], dtype=torch.int32
            ),
        )
        result = visualize_detection_crop(det, square=True)
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
