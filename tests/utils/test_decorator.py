"""Tests for exordium.utils.decorator: timer, LoaderFactory, loaders, load_or_create."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.utils.decorator import (
    FrameDetLoader,
    LoaderFactory,
    NpyLoader,
    PickleLoader,
    SafetensorsLoader,
    TrackLoader,
    VideoDetLoader,
    load_or_create,
    timer,
)
from tests.fixtures import IMAGE_FACE


def _make_frame_detections():
    from exordium.video.core.detection import DetectionFactory, FrameDetections

    fd = FrameDetections()
    fd.add(
        DetectionFactory.create_detection(
            frame_id=0,
            source=str(IMAGE_FACE),
            score=0.9,
            bb_xywh=torch.tensor([10, 10, 30, 30], dtype=torch.long),
            landmarks=torch.zeros((5, 2), dtype=torch.long),
        )
    )
    return fd


def _make_video_detections():
    from exordium.video.core.detection import VideoDetections

    vd = VideoDetections()
    vd.add(_make_frame_detections())
    return vd


def _make_track():
    from exordium.video.core.detection import DetectionFactory, Track

    t = Track(0)
    t.add(
        DetectionFactory.create_detection(
            frame_id=0,
            source=str(IMAGE_FACE),
            score=0.9,
            bb_xywh=torch.tensor([10, 10, 30, 30], dtype=torch.long),
            landmarks=torch.zeros((5, 2), dtype=torch.long),
        )
    )
    return t


class TestTimer(unittest.TestCase):
    def test_returns_correct_result(self):
        @timer
        def add(a, b):
            return a + b

        result = add(2, 3)
        self.assertEqual(result, 5)

    def test_works_with_keyword_args(self):
        @timer
        def greet(name="World"):
            return f"Hello, {name}"

        result = greet(name="Test")
        self.assertEqual(result, "Hello, Test")


class TestLoaderFactory(unittest.TestCase):
    def test_get_pkl_returns_pickle_loader(self):
        loader = LoaderFactory.get("pkl")
        self.assertIsInstance(loader, PickleLoader)

    def test_get_npy_returns_npy_loader(self):
        loader = LoaderFactory.get("npy")
        self.assertIsInstance(loader, NpyLoader)

    def test_unknown_format_raises(self):
        with self.assertRaises(NotImplementedError):
            LoaderFactory.get("unknown_format")

    def test_get_fdet(self):
        loader = LoaderFactory.get("fdet")
        self.assertIsInstance(loader, FrameDetLoader)

    def test_get_vdet(self):
        loader = LoaderFactory.get("vdet")
        self.assertIsInstance(loader, VideoDetLoader)

    def test_get_track(self):
        loader = LoaderFactory.get("track")
        self.assertIsInstance(loader, TrackLoader)

    def test_get_safetensors(self):
        loader = LoaderFactory.get("st")
        self.assertIsInstance(loader, SafetensorsLoader)

    def test_get_unknown_raises(self):
        with self.assertRaises(NotImplementedError):
            LoaderFactory.get("unknown_format_xyz")


class TestPickleLoader(unittest.TestCase):
    def test_round_trip_list(self):
        data = [1, 2, 3, "hello"]
        loader = PickleLoader()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            loader.save(data, path)
            loaded = loader.load(path)
            self.assertEqual(loaded, data)
        finally:
            path.unlink(missing_ok=True)

    def test_round_trip_dict(self):
        data = {"key": 42, "values": [1.0, 2.0]}
        loader = PickleLoader()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            loader.save(data, path)
            loaded = loader.load(path)
            self.assertEqual(loaded, data)
        finally:
            path.unlink(missing_ok=True)


class TestNpyLoader(unittest.TestCase):
    def test_round_trip_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        loader = NpyLoader()
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = Path(f.name)
        try:
            loader.save(arr, path)
            loaded = loader.load(path)
            np.testing.assert_array_equal(loaded, arr)
        finally:
            path.unlink(missing_ok=True)

    def test_shape_preserved(self):
        arr = np.zeros((5, 10, 3), dtype=np.float64)
        loader = NpyLoader()
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = Path(f.name)
        try:
            loader.save(arr, path)
            loaded = loader.load(path)
            self.assertEqual(loaded.shape, arr.shape)
        finally:
            path.unlink(missing_ok=True)

    def test_round_trip_2d(self):
        loader = NpyLoader()
        arr = np.random.randn(5, 10).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            p = Path(f.name)
        try:
            loader.save(arr, p)
            loaded = loader.load(p)
            np.testing.assert_array_almost_equal(loaded, arr)
        finally:
            p.unlink(missing_ok=True)


class TestSafetensorsLoader(unittest.TestCase):
    def test_save_dict_and_load(self):
        loader = SafetensorsLoader()
        data = {"a": torch.randn(3, 4), "b": torch.randn(2)}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            p = Path(f.name)
        try:
            loader.save(data, p)
            loaded = loader.load(p)
            self.assertIsInstance(loaded, dict)
            self.assertIn("a", loaded)
        finally:
            p.unlink(missing_ok=True)

    def test_save_invalid_type_raises_type_error(self):
        loader = SafetensorsLoader()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "out.safetensors"
            with self.assertRaises(TypeError):
                loader.save([1, 2, 3], p)

    def test_save_and_load_tensor(self):
        loader = SafetensorsLoader()
        t = torch.randn(3, 4)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            p = Path(f.name)
        try:
            loader.save(t, p)
            loaded = loader.load(p)
            if isinstance(loaded, dict):
                loaded = next(iter(loaded.values()))
            torch.testing.assert_close(loaded, t)
        finally:
            p.unlink(missing_ok=True)


class TestFrameDetLoader(unittest.TestCase):
    def test_save_and_load(self):
        fd = _make_frame_detections()
        loader = FrameDetLoader()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            p = Path(f.name)
        try:
            loader.save(fd, p)
            loaded = loader.load(p)
            self.assertEqual(len(loaded), 1)
        finally:
            p.unlink(missing_ok=True)


class TestVideoDetLoader(unittest.TestCase):
    def test_save_and_load(self):
        vd = _make_video_detections()
        loader = VideoDetLoader()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            p = Path(f.name)
        try:
            loader.save(vd, p)
            loaded = loader.load(p)
            self.assertGreaterEqual(len(loaded), 1)
        finally:
            p.unlink(missing_ok=True)


class TestTrackLoader(unittest.TestCase):
    def test_save_and_load(self):
        track = _make_track()
        loader = TrackLoader()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            p = Path(f.name)
        try:
            loader.save(track, p)
            loaded = loader.load(p)
            self.assertGreaterEqual(len(loaded), 1)
        finally:
            p.unlink(missing_ok=True)


class TestLoadOrCreate(unittest.TestCase):
    def test_creates_and_caches_npy(self):
        call_count = [0]

        @load_or_create("npy")
        def compute(**kwargs):
            call_count[0] += 1
            return np.array([1.0, 2.0, 3.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "result.npy"
            result1 = compute(output_path=p)
            self.assertEqual(call_count[0], 1)
            self.assertTrue(p.exists())

            result2 = compute(output_path=p)
            self.assertEqual(call_count[0], 1)
            np.testing.assert_array_equal(result1, result2)

    def test_overwrite_forces_recompute(self):
        call_count = [0]

        @load_or_create("npy")
        def compute(**kwargs):
            call_count[0] += 1
            return np.zeros(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "result.npy"
            compute(output_path=p)
            compute(output_path=p, overwrite=True)
            self.assertEqual(call_count[0], 2)

    def test_no_output_path_runs_without_saving(self):
        call_count = [0]

        @load_or_create("npy")
        def compute(**kwargs):
            call_count[0] += 1
            return np.array([5.0])

        result = compute()
        self.assertEqual(call_count[0], 1)
        np.testing.assert_array_equal(result, np.array([5.0]))


if __name__ == "__main__":
    unittest.main()
