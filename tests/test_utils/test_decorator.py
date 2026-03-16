"""Tests for exordium.utils.decorator module."""

import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from exordium.utils.decorator import (
    FrameDetLoader,
    LoaderFactory,
    NpyLoader,
    PickleLoader,
    TrackLoader,
    VideoDetLoader,
    load_or_create,
    timer,
    timer_with_return,
)
from exordium.video.core.detection import (
    DetectionFromImage,
    FrameDetections,
    Track,
    VideoDetections,
)
from tests.fixtures import IMAGE_FACE


class TestTimer(unittest.TestCase):
    """Test timer decorator."""

    def test_timer_executes_function(self):
        """Test that timer decorator executes the wrapped function."""

        @timer
        def sample_func():
            return "executed"

        # Capture stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = sample_func()

        # Timer doesn't return value, so result is None
        self.assertIsNone(result)
        output = f.getvalue()
        self.assertIn("Function took:", output)
        self.assertIn("seconds", output)

    def test_timer_measures_time(self):
        """Test that timer measures execution time."""

        @timer
        def sleep_func():
            time.sleep(0.01)

        f = io.StringIO()
        with redirect_stdout(f):
            sleep_func()

        output = f.getvalue()
        self.assertIn("Function took:", output)

    def test_timer_with_args(self):
        """Test timer with function arguments."""

        @timer
        def func_with_args(a, b):
            return a + b

        f = io.StringIO()
        with redirect_stdout(f):
            result = func_with_args(2, 3)

        self.assertIsNone(result)  # timer doesn't return
        output = f.getvalue()
        self.assertIn("Function took:", output)


class TestTimerWithReturn(unittest.TestCase):
    """Test timer_with_return decorator."""

    @patch("exordium.utils.decorator.TIMING_ENABLED", True)
    def test_timer_with_return_enabled(self):
        """Test timer_with_return when timing is enabled."""

        @timer_with_return
        def sample_func():
            return "result"

        f = io.StringIO()
        with redirect_stdout(f):
            result = sample_func()

        self.assertEqual(result, "result")
        output = f.getvalue()
        self.assertIn("Execution time of sample_func:", output)

    @patch("exordium.utils.decorator.TIMING_ENABLED", False)
    def test_timer_with_return_disabled(self):
        """Test timer_with_return when timing is disabled."""

        @timer_with_return
        def sample_func():
            return "result"

        f = io.StringIO()
        with redirect_stdout(f):
            result = sample_func()

        self.assertEqual(result, "result")
        output = f.getvalue()
        self.assertEqual(output, "")  # No timing output

    @patch("exordium.utils.decorator.TIMING_ENABLED", True)
    def test_timer_with_return_preserves_args(self):
        """Test that timer_with_return preserves function arguments."""

        @timer_with_return
        def func_with_args(a, b, c=3):
            return a + b + c

        f = io.StringIO()
        with redirect_stdout(f):
            result = func_with_args(1, 2, c=4)

        self.assertEqual(result, 7)

    @patch("exordium.utils.decorator.TIMING_ENABLED", True)
    def test_timer_with_return_function_name(self):
        """Test that timer_with_return shows correct function name."""

        @timer_with_return
        def custom_named_function():
            return True

        f = io.StringIO()
        with redirect_stdout(f):
            custom_named_function()

        output = f.getvalue()
        self.assertIn("custom_named_function", output)


class TestPickleLoader(unittest.TestCase):
    """Test PickleLoader class."""

    def test_save_and_load(self):
        """Test saving and loading with PickleLoader."""
        loader = PickleLoader()
        data = {"key": "value", "number": 42}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.pkl"
            loader.save(data, file_path)

            self.assertTrue(file_path.exists())

            loaded_data = loader.load(file_path)
            self.assertEqual(data, loaded_data)

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        loader = PickleLoader()
        data = {"test": 123}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test.pkl"
            loader.save(data, file_path)

            self.assertTrue(file_path.exists())

    def test_load_complex_data(self):
        """Test loading complex Python objects."""
        loader = PickleLoader()
        data = {
            "list": [1, 2, 3],
            "dict": {"a": 1},
            "tuple": (1, 2),
            "nested": {"deep": {"value": 42}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "complex.pkl"
            loader.save(data, file_path)
            loaded_data = loader.load(file_path)

            self.assertEqual(data, loaded_data)


class TestNpyLoader(unittest.TestCase):
    """Test NpyLoader class."""

    def test_save_and_load(self):
        """Test saving and loading numpy arrays."""
        loader = NpyLoader()
        data = np.array([[1, 2, 3], [4, 5, 6]])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npy"
            loader.save(data, file_path)

            loaded_data = loader.load(file_path)
            np.testing.assert_array_equal(data, loaded_data)

    def test_save_different_dtypes(self):
        """Test saving arrays with different dtypes."""
        loader = NpyLoader()

        test_cases = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, data in enumerate(test_cases):
                file_path = Path(tmpdir) / f"test_{i}.npy"
                loader.save(data, file_path)
                loaded_data = loader.load(file_path)

                np.testing.assert_array_equal(data, loaded_data)
                self.assertEqual(data.dtype, loaded_data.dtype)


class TestLoaderFactory(unittest.TestCase):
    """Test LoaderFactory class."""

    def test_get_pickle_loader(self):
        """Test getting PickleLoader from factory."""
        loader = LoaderFactory.get("pkl")
        self.assertIsInstance(loader, PickleLoader)

    def test_get_npy_loader(self):
        """Test getting NpyLoader from factory."""
        loader = LoaderFactory.get("npy")
        self.assertIsInstance(loader, NpyLoader)

    def test_unsupported_format(self):
        """Test that unsupported format raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            LoaderFactory.get("unsupported_format")

    def test_factory_creates_new_instances(self):
        """Test that factory creates new instances each time."""
        loader1 = LoaderFactory.get("pkl")
        loader2 = LoaderFactory.get("pkl")

        self.assertIsNot(loader1, loader2)


class TestLoadOrCreate(unittest.TestCase):
    """Test load_or_create decorator."""

    def test_creates_when_no_output_path(self):
        """Test that function is executed when output_path is None."""

        @load_or_create(format="pkl")
        def create_data():
            return {"created": True}

        result = create_data()
        self.assertEqual(result, {"created": True})

    def test_creates_and_saves_with_output_path(self):
        """Test that data is created and saved with output_path."""

        @load_or_create(format="pkl")
        def create_data(output_path=None, overwrite=False):
            return {"created": True}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pkl"
            result = create_data(output_path=output_path)

            self.assertEqual(result, {"created": True})
            self.assertTrue(output_path.exists())

    def test_loads_existing_file(self):
        """Test that existing file is loaded instead of recreating."""
        call_count = 0

        @load_or_create(format="pkl")
        def create_data(output_path=None, overwrite=False):
            nonlocal call_count
            call_count += 1
            return {"call": call_count}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pkl"

            # First call creates the file
            result1 = create_data(output_path=output_path)
            self.assertEqual(result1, {"call": 1})
            self.assertEqual(call_count, 1)

            # Second call loads the file
            result2 = create_data(output_path=output_path)
            self.assertEqual(result2, {"call": 1})  # Same data
            self.assertEqual(call_count, 1)  # Function not called again

    def test_overwrites_when_overwrite_true(self):
        """Test that file is overwritten when overwrite=True."""
        call_count = 0

        @load_or_create(format="pkl")
        def create_data(output_path=None, overwrite=False):
            nonlocal call_count
            call_count += 1
            return {"call": call_count}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pkl"

            # First call
            result1 = create_data(output_path=output_path)
            self.assertEqual(result1, {"call": 1})

            # Second call with overwrite
            result2 = create_data(output_path=output_path, overwrite=True)
            self.assertEqual(result2, {"call": 2})
            self.assertEqual(call_count, 2)

    def test_creates_parent_directory(self):
        """Test that parent directories are created."""

        @load_or_create(format="pkl")
        def create_data(output_path=None, overwrite=False):
            return {"data": 123}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "output.pkl"
            result = create_data(output_path=output_path)

            self.assertEqual(result, {"data": 123})
            self.assertTrue(output_path.exists())


def _make_detection(frame_id: int = 0) -> DetectionFromImage:
    """Creates a DetectionFromImage using the face fixture."""
    return DetectionFromImage(
        frame_id=frame_id,
        source=str(IMAGE_FACE),
        score=0.99,
        bb_xywh=np.array([10, 20, 50, 60]),
        landmarks=np.array([[15, 25], [45, 25], [30, 40], [18, 55], [42, 55]]),
    )


class TestFrameDetLoader(unittest.TestCase):
    """Test FrameDetLoader class."""

    def test_save_and_load_roundtrip(self):
        fdet = FrameDetections().add(_make_detection(frame_id=0))
        loader = FrameDetLoader()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frame.fdet"
            loader.save(fdet, path)
            self.assertTrue(path.exists())
            loaded = loader.load(path)
        self.assertIsInstance(loaded, FrameDetections)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].frame_id, 0)

    def test_loader_factory_fdet(self):
        loader = LoaderFactory.get("fdet")
        self.assertIsInstance(loader, FrameDetLoader)

    def test_load_or_create_with_fdet(self):
        @load_or_create(format="fdet")
        def make_fdet(output_path=None, overwrite=False):
            return FrameDetections().add(_make_detection(frame_id=1))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frame.fdet"
            result = make_fdet(output_path=path)
            self.assertIsInstance(result, FrameDetections)
            self.assertTrue(path.exists())
            loaded = make_fdet(output_path=path)
            self.assertEqual(loaded[0].frame_id, 1)


class TestVideoDetLoader(unittest.TestCase):
    """Test VideoDetLoader class."""

    def test_save_and_load_roundtrip(self):
        vdet = VideoDetections()
        vdet.add(FrameDetections().add(_make_detection(frame_id=0)))
        vdet.add(FrameDetections().add(_make_detection(frame_id=1)))
        loader = VideoDetLoader()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "video.vdet"
            loader.save(vdet, path)
            self.assertTrue(path.exists())
            loaded = loader.load(path)
        self.assertIsInstance(loaded, VideoDetections)
        self.assertEqual(len(loaded), 2)

    def test_loader_factory_vdet(self):
        loader = LoaderFactory.get("vdet")
        self.assertIsInstance(loader, VideoDetLoader)

    def test_load_or_create_with_vdet(self):
        @load_or_create(format="vdet")
        def make_vdet(output_path=None, overwrite=False):
            vdet = VideoDetections()
            vdet.add(FrameDetections().add(_make_detection(frame_id=3)))
            return vdet

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "video.vdet"
            result = make_vdet(output_path=path)
            self.assertIsInstance(result, VideoDetections)
            self.assertTrue(path.exists())


class TestTrackLoader(unittest.TestCase):
    """Test TrackLoader class."""

    def test_save_and_load_roundtrip(self):
        track = Track(track_id=7)
        track.add(_make_detection(frame_id=0))
        track.add(_make_detection(frame_id=1))
        loader = TrackLoader()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "track.track"
            loader.save(track, path)
            self.assertTrue(path.exists())
            loaded = loader.load(path)
        self.assertIsInstance(loaded, Track)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].frame_id, 0)

    def test_loader_factory_track(self):
        loader = LoaderFactory.get("track")
        self.assertIsInstance(loader, TrackLoader)

    def test_load_or_create_with_track(self):
        @load_or_create(format="track")
        def make_track(output_path=None, overwrite=False):
            t = Track(track_id=1)
            t.add(_make_detection(frame_id=5))
            return t

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "track.track"
            result = make_track(output_path=path)
            self.assertIsInstance(result, Track)
            self.assertTrue(path.exists())
            loaded = make_track(output_path=path)
            self.assertEqual(loaded[0].frame_id, 5)


if __name__ == "__main__":
    unittest.main()
