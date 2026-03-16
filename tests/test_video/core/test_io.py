import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from exordium.video.core.io import (
    _BACKEND,
    TorchCodecBackend,
    Video,
    VideoBackend,
    batch_iterator,
    check_same_image_dims,
    get_video_metadata,
    image_to_np,
    images_to_np,
    load_frames,
    load_video,
    save_frames,
    save_frames_with_ids,
    save_video,
    sequence_to_video,
    video_to_frames,
)
from tests.fixtures import IMAGE_CAT_TIE, VIDEO_MULTISPEAKER_SHORT


class TestVideoBackend(unittest.TestCase):
    """Tests for VideoBackend abstract class and TorchCodecBackend implementation."""

    def test_backend_is_instance(self):
        """Test that global backend is properly instantiated."""
        self.assertIsInstance(_BACKEND, VideoBackend)
        self.assertIsInstance(_BACKEND, TorchCodecBackend)

    def test_backend_get_metadata(self):
        """Test backend get_metadata method."""
        metadata = _BACKEND.get_metadata(str(VIDEO_MULTISPEAKER_SHORT))
        self.assertIn("fps", metadata)
        self.assertIn("num_frames", metadata)
        self.assertIn("height", metadata)
        self.assertIn("width", metadata)
        self.assertIn("duration", metadata)
        self.assertGreater(metadata["fps"], 0)
        self.assertGreater(metadata["num_frames"], 0)

    def test_backend_decode_frames(self):
        """Test backend decode_frames method."""
        indices = [0, 1, 2]
        frames = _BACKEND.decode_frames(str(VIDEO_MULTISPEAKER_SHORT), indices)
        self.assertEqual(frames.shape[0], 3, "Should decode 3 frames")
        self.assertEqual(frames.shape[1], 3, "Should have 3 channels")
        self.assertEqual(len(frames.shape), 4, "Should be 4D tensor (T, C, H, W)")

    def test_backend_decode_all(self):
        """Test backend decode_all method."""
        frames = _BACKEND.decode_all(str(VIDEO_MULTISPEAKER_SHORT))
        self.assertGreater(frames.shape[0], 0, "Should decode at least one frame")
        self.assertEqual(frames.shape[1], 3, "Should have 3 channels")


class TestGetVideoMetadata(unittest.TestCase):
    """Tests for get_video_metadata function."""

    def test_get_metadata_basic(self):
        """Test basic metadata extraction."""
        metadata = get_video_metadata(VIDEO_MULTISPEAKER_SHORT)
        self.assertIn("fps", metadata)
        self.assertIn("num_frames", metadata)
        self.assertIn("height", metadata)
        self.assertIn("width", metadata)
        self.assertIn("duration", metadata)

    def test_get_metadata_values(self):
        """Test that metadata values are reasonable."""
        metadata = get_video_metadata(VIDEO_MULTISPEAKER_SHORT)
        self.assertGreater(metadata["fps"], 0, "FPS should be positive")
        self.assertGreater(metadata["num_frames"], 0, "Frame count should be positive")
        self.assertGreater(metadata["height"], 0, "Height should be positive")
        self.assertGreater(metadata["width"], 0, "Width should be positive")
        self.assertGreater(metadata["duration"], 0, "Duration should be positive")


class TestLoadVideo(unittest.TestCase):
    """Tests for load_video function."""

    def test_load_video_basic(self):
        """Test basic video loading."""
        frames, fps = load_video(VIDEO_MULTISPEAKER_SHORT)
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(len(frames.shape), 4, "Should be 4D tensor (T, C, H, W)")
        self.assertEqual(frames.shape[1], 3, "Should have 3 channels")
        self.assertGreater(fps, 0, "FPS should be positive")

    def test_load_video_with_start_end(self):
        """Test loading video with start and end frames."""
        frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT, start_frame=5, end_frame=15)
        self.assertEqual(frames.shape[0], 10, "Should load 10 frames")

    def test_load_video_with_fps_conversion(self):
        """Test FPS conversion when loading video."""
        metadata = get_video_metadata(VIDEO_MULTISPEAKER_SHORT)
        native_fps = metadata["fps"]

        # Load at half the native FPS
        target_fps = native_fps / 2
        frames, actual_fps = load_video(VIDEO_MULTISPEAKER_SHORT, fps=target_fps)

        self.assertEqual(actual_fps, target_fps, "Output FPS should match requested FPS")
        # Should have approximately half the frames
        expected_frames = int(metadata["num_frames"] * target_fps / native_fps)
        self.assertAlmostEqual(
            frames.shape[0], expected_frames, delta=2, msg="Frame count should match FPS conversion"
        )

    def test_load_video_with_resize_int(self):
        """Test loading video with integer resize (smallest dimension)."""
        frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT, resize=200)
        height, width = frames.shape[2], frames.shape[3]
        self.assertEqual(
            min(height, width), 200, f"Smallest dimension should be 200, got shape {frames.shape}"
        )

    def test_load_video_with_resize_tuple(self):
        """Test loading video with tuple resize (H, W)."""
        frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT, resize=(240, 320))
        self.assertEqual(frames.shape[2], 240, "Height should be 240")
        self.assertEqual(frames.shape[3], 320, "Width should be 320")

    def test_load_video_with_crop(self):
        """Test loading video with crop."""
        frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT, crop=(10, 10, 100, 100))  # cy, cx, ch, cw
        self.assertEqual(frames.shape[2], 100, "Cropped height should be 100")
        self.assertEqual(frames.shape[3], 100, "Cropped width should be 100")

    def test_load_video_with_resize_and_crop(self):
        """Test loading video with both resize and crop."""
        frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT, resize=256, crop=(10, 10, 200, 200))
        self.assertEqual(frames.shape[2], 200, "Final height should be 200 after crop")
        self.assertEqual(frames.shape[3], 200, "Final width should be 200 after crop")

    def test_load_video_batch_size(self):
        """Test that different batch sizes produce same results."""
        frames1, fps1 = load_video(VIDEO_MULTISPEAKER_SHORT, batch_size=8)
        frames2, fps2 = load_video(VIDEO_MULTISPEAKER_SHORT, batch_size=16)

        self.assertEqual(frames1.shape, frames2.shape, "Batch size should not affect output shape")
        self.assertEqual(fps1, fps2, "Batch size should not affect FPS")


class TestLoadFrames(unittest.TestCase):
    """Tests for load_frames function."""

    def test_load_frames_basic(self):
        """Test basic frame loading with specific frame IDs."""
        frame_ids = [0, 5, 10, 15]
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids)
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.shape[0], 4, "Should load 4 frames")
        self.assertEqual(len(frames.shape), 4, "Should be 4D tensor (T, C, H, W)")
        self.assertEqual(frames.shape[1], 3, "Should have 3 channels")

    def test_load_frames_with_start_frame(self):
        """Test loading frames with start_frame offset."""
        frame_ids = [0, 1, 2, 3]
        start_frame = 10
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids, start_frame=start_frame)
        # Should load frames at absolute indices: 10, 11, 12, 13
        self.assertEqual(frames.shape[0], 4, "Should load 4 frames")

    def test_load_frames_numpy_array(self):
        """Test loading frames with numpy array frame_ids."""
        frame_ids = np.array([0, 2, 4, 6, 8])
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids)
        self.assertEqual(frames.shape[0], 5, "Should load 5 frames")

    def test_load_frames_with_resize_int(self):
        """Test loading frames with integer resize (smallest dimension)."""
        frame_ids = [0, 5, 10]
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids, resize=200)
        height, width = frames.shape[2], frames.shape[3]
        self.assertEqual(
            min(height, width), 200, f"Smallest dimension should be 200, got shape {frames.shape}"
        )

    def test_load_frames_with_resize_tuple(self):
        """Test loading frames with tuple resize (H, W)."""
        frame_ids = [0, 10]
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids, resize=(240, 320))
        self.assertEqual(frames.shape[2], 240, "Height should be 240")
        self.assertEqual(frames.shape[3], 320, "Width should be 320")

    def test_load_frames_with_crop(self):
        """Test loading frames with crop."""
        frame_ids = [0, 5]
        frames = load_frames(
            VIDEO_MULTISPEAKER_SHORT,
            frame_ids=frame_ids,
            crop=(10, 10, 100, 100),  # cy, cx, ch, cw
        )
        self.assertEqual(frames.shape[2], 100, "Cropped height should be 100")
        self.assertEqual(frames.shape[3], 100, "Cropped width should be 100")

    def test_load_frames_with_resize_and_crop(self):
        """Test loading frames with both resize and crop."""
        frame_ids = [0, 10, 20]
        frames = load_frames(
            VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids, resize=256, crop=(10, 10, 200, 200)
        )
        self.assertEqual(frames.shape[0], 3, "Should load 3 frames")
        self.assertEqual(frames.shape[2], 200, "Final height should be 200 after crop")
        self.assertEqual(frames.shape[3], 200, "Final width should be 200 after crop")

    def test_load_frames_batch_size(self):
        """Test that different batch sizes produce same results."""
        frame_ids = [0, 5, 10, 15, 20, 25, 29]
        frames1 = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids, batch_size=3)
        frames2 = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids, batch_size=10)

        self.assertEqual(frames1.shape, frames2.shape, "Batch size should not affect output shape")
        # Check that tensors are identical
        self.assertTrue(torch.equal(frames1, frames2), "Batch size should not affect output values")

    def test_load_frames_single_frame(self):
        """Test loading a single frame."""
        frame_ids = [5]
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids)
        self.assertEqual(frames.shape[0], 1, "Should load 1 frame")

    def test_load_frames_non_sequential(self):
        """Test loading non-sequential frames."""
        frame_ids = [20, 5, 29, 10, 0]  # Non-sequential order
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids)
        self.assertEqual(frames.shape[0], 5, "Should load 5 frames in order specified")


class TestVideoToFrames(unittest.TestCase):
    """Tests for video_to_frames function."""

    @classmethod
    def setUpClass(cls):
        """Create temporary directory"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.video_path = VIDEO_MULTISPEAKER_SHORT

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_video_to_frames_basic(self):
        """Test basic video frame extraction."""
        output_dir = Path(self.temp_dir) / "frames_basic"
        video_to_frames(self.video_path, output_dir)

        frames = sorted(list(output_dir.glob("*.png")))
        self.assertGreater(len(frames), 0, "Should extract at least one frame")
        self.assertTrue(frames[0].name.startswith("000000"), "First frame should start at 000000")

    def test_video_to_frames_with_fps(self):
        """Test frame extraction with custom fps."""
        output_dir = Path(self.temp_dir) / "frames_fps"
        video_to_frames(self.video_path, output_dir, fps=5)

        frames = sorted(list(output_dir.glob("*.png")))
        self.assertGreater(len(frames), 0, "Should extract at least one frame")

    def test_video_to_frames_with_resize(self):
        """Test frame extraction with resize."""
        output_dir = Path(self.temp_dir) / "frames_resize"
        video_to_frames(self.video_path, output_dir, smallest_dim=200)

        frames = sorted(list(output_dir.glob("*.png")))
        self.assertGreater(len(frames), 0, "Should extract at least one frame")

        # Check first frame is resized
        img = cv2.imread(str(frames[0]))
        self.assertEqual(
            min(img.shape[:2]),
            200,
            f"Smallest dimension should be 200. Image has shape of {img.shape}",
        )

    def test_video_to_frames_with_crop(self):
        """Test frame extraction with crop."""
        output_dir = Path(self.temp_dir) / "frames_crop"
        # Crop: (cy=10, cx=10, ch=100, cw=100)
        video_to_frames(self.video_path, output_dir, crop=(10, 10, 100, 100))

        frames = sorted(list(output_dir.glob("*.png")))
        self.assertGreater(len(frames), 0, "Should extract at least one frame")

        # Check first frame is cropped
        img = cv2.imread(str(frames[0]))
        self.assertEqual(img.shape[:2], (100, 100), "Frame should be 100x100")

    def test_video_to_frames_start_number(self):
        """Test frame extraction with custom start number."""
        output_dir = Path(self.temp_dir) / "frames_start"
        video_to_frames(self.video_path, output_dir, start_number=10)

        frames = sorted(list(output_dir.glob("*.png")))
        self.assertGreater(len(frames), 0, "Should extract at least one frame")
        self.assertTrue(frames[0].name.startswith("000010"), "First frame should start at 000010")

    def test_video_to_frames_extension(self):
        """Test frame extraction with custom extension."""
        output_dir = Path(self.temp_dir) / "frames_jpg"
        video_to_frames(self.video_path, output_dir, extension=".jpg")

        frames = sorted(list(output_dir.glob("*.jpg")))
        self.assertGreater(len(frames), 0, "Should extract at least one JPG frame")

    def test_video_to_frames_no_overwrite(self):
        """Test that overwrite=False skips existing directory."""
        output_dir = Path(self.temp_dir) / "frames_no_overwrite"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Should return immediately without extracting
        video_to_frames(self.video_path, output_dir, overwrite=False)

        frames = list(output_dir.glob("*.png"))
        self.assertEqual(len(frames), 0, "Should not extract frames when overwrite=False")


class TestSaveVideo(unittest.TestCase):
    """Tests for save_video function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_video_from_tensor_tchw(self):
        """Test saving video from torch.Tensor (T, C, H, W)."""
        frames = torch.randint(0, 255, (10, 3, 100, 100), dtype=torch.uint8)
        output_path = Path(self.temp_dir) / "output_tchw.mp4"

        save_video(frames, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")
        self.assertGreater(output_path.stat().st_size, 0, "Video file should not be empty")

    def test_save_video_from_tensor_thwc(self):
        """Test saving video from torch.Tensor (T, H, W, C)."""
        frames = torch.randint(0, 255, (10, 100, 100, 3), dtype=torch.uint8)
        output_path = Path(self.temp_dir) / "output_thwc.mp4"

        save_video(frames, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")
        self.assertGreater(output_path.stat().st_size, 0, "Video file should not be empty")

    def test_save_video_from_numpy_tchw(self):
        """Test saving video from numpy array (T, C, H, W)."""
        frames = np.random.randint(0, 255, (10, 3, 100, 100), dtype=np.uint8)
        output_path = Path(self.temp_dir) / "output_np_tchw.mp4"

        save_video(frames, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")

    def test_save_video_from_numpy_thwc(self):
        """Test saving video from numpy array (T, H, W, C)."""
        frames = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)
        output_path = Path(self.temp_dir) / "output_np_thwc.mp4"

        save_video(frames, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")

    def test_save_video_from_sequence_tensors(self):
        """Test saving video from sequence of tensors."""
        frames = [torch.randint(0, 255, (3, 100, 100), dtype=torch.uint8) for _ in range(10)]
        output_path = Path(self.temp_dir) / "output_seq_tensor.mp4"

        save_video(frames, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")

    def test_save_video_from_sequence_numpy(self):
        """Test saving video from sequence of numpy arrays."""
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
        output_path = Path(self.temp_dir) / "output_seq_numpy.mp4"

        save_video(frames, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")

    def test_save_video_no_overwrite(self):
        """Test that overwrite=False skips existing file."""
        frames = torch.randint(0, 255, (5, 3, 100, 100), dtype=torch.uint8)
        output_path = Path(self.temp_dir) / "output_no_overwrite.mp4"

        # Create video first time
        save_video(frames, output_path, fps=25, overwrite=True)
        original_size = output_path.stat().st_size

        # Try to create again with different frames but overwrite=False
        frames_new = torch.zeros((5, 3, 100, 100), dtype=torch.uint8)
        save_video(frames_new, output_path, fps=25, overwrite=False)

        # Size should be unchanged
        self.assertEqual(output_path.stat().st_size, original_size, "File should not change")

    def test_save_video_custom_fps(self):
        """Test saving video with custom FPS."""
        frames = torch.randint(0, 255, (10, 3, 100, 100), dtype=torch.uint8)
        output_path = Path(self.temp_dir) / "output_custom_fps.mp4"

        save_video(frames, output_path, fps=30)

        self.assertTrue(output_path.exists(), "Video file should be created")


class TestSaveFrames(unittest.TestCase):
    """Tests for save_frames function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_frames_from_tensor(self):
        """Test saving frames from torch.Tensor."""
        frames = torch.randint(0, 255, (10, 3, 100, 100), dtype=torch.uint8)
        output_dir = Path(self.temp_dir) / "frames_tensor"

        save_frames(frames, output_dir)

        saved_frames = sorted(list(output_dir.glob("*.jpg")))
        self.assertEqual(len(saved_frames), 10, "Should save 10 frames")
        self.assertTrue(saved_frames[0].name.startswith("000000"), "First frame should be 000000")

    def test_save_frames_from_numpy_sequence(self):
        """Test saving frames from sequence of numpy arrays."""
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        output_dir = Path(self.temp_dir) / "frames_numpy_seq"

        save_frames(frames, output_dir)

        saved_frames = sorted(list(output_dir.glob("*.jpg")))
        self.assertEqual(len(saved_frames), 5, "Should save 5 frames")

    def test_save_frames_start_number(self):
        """Test saving frames with custom start number."""
        frames = torch.randint(0, 255, (5, 3, 100, 100), dtype=torch.uint8)
        output_dir = Path(self.temp_dir) / "frames_start"

        save_frames(frames, output_dir, start_number=100)

        saved_frames = sorted(list(output_dir.glob("*.jpg")))
        self.assertTrue(saved_frames[0].name.startswith("000100"), "First frame should be 000100")
        self.assertTrue(saved_frames[-1].name.startswith("000104"), "Last frame should be 000104")

    def test_save_frames_zfill(self):
        """Test saving frames with custom zero-padding."""
        frames = torch.randint(0, 255, (3, 3, 100, 100), dtype=torch.uint8)
        output_dir = Path(self.temp_dir) / "frames_zfill"

        save_frames(frames, output_dir, zfill=3)

        saved_frames = sorted(list(output_dir.glob("*.jpg")))
        self.assertTrue(saved_frames[0].name.startswith("000"), "Should use 3-digit padding")
        self.assertEqual(len(saved_frames[0].stem), 3, "Filename should have 3 digits")

    def test_save_frames_extension(self):
        """Test saving frames with custom extension."""
        frames = torch.randint(0, 255, (5, 3, 100, 100), dtype=torch.uint8)
        output_dir = Path(self.temp_dir) / "frames_png"

        save_frames(frames, output_dir, extension=".png")

        saved_frames = list(output_dir.glob("*.png"))
        self.assertEqual(len(saved_frames), 5, "Should save 5 PNG frames")


class TestSaveFramesWithIds(unittest.TestCase):
    """Tests for save_frames_with_ids function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_frames_with_ids_basic(self):
        """Test saving frames with custom IDs."""
        frames = torch.randint(0, 255, (5, 3, 100, 100), dtype=torch.uint8)
        frame_ids = [10, 20, 30, 40, 50]
        output_dir = Path(self.temp_dir) / "frames_ids"

        save_frames_with_ids(frames, frame_ids, output_dir)

        saved_frames = sorted(list(output_dir.glob("*.jpg")))
        self.assertEqual(len(saved_frames), 5, "Should save 5 frames")
        self.assertTrue(saved_frames[0].name.startswith("000010"), "First frame should be 000010")
        self.assertTrue(saved_frames[-1].name.startswith("000050"), "Last frame should be 000050")

    def test_save_frames_with_ids_numpy(self):
        """Test saving frames with numpy array of IDs."""
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        frame_ids = np.array([100, 200, 300])
        output_dir = Path(self.temp_dir) / "frames_ids_numpy"

        save_frames_with_ids(frames, frame_ids, output_dir)

        saved_frames = sorted(list(output_dir.glob("*.jpg")))
        self.assertEqual(len(saved_frames), 3, "Should save 3 frames")
        self.assertTrue(saved_frames[0].name.startswith("000100"), "First frame should be 000100")

    def test_save_frames_with_ids_extension(self):
        """Test saving frames with custom extension."""
        frames = torch.randint(0, 255, (3, 3, 100, 100), dtype=torch.uint8)
        frame_ids = [5, 10, 15]
        output_dir = Path(self.temp_dir) / "frames_ids_png"

        save_frames_with_ids(frames, frame_ids, output_dir, extension=".png")

        saved_frames = list(output_dir.glob("*.png"))
        self.assertEqual(len(saved_frames), 3, "Should save 3 PNG frames")


class TestSequenceToVideo(unittest.TestCase):
    """Tests for sequence_to_video function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sequence_to_video_from_arrays(self):
        """Test video creation from numpy arrays."""
        # Create test frames
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]

        output_path = Path(self.temp_dir) / "output.mp4"
        sequence_to_video(frames, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")
        self.assertGreater(output_path.stat().st_size, 0, "Video file should not be empty")

    def test_sequence_to_video_from_directory(self):
        """Test video creation from directory of frames."""
        # Extract frames first
        frames_dir = Path(self.temp_dir) / "frames"
        video_to_frames(VIDEO_MULTISPEAKER_SHORT, frames_dir)

        # Create video from frames directory
        output_path = Path(self.temp_dir) / "reconstructed.mp4"
        sequence_to_video(frames_dir, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")
        self.assertGreater(output_path.stat().st_size, 0, "Video file should not be empty")

    def test_sequence_to_video_from_paths(self):
        """Test video creation from list of file paths."""
        # Extract frames first
        frames_dir = Path(self.temp_dir) / "frames"
        video_to_frames(VIDEO_MULTISPEAKER_SHORT, frames_dir)

        # Get list of frame paths
        frame_paths = sorted(list(frames_dir.glob("*.png")))

        # Create video from paths
        output_path = Path(self.temp_dir) / "from_paths.mp4"
        sequence_to_video(frame_paths, output_path, fps=25)

        self.assertTrue(output_path.exists(), "Video file should be created")
        self.assertGreater(output_path.stat().st_size, 0, "Video file should not be empty")

    def test_sequence_to_video_no_overwrite(self):
        """Test that overwrite=False skips existing file."""
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        output_path = Path(self.temp_dir) / "output_no_overwrite.mp4"

        # Create video first time
        sequence_to_video(frames, output_path, fps=25, overwrite=True)
        original_size = output_path.stat().st_size

        # Try to create again with different frames but overwrite=False
        frames_new = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        sequence_to_video(frames_new, output_path, fps=25, overwrite=False)

        # Size should be unchanged
        self.assertEqual(output_path.stat().st_size, original_size, "File should not change")


class TestImageToNp(unittest.TestCase):
    """Tests for image_to_np function."""

    def test_image_to_np_from_path(self):
        """Test loading image from path."""
        img = image_to_np(IMAGE_CAT_TIE)
        self.assertEqual(len(img.shape), 3, "Should return 3D array")
        self.assertEqual(img.shape[2], 3, "Should have 3 channels")
        self.assertEqual(img.dtype, np.uint8, "Should be uint8 dtype")

    def test_image_to_np_from_array(self):
        """Test converting numpy array."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = image_to_np(arr)
        self.assertEqual(img.shape, (100, 100, 3), "Should maintain shape")

    def test_image_to_np_grayscale_to_rgb(self):
        """Test converting grayscale to RGB."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = image_to_np(gray)
        self.assertEqual(img.shape, (100, 100, 3), "Should convert to 3 channels")

    def test_image_to_np_channel_order_bgr(self):
        """Test BGR channel order conversion."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = image_to_np(arr, channel_order="BGR")
        self.assertEqual(img.shape, (100, 100, 3), "Should maintain 3 channels")

    def test_image_to_np_channel_order_gray(self):
        """Test grayscale channel order conversion."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = image_to_np(arr, channel_order="GRAY")
        self.assertEqual(img.shape, (100, 100, 3), "Should still have 3 channels after expansion")


class TestImagesToNp(unittest.TestCase):
    """Tests for images_to_np function."""

    def test_images_to_np_basic(self):
        """Test loading multiple images."""
        images = [IMAGE_CAT_TIE, IMAGE_CAT_TIE]
        result = images_to_np(images)
        self.assertEqual(len(result.shape), 4, "Should return 4D array (N, H, W, C)")
        self.assertEqual(result.shape[0], 2, "Should have 2 images")
        self.assertEqual(result.shape[3], 3, "Should have 3 channels")

    def test_images_to_np_with_resize(self):
        """Test loading and resizing multiple images."""
        images = [IMAGE_CAT_TIE, IMAGE_CAT_TIE]
        result = images_to_np(images, resize=(200, 200))
        self.assertEqual(result.shape, (2, 200, 200, 3), "Should resize to 200x200")

    def test_images_to_np_mixed_sources(self):
        """Test loading from mixed sources (paths and arrays)."""
        arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        images = [IMAGE_CAT_TIE, arr]
        result = images_to_np(images, resize=(100, 100))
        self.assertEqual(result.shape[0], 2, "Should have 2 images")
        self.assertEqual(result.shape[1:3], (100, 100), "Should resize both to 100x100")


class TestCheckSameImageDims(unittest.TestCase):
    """Tests for check_same_image_dims function."""

    def test_check_same_dims_valid(self):
        """Test with images of same dimensions."""
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8),
        ]
        # Should not raise
        check_same_image_dims(images)

    def test_check_same_dims_invalid(self):
        """Test with images of different dimensions."""
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((200, 200, 3), dtype=np.uint8),
        ]
        with self.assertRaises(ValueError):
            check_same_image_dims(images)


class TestBatchIterator(unittest.TestCase):
    """Tests for batch_iterator function."""

    def test_batch_iterator_basic(self):
        """Test basic batch iteration."""
        data = list(range(10))
        batches = list(batch_iterator(data, batch_size=3))
        self.assertEqual(len(batches), 4, "Should create 4 batches")
        self.assertEqual(batches[0], [0, 1, 2], "First batch should be [0, 1, 2]")
        self.assertEqual(batches[-1], [9], "Last batch should be [9]")

    def test_batch_iterator_exact_division(self):
        """Test when size divides evenly."""
        data = list(range(12))
        batches = list(batch_iterator(data, batch_size=4))
        self.assertEqual(len(batches), 3, "Should create exactly 3 batches")
        self.assertEqual(len(batches[-1]), 4, "Last batch should be full")

    def test_batch_iterator_empty(self):
        """Test with empty iterable."""
        data = []
        batches = list(batch_iterator(data, batch_size=5))
        self.assertEqual(len(batches), 0, "Should create 0 batches")


class TestRoundTripConversion(unittest.TestCase):
    """Integration tests for round-trip video conversions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_save_load_consistency(self):
        """Test that loading, saving, and loading again produces consistent results."""
        # Load original video
        frames1, fps1 = load_video(VIDEO_MULTISPEAKER_SHORT, end_frame=20)

        # Save to new video
        output_path = Path(self.temp_dir) / "intermediate.mp4"
        save_video(frames1, output_path, fps=fps1)

        # Load the saved video
        frames2, _ = load_video(output_path)

        # Check consistency
        self.assertEqual(frames1.shape[0], frames2.shape[0], "Frame count should match")
        self.assertEqual(frames1.shape[1:], frames2.shape[1:], "Frame dimensions should match")

    def test_extract_reconstruct_cycle(self):
        """Test extracting frames to directory and reconstructing video."""
        # Extract frames
        frames_dir = Path(self.temp_dir) / "frames"
        video_to_frames(VIDEO_MULTISPEAKER_SHORT, frames_dir)

        frame_count = len(list(frames_dir.glob("*.png")))

        # Reconstruct video
        output_path = Path(self.temp_dir) / "reconstructed.mp4"
        sequence_to_video(frames_dir, output_path, fps=25)

        # Verify reconstructed video
        metadata = get_video_metadata(output_path)
        self.assertEqual(
            metadata["num_frames"], frame_count, "Reconstructed video should have same frame count"
        )


class TestVideo(unittest.TestCase):
    """Tests for the stateful Video class."""

    def test_open_and_metadata(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        self.assertGreater(v.num_frames, 0)
        self.assertGreater(v.fps, 0)
        self.assertGreater(v.height, 0)
        self.assertGreater(v.width, 0)
        self.assertGreater(v.duration, 0)
        v.close()

    def test_metadata_matches_get_video_metadata(self):
        meta = get_video_metadata(VIDEO_MULTISPEAKER_SHORT)
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        self.assertEqual(v.num_frames, meta["num_frames"])
        self.assertAlmostEqual(v.fps, meta["fps"], places=2)
        self.assertEqual(v.height, meta["height"])
        self.assertEqual(v.width, meta["width"])
        self.assertAlmostEqual(v.duration, meta["duration"], places=2)
        v.close()

    def test_len(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        self.assertEqual(len(v), v.num_frames)
        v.close()

    def test_getitem_int(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        frame = v[0]
        self.assertEqual(frame.ndim, 3, "Single frame should be 3D (C, H, W)")
        self.assertEqual(frame.shape[0], 3, "Should have 3 channels")
        self.assertEqual(frame.shape[1], v.height)
        self.assertEqual(frame.shape[2], v.width)
        v.close()

    def test_getitem_negative_index(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        last_frame = v[-1]
        last_frame_explicit = v[v.num_frames - 1]
        self.assertTrue(torch.equal(last_frame, last_frame_explicit))
        v.close()

    def test_getitem_index_error(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        with self.assertRaises(IndexError):
            _ = v[v.num_frames]
        with self.assertRaises(IndexError):
            _ = v[-(v.num_frames + 1)]
        v.close()

    def test_getitem_type_error(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        with self.assertRaises(TypeError):
            _ = v["invalid"]
        v.close()

    def test_getitem_slice(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        frames = v[0:5]
        self.assertEqual(frames.ndim, 4, "Slice should return 4D tensor")
        self.assertEqual(frames.shape[0], 5)
        self.assertEqual(frames.shape[1], 3)
        v.close()

    def test_getitem_slice_with_step(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        frames = v[0:10:2]
        self.assertEqual(frames.shape[0], 5, "Should get 5 frames with step 2 over range 10")
        v.close()

    def test_getitem_slice_open_ended(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        frames = v[:]
        self.assertEqual(frames.shape[0], v.num_frames)
        v.close()

    def test_getitem_empty_slice(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        frames = v[5:5]
        self.assertEqual(frames.shape[0], 0)
        v.close()

    def test_get_batch(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        frames = v.get_batch(0, 5)
        self.assertEqual(frames.shape[0], 5)
        self.assertEqual(frames.ndim, 4)
        v.close()

    def test_iter_batches_covers_all_frames(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        total = 0
        for batch in v.iter_batches(batch_size=8):
            self.assertEqual(batch.ndim, 4)
            self.assertEqual(batch.shape[1], 3)
            total += batch.shape[0]
        self.assertEqual(total, v.num_frames, "Should iterate over all frames")
        v.close()

    def test_iter_batches_last_batch_size(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        batch_size = 7
        batches = list(v.iter_batches(batch_size=batch_size))
        if v.num_frames % batch_size != 0:
            self.assertLess(batches[-1].shape[0], batch_size)
        else:
            self.assertEqual(batches[-1].shape[0], batch_size)
        v.close()

    def test_iter_batches_batch_size_larger_than_video(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        batches = list(v.iter_batches(batch_size=v.num_frames + 100))
        self.assertEqual(len(batches), 1, "Should yield a single batch")
        self.assertEqual(batches[0].shape[0], v.num_frames)
        v.close()

    def test_context_manager(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            frame = v[0]
            self.assertEqual(frame.ndim, 3)

    def test_close_releases_decoder(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        v.close()
        self.assertIsNone(v._decoder)

    def test_repr(self):
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        r = repr(v)
        self.assertIn("Video", r)
        self.assertIn("frames=", r)
        self.assertIn("fps=", r)
        v.close()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Video("/nonexistent/path/to/video.mp4")

    def test_consistent_with_load_frames(self):
        frame_ids = [0, 5, 10]
        frames_stateless = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=frame_ids)
        v = Video(VIDEO_MULTISPEAKER_SHORT)
        frames_stateful = torch.stack([v[i] for i in frame_ids])
        self.assertTrue(
            torch.equal(frames_stateless, frames_stateful),
            "Stateful indexed access should match load_frames output",
        )
        v.close()


if __name__ == "__main__":
    unittest.main()
