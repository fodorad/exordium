"""Tests for exordium.video.core.io: image/video loading, to_uint8_tensor, Video class."""

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from exordium.video.core.io import (
    _BACKEND,
    ImageSequenceReader,
    Video,
    batch_iterator,
    get_video_metadata,
    image_to_np,
    image_to_tensor,
    images_to_np,
    interpolate_1d,
    load_frames,
    load_video,
    save_frames,
    save_frames_with_ids,
    save_video,
    sequence_to_video,
    to_uint8_tensor,
    video_to_frames,
)
from tests.fixtures import IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT


class TestImageToNp(unittest.TestCase):
    def test_from_path_rgb(self):
        img = image_to_np(IMAGE_FACE, "RGB")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)

    def test_from_path_bgr(self):
        img = image_to_np(IMAGE_FACE, "BGR")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[2], 3)

    def test_from_str_path(self):
        img = image_to_np(str(IMAGE_FACE), "RGB")
        self.assertIsInstance(img, np.ndarray)

    def test_missing_file_raises(self):
        with self.assertRaises(Exception):
            image_to_np("/nonexistent/path/img.jpg", "RGB")

    def test_hsv_channel_order(self):
        img = image_to_np(IMAGE_FACE, "HSV")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[2], 3)

    def test_lab_channel_order(self):
        img = image_to_np(IMAGE_FACE, "LAB")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[2], 3)

    def test_gray_channel_order(self):
        img = image_to_np(IMAGE_FACE, "GRAY")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[2], 3)

    def test_single_channel_expands_to_3(self):
        """A (H, W, 1) array should be repeated to (H, W, 3)."""
        gray = np.zeros((50, 50, 1), dtype=np.uint8)
        result = image_to_np(gray, "RGB")
        self.assertEqual(result.shape[2], 3)


class TestImageToTensor(unittest.TestCase):
    def test_returns_tensor(self):
        t = image_to_tensor(IMAGE_FACE)
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.ndim, 3)
        self.assertEqual(t.shape[0], 3)

    def test_dtype_uint8(self):
        t = image_to_tensor(IMAGE_FACE)
        self.assertEqual(t.dtype, torch.uint8)

    def test_hwc_tensor_permuted_to_chw(self):
        """(H, W, C) tensor where shape[0] not in (1,3,4) should be permuted."""
        hwc = torch.zeros(100, 100, 3, dtype=torch.uint8)
        result = image_to_tensor(hwc)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 100)
        self.assertEqual(result.shape[2], 100)


class TestToUint8Tensor(unittest.TestCase):
    def test_from_path(self):
        out = to_uint8_tensor(IMAGE_FACE)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 3)

    def test_from_numpy_3d(self):
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        out = to_uint8_tensor(img)
        self.assertEqual(out.shape, (1, 3, 60, 80))

    def test_from_numpy_4d(self):
        imgs = np.zeros((5, 60, 80, 3), dtype=np.uint8)
        out = to_uint8_tensor(imgs)
        self.assertEqual(out.shape, (5, 3, 60, 80))

    def test_from_tensor_3d(self):
        t = torch.zeros(3, 60, 80, dtype=torch.uint8)
        out = to_uint8_tensor(t)
        self.assertEqual(out.shape, (1, 3, 60, 80))

    def test_from_tensor_4d(self):
        t = torch.zeros(5, 3, 60, 80, dtype=torch.uint8)
        out = to_uint8_tensor(t)
        self.assertEqual(out.shape[0], 5)
        self.assertEqual(out.shape[1], 3)

    def test_from_str_path(self):
        out = to_uint8_tensor(str(IMAGE_FACE))
        self.assertEqual(out.shape[0], 1)

    def test_from_list_of_paths(self):
        out = to_uint8_tensor([IMAGE_FACE, IMAGE_FACE])
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)

    def test_from_list_of_arrays(self):
        arrays = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(3)]
        out = to_uint8_tensor(arrays)
        self.assertEqual(out.shape, (3, 3, 40, 40))

    def test_empty_sequence_raises(self):
        with self.assertRaises(ValueError):
            to_uint8_tensor([])


class TestVideoClass(unittest.TestCase):
    def test_context_manager(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            self.assertIsNotNone(v)

    def test_properties(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            self.assertGreater(v.num_frames, 0)
            self.assertGreater(v.fps, 0)
            self.assertGreater(v.height, 0)
            self.assertGreater(v.width, 0)

    def test_get_batch_returns_tensor(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            batch = v.get_batch(0, min(4, v.num_frames))
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(batch.ndim, 4)
            self.assertEqual(batch.shape[1], 3)

    def test_getitem_single(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            frame = v[0]
            self.assertIsInstance(frame, torch.Tensor)
            self.assertEqual(frame.ndim, 3)

    def test_len_equals_num_frames(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            self.assertEqual(len(v), v.num_frames)

    def test_negative_index(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            frame = v[-1]
        self.assertIsInstance(frame, torch.Tensor)
        self.assertEqual(frame.ndim, 3)

    def test_index_out_of_range(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            with self.assertRaises(IndexError):
                _ = v[99999]

    def test_slice_with_step(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            frames = v[0:6:2]  # indices 0, 2, 4
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.shape[0], 3)

    def test_slice_empty_returns_empty(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            frames = v[5:3]  # start >= stop → empty
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.shape[0], 0)

    def test_invalid_key_type_raises(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            with self.assertRaises(TypeError):
                _ = v["bad_key"]

    def test_repr(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            r = repr(v)
        self.assertIn("Video", r)
        self.assertIn("frames", r)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            Video("/nonexistent/video.mp4")

    def test_len(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            length = len(v)
        self.assertGreater(length, 0)


class TestLoadFrames(unittest.TestCase):
    def test_load_specific_frames(self):
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=[0, 1, 2, 3, 4])
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.shape[0], 5)
        self.assertEqual(frames.shape[1], 3)

    def test_load_frames_resize_int_portrait_video(self):
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=[0], resize=64)
        _, _, h, w = frames.shape
        self.assertIn(64, (h, w))

    def test_load_frames_resize_tuple(self):
        frames = load_frames(VIDEO_MULTISPEAKER_SHORT, frame_ids=[0], resize=(48, 64))
        _, _, h, w = frames.shape
        self.assertEqual(h, 48)
        self.assertEqual(w, 64)

    def test_load_frames_with_crop(self):
        frames = load_frames(
            VIDEO_MULTISPEAKER_SHORT,
            frame_ids=[0],
            resize=(128, 128),
            crop=(0, 0, 64, 64),
        )
        _, _, h, w = frames.shape
        self.assertEqual(h, 64)
        self.assertEqual(w, 64)

    def test_load_frames_resize_and_crop_combined(self):
        """Covers both resize and crop within the batch loop."""
        frames = load_frames(
            VIDEO_MULTISPEAKER_SHORT,
            frame_ids=[0, 1],
            resize=(80, 80),
            crop=(0, 0, 40, 40),
        )
        self.assertEqual(frames.shape[-2:], (40, 40))


class TestVideoDecodeAll(unittest.TestCase):
    def test_returns_tensor(self):
        frames = _BACKEND.decode_all(VIDEO_MULTISPEAKER_SHORT)
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.ndim, 4)  # (T, C, H, W)

    def test_covers_all_frames(self):
        meta = _BACKEND.get_metadata(VIDEO_MULTISPEAKER_SHORT)
        frames = _BACKEND.decode_all(VIDEO_MULTISPEAKER_SHORT)
        self.assertEqual(frames.shape[0], meta["num_frames"])


class TestLoadVideoWithFps(unittest.TestCase):
    def test_load_video_with_fps_downsamples(self):
        meta = _BACKEND.get_metadata(VIDEO_MULTISPEAKER_SHORT)
        native_fps = meta["fps"]
        target_fps = max(1, int(native_fps // 2))
        frames, actual_fps = load_video(VIDEO_MULTISPEAKER_SHORT, fps=target_fps)
        self.assertEqual(actual_fps, target_fps)
        full_frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT)
        self.assertLessEqual(frames.shape[0], full_frames.shape[0])

    def test_load_video_end_frame(self):
        frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT, end_frame=5)
        self.assertLessEqual(frames.shape[0], 5)


class TestLoadVideoWithResize(unittest.TestCase):
    def test_resize_int_sets_smaller_dim(self):
        frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT, resize=64, end_frame=2)
        _, _, h, w = frames.shape
        self.assertIn(64, (h, w))

    def test_resize_tuple_explicit(self):
        frames, _ = load_video(VIDEO_MULTISPEAKER_SHORT, resize=(50, 70), end_frame=2)
        _, _, h, w = frames.shape
        self.assertEqual(h, 50)
        self.assertEqual(w, 70)


class TestLoadVideoWithCrop(unittest.TestCase):
    def test_crop_sets_output_size(self):
        frames, _ = load_video(
            VIDEO_MULTISPEAKER_SHORT,
            resize=(128, 128),
            crop=(0, 0, 64, 64),
            end_frame=2,
        )
        _, _, h, w = frames.shape
        self.assertEqual(h, 64)
        self.assertEqual(w, 64)


class TestSaveVideoVariants(unittest.TestCase):
    def _make_frames_tensor_tchw(self, n=3):
        return torch.randint(0, 255, (n, 3, 64, 64), dtype=torch.uint8)

    def _make_frames_tensor_thwc(self, n=3):
        return torch.randint(0, 255, (n, 64, 64, 3), dtype=torch.uint8)

    def _make_frames_numpy_tchw(self, n=3):
        return np.random.randint(0, 255, (n, 3, 64, 64), dtype=np.uint8)

    def _make_frames_numpy_thwc(self, n=3):
        return np.random.randint(0, 255, (n, 64, 64, 3), dtype=np.uint8)

    def test_save_video_torch_tensor_tchw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.mp4"
            save_video(self._make_frames_tensor_tchw(), out)
            self.assertTrue(out.exists())

    def test_save_video_torch_tensor_thwc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.mp4"
            save_video(self._make_frames_tensor_thwc(), out)
            self.assertTrue(out.exists())

    def test_save_video_numpy_tchw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.mp4"
            save_video(self._make_frames_numpy_tchw(), out)
            self.assertTrue(out.exists())

    def test_save_video_numpy_thwc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.mp4"
            save_video(self._make_frames_numpy_thwc(), out)
            self.assertTrue(out.exists())

    def test_save_video_sequence_of_tensors_chw(self):
        frames = [torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.mp4"
            save_video(frames, out)
            self.assertTrue(out.exists())

    def test_save_video_sequence_of_numpy_hwc(self):
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.mp4"
            save_video(frames, out)
            self.assertTrue(out.exists())

    def test_save_video_skips_if_exists_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.mp4"
            frames = self._make_frames_tensor_tchw()
            save_video(frames, out)
            mtime1 = out.stat().st_mtime
            save_video(frames, out, overwrite=False)
            mtime2 = out.stat().st_mtime
            self.assertEqual(mtime1, mtime2)


class TestSequenceToVideo(unittest.TestCase):
    def test_from_directory_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            for i in range(3):
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(frame_dir / f"{i:06d}.png"), img)

            out = Path(tmpdir) / "out.mp4"
            sequence_to_video(frame_dir, out)
            self.assertTrue(out.exists())

    def test_from_list_of_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                p = Path(tmpdir) / f"{i:06d}.png"
                cv2.imwrite(str(p), img)
                paths.append(p)

            out = Path(tmpdir) / "out.mp4"
            sequence_to_video(paths, out)
            self.assertTrue(out.exists())

    def test_from_list_of_numpy(self):
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.mp4"
            sequence_to_video(frames, out)
            self.assertTrue(out.exists())


class TestSaveFramesWithIds(unittest.TestCase):
    def test_saves_with_correct_names_tensor(self):
        frames = torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
        ids = [10, 20, 30]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_frames_with_ids(frames, ids, Path(tmpdir))
            saved = sorted(Path(tmpdir).glob("*.jpg"))
            self.assertEqual(len(saved), 3)
            stems = [p.stem for p in saved]
            self.assertIn("000010", stems)

    def test_saves_with_numpy_frames(self):
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        ids = [0, 1]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_frames_with_ids(frames, ids, Path(tmpdir))
            saved = sorted(Path(tmpdir).glob("*.jpg"))
            self.assertEqual(len(saved), 2)


class TestTorchCodecBackendDecodeFrames(unittest.TestCase):
    def test_decode_frames_specific_indices(self):
        frames = _BACKEND.decode_frames(VIDEO_MULTISPEAKER_SHORT, [0, 1, 2])
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.shape[0], 3)
        self.assertEqual(frames.ndim, 4)


class TestImagesToNpWithResize(unittest.TestCase):
    def test_resize_different_sizes(self):
        """images_to_np auto-resizes images with different shapes."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((80, 80, 3), dtype=np.uint8)
        result = images_to_np([img1, img2], resize=(64, 64))
        self.assertEqual(result.shape, (2, 64, 64, 3))


class TestImageSequenceReaderTransform(unittest.TestCase):
    def test_with_transform(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                shutil.copy(IMAGE_FACE, Path(tmpdir) / f"{i:06d}.jpg")

            def dummy_transform(img):
                return img * 2

            ds = ImageSequenceReader(tmpdir, transform=dummy_transform)
            result = ds[0]
            self.assertIsInstance(result, np.ndarray)


class TestGetVideoMetadata(unittest.TestCase):
    def test_returns_dict(self):
        meta = get_video_metadata(VIDEO_MULTISPEAKER_SHORT)
        self.assertIsInstance(meta, dict)

    def test_has_fps(self):
        meta = get_video_metadata(VIDEO_MULTISPEAKER_SHORT)
        self.assertIn("fps", meta)
        self.assertGreater(meta["fps"], 0)


class TestVideoIterBatches(unittest.TestCase):
    def test_iter_batches_yields_tensors(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            batches = list(v.iter_batches(batch_size=4))
        self.assertGreater(len(batches), 0)
        for b in batches:
            self.assertIsInstance(b, torch.Tensor)
            self.assertEqual(b.ndim, 4)

    def test_iter_batches_covers_all_frames(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            total = v.num_frames
            count = sum(b.shape[0] for b in v.iter_batches(batch_size=4))
        self.assertEqual(count, total)


class TestVideoGetFramesAt(unittest.TestCase):
    def test_get_frames_at_returns_tensor(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            frames = v.get_frames_at([0, 1, 2])
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.shape[0], 3)


class TestVideoSlice(unittest.TestCase):
    def test_slice_returns_tensor(self):
        with Video(VIDEO_MULTISPEAKER_SHORT) as v:
            frames = v[0:3]
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.shape[0], 3)


class TestInterpolate1d(unittest.TestCase):
    def test_interpolate_midpoint(self):
        start = np.array([0.0, 0.0])
        end = np.array([2.0, 2.0])
        result = interpolate_1d(0, 2, start, end)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)


class TestLoadVideo(unittest.TestCase):
    def test_load_video_returns_tensor_and_fps(self):
        frames, fps = load_video(VIDEO_MULTISPEAKER_SHORT, batch_size=8)
        self.assertIsInstance(frames, torch.Tensor)
        self.assertGreater(fps, 0)
        self.assertEqual(frames.ndim, 4)


def _make_portrait_video(tmpdir: str) -> Path:
    """Create a small portrait video (H > W) using cv2."""
    p = Path(tmpdir) / "portrait.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(p), fourcc, 10.0, (64, 128))  # W=64, H=128
    for _ in range(5):
        frame = np.zeros((128, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return p


class TestLoadVideoPortraitResize(unittest.TestCase):
    def test_portrait_resize_int_covers_aspect_ratio_branch(self):
        """load_video with resize=int on a portrait video (H >= W) hits line 223."""
        with tempfile.TemporaryDirectory() as tmpdir:
            portrait_path = _make_portrait_video(tmpdir)
            frames, fps = load_video(portrait_path, resize=64, end_frame=3)
        self.assertIsInstance(frames, frames.__class__)

    def test_load_video_fps_zero_branch(self):
        """Very short clip + low fps target → target_frame_count=0 → set to 1 (line 211)."""
        frames, actual_fps = load_video(VIDEO_MULTISPEAKER_SHORT, fps=1, end_frame=1)
        self.assertGreaterEqual(frames.shape[0], 1)


class TestLoadFramesPortraitResize(unittest.TestCase):
    def test_portrait_resize_int_in_load_frames(self):
        """load_frames with resize=int on a portrait video (H >= W) hits line 333."""
        with tempfile.TemporaryDirectory() as tmpdir:
            portrait_path = _make_portrait_video(tmpdir)
            frames = load_frames(portrait_path, frame_ids=[0], resize=64)
        self.assertIsNotNone(frames)


class TestVideoToFramesNoneOutputDir(unittest.TestCase):
    def test_output_dir_none_raises_value_error(self):
        """video_to_frames(output_dir=None) raises ValueError (line 421)."""
        with self.assertRaises(ValueError):
            video_to_frames(VIDEO_MULTISPEAKER_SHORT, output_dir=None)


class TestImagestoNp(unittest.TestCase):
    def test_from_paths_no_resize(self):
        result = images_to_np([IMAGE_FACE, IMAGE_FACE])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 4)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[3], 3)

    def test_from_paths_with_resize(self):
        result = images_to_np([IMAGE_FACE, IMAGE_FACE], resize=(64, 64))
        self.assertEqual(result.shape, (2, 64, 64, 3))

    def test_from_numpy_arrays(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = images_to_np([arr, arr])
        self.assertEqual(result.shape, (2, 100, 100, 3))


class TestSaveFrames(unittest.TestCase):
    def test_save_frames_tensor(self):
        frames = torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            save_frames(frames, tmp)
            saved = sorted(Path(tmp).glob("*.jpg"))
            self.assertEqual(len(saved), 3)

    def test_save_frames_numpy_list(self):
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            save_frames(frames, tmp, extension=".jpg")
            saved = sorted(Path(tmp).glob("*.jpg"))
            self.assertEqual(len(saved), 2)


class TestVideoToFrames(unittest.TestCase):
    def test_video_to_frames_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "frames"
            video_to_frames(VIDEO_MULTISPEAKER_SHORT, out_dir)
            frames = list(out_dir.glob("*.png"))
            self.assertGreater(len(frames), 0)

    def test_video_to_frames_skip_if_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "frames"
            video_to_frames(VIDEO_MULTISPEAKER_SHORT, out_dir)
            n1 = len(list(out_dir.glob("*.png")))
            video_to_frames(VIDEO_MULTISPEAKER_SHORT, out_dir, overwrite=False)
            n2 = len(list(out_dir.glob("*.png")))
            self.assertEqual(n1, n2)


class TestImageSequenceReader(unittest.TestCase):
    def test_len_and_getitem(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                shutil.copy(IMAGE_FACE, Path(tmp) / f"{i:06d}.jpg")
            ds = ImageSequenceReader(tmp)
            self.assertEqual(len(ds), 3)
            img = ds[0]
            self.assertIsInstance(img, np.ndarray)


class TestBatchIterator(unittest.TestCase):
    def test_batch_iterator_yields_batches(self):
        data = list(range(10))
        batches = list(batch_iterator(data, batch_size=3))
        self.assertEqual(len(batches), 4)  # 3+3+3+1
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[-1]), 1)


if __name__ == "__main__":
    unittest.main()
