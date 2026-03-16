import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from exordium.video.core.detection import (
    DetectionFactory,
    DetectionFromImage,
    DetectionFromTensor,
    DetectionFromVideo,
    FrameDetections,
    Track,
    VideoDetections,
    add_detections_to_frame,
    save_detections_to_video,
    save_track_target_to_images,
    visualize_detection,
)
from exordium.video.core.io import video_to_frames
from tests.fixtures import IMAGE_CAT_TIE, VIDEO_MULTISPEAKER_SHORT


class TestDetectionFromImage(unittest.TestCase):
    """Tests for DetectionFromImage class."""

    def setUp(self):
        """Create test detection from image."""
        self.detection = DetectionFromImage(
            frame_id=0,
            source=IMAGE_CAT_TIE,
            score=0.95,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[110, 120], [180, 120], [145, 160], [120, 200], [170, 200]]),
        )

    def test_frame(self):
        """Test loading frame from image."""
        frame = self.detection.frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(len(frame.shape), 3, "Frame should be 3D (H, W, C)")
        self.assertEqual(frame.shape[2], 3, "Frame should have 3 channels")

    def test_bb_crop(self):
        """Test cropping bounding box from image."""
        cropped = self.detection.bb_crop()
        self.assertIsInstance(cropped, np.ndarray)
        self.assertEqual(len(cropped.shape), 3, "Cropped image should be 3D")
        self.assertEqual(cropped.shape[2], 3, "Cropped image should have 3 channels")

    def test_bb_crop_wide(self):
        """Test cropping wider bounding box from image."""
        cropped = self.detection.bb_crop_wide(extra_space=1.5)
        self.assertIsInstance(cropped, np.ndarray)
        self.assertEqual(len(cropped.shape), 3, "Cropped image should be 3D")

        # Wide crop should be larger than normal crop
        normal_crop = self.detection.bb_crop()
        self.assertGreater(cropped.shape[0], normal_crop.shape[0], "Wide crop should be taller")
        self.assertGreater(cropped.shape[1], normal_crop.shape[1], "Wide crop should be wider")

    def test_frame_center(self):
        """Test getting frame center."""
        center = self.detection.frame_center()
        frame = self.detection.frame()
        height, width = frame.shape[:2]

        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(center.shape, (2,), "Center should be (x, y)")
        self.assertEqual(center[0], int(width / 2), "Center x should be width/2")
        self.assertEqual(center[1], int(height / 2), "Center y should be height/2")

    def test_bb_properties(self):
        """Test bounding box property conversions."""
        # Test bb_xyxy property
        bb_xyxy = self.detection.bb_xyxy
        self.assertEqual(bb_xyxy.shape, (4,), "bb_xyxy should have 4 values")

        # Test bb_center property
        center = self.detection.bb_center
        self.assertEqual(center.shape, (2,), "bb_center should have 2 values")

    def test_is_interpolated(self):
        """Test is_interpolated property."""
        self.assertFalse(self.detection.is_interpolated, "Score 0.95 should not be interpolated")

        # Test with interpolated detection (score = -1)
        interpolated = DetectionFromImage(
            frame_id=0,
            source=IMAGE_CAT_TIE,
            score=-1,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.zeros((5, 2)),
        )
        self.assertTrue(interpolated.is_interpolated, "Score -1 should be interpolated")


class TestDetectionFromVideo(unittest.TestCase):
    """Tests for DetectionFromVideo class."""

    @classmethod
    def setUpClass(cls):
        """Set up temporary directory and extract test frames."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.frames_dir = cls.temp_dir / "frames"

        # Extract frames for image-based tests
        video_to_frames(VIDEO_MULTISPEAKER_SHORT, cls.frames_dir, overwrite=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Create test detection from video."""
        self.detection = DetectionFromVideo(
            frame_id=5,
            source=VIDEO_MULTISPEAKER_SHORT,
            score=0.92,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[160, 170], [380, 170], [270, 240], [200, 320], [340, 320]]),
        )

    def test_frame(self):
        """Test loading frame from video."""
        frame = self.detection.frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(len(frame.shape), 3, "Frame should be 3D (H, W, C)")
        self.assertEqual(frame.shape[2], 3, "Frame should have 3 channels")
        self.assertEqual(frame.dtype, np.uint8, "Frame should be uint8")

    def test_bb_crop(self):
        """Test cropping bounding box from video frame."""
        cropped = self.detection.bb_crop()
        self.assertIsInstance(cropped, np.ndarray)
        self.assertEqual(len(cropped.shape), 3, "Cropped image should be 3D")
        self.assertEqual(cropped.shape[2], 3, "Cropped image should have 3 channels")

    def test_bb_crop_wide(self):
        """Test cropping wider bounding box from video frame."""
        cropped = self.detection.bb_crop_wide(extra_space=2.0)
        self.assertIsInstance(cropped, np.ndarray)
        self.assertEqual(len(cropped.shape), 3, "Cropped image should be 3D")

        # Wide crop should be larger than normal crop
        normal_crop = self.detection.bb_crop()
        self.assertGreater(cropped.shape[0], normal_crop.shape[0], "Wide crop should be taller")
        self.assertGreater(cropped.shape[1], normal_crop.shape[1], "Wide crop should be wider")

    def test_frame_center(self):
        """Test getting frame center from video."""
        center = self.detection.frame_center()
        frame = self.detection.frame()
        height, width = frame.shape[:2]

        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(center.shape, (2,), "Center should be (x, y)")
        self.assertEqual(center[0], int(width / 2), "Center x should be width/2")
        self.assertEqual(center[1], int(height / 2), "Center y should be height/2")

    def test_multiple_frames(self):
        """Test loading multiple frames from the same video."""
        # Create multiple detections at different frame IDs
        detections = [
            DetectionFromVideo(
                frame_id=frame_id,
                source=VIDEO_MULTISPEAKER_SHORT,
                score=0.9,
                bb_xywh=np.array([100, 100, 200, 200]),
                landmarks=np.zeros((5, 2)),
            )
            for frame_id in [0, 5, 10, 15, 20]
        ]

        # Test that each detection can load its frame
        for detection in detections:
            frame = detection.frame()
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3, "Frame should be 3D")

            # Test cropping works for each frame
            cropped = detection.bb_crop()
            self.assertIsInstance(cropped, np.ndarray)

    def test_bb_properties(self):
        """Test bounding box property conversions."""
        # Test bb_xyxy property
        bb_xyxy = self.detection.bb_xyxy
        self.assertEqual(bb_xyxy.shape, (4,), "bb_xyxy should have 4 values")

        # Test bb_center property
        center = self.detection.bb_center
        self.assertEqual(center.shape, (2,), "bb_center should have 2 values")


class TestDetectionFromTensor(unittest.TestCase):
    """Tests for DetectionFromTensor class."""

    def setUp(self):
        """Create test detection from tensor."""
        # Create a random frame
        self.frame_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        self.detection = DetectionFromTensor(
            frame_id=0,
            source=self.frame_array,
            score=0.88,
            bb_xywh=np.array([200, 150, 180, 180]),
            landmarks=np.array([[210, 170], [370, 170], [290, 230], [240, 300], [340, 300]]),
        )

    def test_frame(self):
        """Test loading frame from tensor."""
        frame = self.detection.frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertTrue(np.array_equal(frame, self.frame_array), "Frame should match source array")

    def test_bb_crop(self):
        """Test cropping bounding box from tensor."""
        cropped = self.detection.bb_crop()
        self.assertIsInstance(cropped, np.ndarray)
        self.assertEqual(len(cropped.shape), 3, "Cropped image should be 3D")
        self.assertEqual(cropped.shape[2], 3, "Cropped image should have 3 channels")

    def test_bb_crop_wide(self):
        """Test cropping wider bounding box from tensor."""
        cropped = self.detection.bb_crop_wide(extra_space=1.8)
        self.assertIsInstance(cropped, np.ndarray)
        self.assertEqual(len(cropped.shape), 3, "Cropped image should be 3D")

    def test_frame_center(self):
        """Test getting frame center from tensor."""
        center = self.detection.frame_center()
        height, width = self.frame_array.shape[:2]

        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(center.shape, (2,), "Center should be (x, y)")
        self.assertEqual(center[0], int(width / 2), "Center x should be width/2")
        self.assertEqual(center[1], int(height / 2), "Center y should be height/2")


class TestDetectionEquality(unittest.TestCase):
    """Tests for Detection equality comparisons."""

    def test_equal_detections_from_image(self):
        """Test equality of identical DetectionFromImage instances."""
        detection1 = DetectionFromImage(
            frame_id=1,
            source=IMAGE_CAT_TIE,
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        detection2 = DetectionFromImage(
            frame_id=1,
            source=IMAGE_CAT_TIE,
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.assertEqual(detection1, detection2, "Identical detections should be equal")

    def test_unequal_detections_different_bb(self):
        """Test inequality of detections with different bounding boxes."""
        detection1 = DetectionFromImage(
            frame_id=1,
            source=IMAGE_CAT_TIE,
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        detection2 = DetectionFromImage(
            frame_id=1,
            source=IMAGE_CAT_TIE,
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 300]),  # Different width
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.assertNotEqual(
            detection1, detection2, "Detections with different bb should not be equal"
        )

    def test_unequal_detections_different_score(self):
        """Test inequality of detections with different scores."""
        detection1 = DetectionFromImage(
            frame_id=1,
            source=IMAGE_CAT_TIE,
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        detection2 = DetectionFromImage(
            frame_id=1,
            source=IMAGE_CAT_TIE,
            score=0.85,  # Different score
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.assertNotEqual(
            detection1, detection2, "Detections with different scores should not be equal"
        )

    def test_unequal_detections_different_frame_id(self):
        """Test inequality of detections with different frame IDs."""
        detection1 = DetectionFromImage(
            frame_id=1,
            source=IMAGE_CAT_TIE,
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        detection2 = DetectionFromImage(
            frame_id=2,  # Different frame ID
            source=IMAGE_CAT_TIE,
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.assertNotEqual(
            detection1, detection2, "Detections with different frame IDs should not be equal"
        )

    def test_unequal_different_types(self):
        """Test inequality when comparing Detection to non-Detection."""
        detection = DetectionFromImage(
            frame_id=1,
            source=IMAGE_CAT_TIE,
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.assertNotEqual(detection, "not a detection", "Detection should not equal string")
        self.assertNotEqual(detection, 42, "Detection should not equal int")
        self.assertNotEqual(detection, None, "Detection should not equal None")


class FrameDetectionsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        if cls.TMP_DIR.exists():
            shutil.rmtree(cls.TMP_DIR)

    def setUp(self):
        self.frame_detections = FrameDetections()

    def test_add_dict_valid_detection(self):
        detection_dict = {
            "frame_id": 1,
            "source": "path/to/frame.jpg",
            "score": 0.8,
            "bb_xywh": np.array([100, 100, 200, 200]),
            "landmarks": np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        }
        self.frame_detections.add_dict(detection_dict)
        self.assertEqual(len(self.frame_detections), 1)

    def test_add_dict_invalid_detection_missing_key(self):
        detection_dict = {
            "frame_id": 1,
            "source": "path/to/frame.jpg",
            "score": 0.8,
            # 'bb_xywh': np.array([100, 100, 200, 200]), # Missing key
            "landmarks": np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        }
        with self.assertRaises(KeyError):
            self.frame_detections.add_dict(detection_dict)

    def test_add_detection(self):
        detection = DetectionFromImage(
            frame_id=1,
            source="path/to/frame.jpg",
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        )
        self.frame_detections.add(detection)
        self.assertEqual(len(self.frame_detections), 1)

    def test_len(self):
        detection = DetectionFromImage(
            frame_id=1,
            source="path/to/frame.jpg",
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        )
        self.frame_detections.add(detection)
        self.assertEqual(len(self.frame_detections), 1)

    def test_getitem(self):
        detection = DetectionFromImage(
            frame_id=1,
            source="path/to/frame.jpg",
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        )
        self.frame_detections.add(detection)
        retrieved_detection = self.frame_detections[0]
        self.assertEqual(retrieved_detection, detection)

    def test_getitem_index_out_of_range(self):
        with self.assertRaises(IndexError):
            self.frame_detections[0]

    def test_iteration(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source="path/to/frame1.jpg",
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source="path/to/frame2.jpg",
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.frame_detections.add(detection1)
        self.frame_detections.add(detection2)
        detections = []
        for detection in self.frame_detections:
            detections.append(detection)
        self.assertEqual(detections, [detection1, detection2])

    def test_get_biggest_bb(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source="path/to/frame1.jpg",
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source="path/to/frame2.jpg",
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.frame_detections.add(detection1)
        self.frame_detections.add(detection2)
        biggest_bb = self.frame_detections.get_detection_with_biggest_bb()
        self.assertEqual(biggest_bb, detection2)

    def test_get_highest_score(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source="path/to/frame1.jpg",
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source="path/to/frame2.jpg",
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.frame_detections.add(detection1)
        self.frame_detections.add(detection2)
        highest_score = self.frame_detections.get_detection_with_highest_score()
        self.assertEqual(highest_score, detection2)

    def test_equal(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source="path/to/frame1.jpg",
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source="path/to/frame2.jpg",
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.frame_detections.add(detection1)
        self.frame_detections.add(detection2)
        other_frame_detections1 = FrameDetections()
        other_frame_detections1.add(detection1)
        other_frame_detections1.add(detection2)
        other_frame_detections2 = FrameDetections()
        other_frame_detections2.add(detection1)
        other_frame_detections2.add(detection1)
        self.assertEqual(self.frame_detections, other_frame_detections1)
        self.assertNotEqual(self.frame_detections, other_frame_detections2)
        self.assertNotEqual(other_frame_detections1, other_frame_detections2)

    def test_save_and_load(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source="path/to/frame1.jpg",
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source="path/to/frame2.jpg",
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]),
        )
        self.frame_detections.add(detection1)
        self.frame_detections.add(detection2)
        output_file = self.TMP_DIR / "output.csv"
        self.frame_detections.save(output_file)
        loaded_detections = FrameDetections().load(output_file)
        self.assertEqual(len(loaded_detections), 2)
        self.assertEqual(loaded_detections[0], detection1)
        self.assertEqual(loaded_detections[1], detection2)


class VideoDetectionsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        if cls.TMP_DIR.exists():
            shutil.rmtree(cls.TMP_DIR)

    def setUp(self):
        self.video_detections = VideoDetections()
        self.detection1 = DetectionFromImage(
            frame_id=0,
            source="frame1.jpg",
            score=0.9,
            bb_xywh=np.zeros((4,)),
            landmarks=np.zeros((5, 2)),
        )
        self.detection2 = DetectionFromImage(
            frame_id=1,
            source="frame2.jpg",
            score=0.8,
            bb_xywh=np.zeros((4,)),
            landmarks=np.zeros((5, 2)),
        )
        self.detection3 = DetectionFromImage(
            frame_id=0,
            source="frame1.jpg",
            score=0.9,
            bb_xywh=np.zeros((4,)),
            landmarks=np.zeros((5, 2)),
        )
        self.detection4 = DetectionFromImage(
            frame_id=1,
            source="frame2.jpg",
            score=0.8,
            bb_xywh=np.zeros((4,)),
            landmarks=np.zeros((5, 2)),
        )
        self.detection5 = DetectionFromImage(
            frame_id=2,
            source="frame2.jpg",
            score=0.7,
            bb_xywh=np.zeros((4,)),
            landmarks=np.zeros((5, 2)),
        )
        self.detection6 = DetectionFromImage(
            frame_id=3,
            source="frame3.jpg",
            score=0.7,
            bb_xywh=np.zeros((4,)),
            landmarks=np.zeros((5, 2)),
        )

    def test_add(self):
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1.add(self.detection1))
        self.video_detections.add(fdet2.add(self.detection2))
        self.assertEqual(len(self.video_detections), 2)
        self.assertEqual(self.video_detections[0], fdet1)
        self.assertEqual(self.video_detections[1], fdet2)

    def test_add_empty(self):
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1)
        self.video_detections.add(fdet2)
        self.assertEqual(len(self.video_detections), 0)

    def test_iterator(self):
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1.add(self.detection1))
        self.video_detections.add(fdet2.add(self.detection2))
        for idx, frame_detections in enumerate(self.video_detections):
            self.assertEqual(frame_detections, self.video_detections[idx])

    def test_equal(self):
        # instance 1
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1.add(self.detection1))
        self.video_detections.add(fdet2.add(self.detection2))
        # instance 2
        other_video_detections1 = VideoDetections()
        fdet3 = FrameDetections()
        fdet4 = FrameDetections()
        other_video_detections1.add(fdet3.add(self.detection3))
        other_video_detections1.add(fdet4.add(self.detection4))
        # instance 3
        other_video_detections2 = VideoDetections()
        fdet5 = FrameDetections()
        fdet6 = FrameDetections()
        other_video_detections2.add(fdet5.add(self.detection5))
        other_video_detections2.add(fdet6.add(self.detection6))

        self.assertEqual(self.video_detections, other_video_detections1)
        self.assertNotEqual(self.video_detections, other_video_detections2)
        self.assertNotEqual(other_video_detections1, other_video_detections2)

    def test_save_and_load(self):
        detection1 = DetectionFromImage(
            frame_id=0,
            source="000000.png",
            score=0.9,
            bb_xywh=np.array([10, 20, 30, 40]),
            landmarks=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        )
        detection2 = DetectionFromImage(
            frame_id=1,
            source="000001.png",
            score=0.8,
            bb_xywh=np.array([15, 25, 35, 45]),
            landmarks=np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]),
        )
        detection3 = DetectionFromImage(
            frame_id=1,
            source="000001.png",
            score=0.7,
            bb_xywh=np.array([30, 40, 20, 20]),
            landmarks=np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]),
        )
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1.add(detection1))
        self.video_detections.add(fdet2.add(detection2).add(detection3))
        output_file = self.TMP_DIR / "output.csv"
        self.video_detections.save(output_file)
        loaded_video_detections = VideoDetections().load(output_file)
        self.assertEqual(self.video_detections, loaded_video_detections)


class TestDetectionFactory(unittest.TestCase):
    def _make_det(self, source):
        return dict(
            frame_id=0,
            source=source,
            score=0.9,
            bb_xywh=np.array([10, 10, 50, 50]),
            landmarks=np.zeros((5, 2)),
        )

    def test_unsupported_extension_raises(self):
        with self.assertRaises(ValueError):
            DetectionFactory.create_detection(**self._make_det("file.xyz"))

    def test_tensor_source(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = DetectionFactory.create_detection(**self._make_det(frame))
        self.assertIsInstance(det, DetectionFromTensor)


class TestFrameDetectionsProperties(unittest.TestCase):
    def _make_fdet(self, frame_id=1, source="path/to/frame.jpg"):
        fd = FrameDetections()
        fd.add(
            DetectionFromImage(
                frame_id=frame_id,
                source=source,
                score=0.9,
                bb_xywh=np.array([10, 10, 50, 50]),
                landmarks=np.zeros((5, 2)),
            )
        )
        return fd

    def test_frame_id_property(self):
        fd = self._make_fdet(frame_id=7)
        self.assertEqual(fd.frame_id, 7)

    def test_source_property(self):
        fd = self._make_fdet(source="test_frame.jpg")
        self.assertEqual(fd.source, "test_frame.jpg")

    def test_equality_same_content(self):
        fd1 = self._make_fdet(frame_id=3)
        fd2 = self._make_fdet(frame_id=3)
        self.assertEqual(fd1, fd2)

    def test_equality_non_instance(self):
        fd = self._make_fdet()
        self.assertNotEqual(fd, "not a FrameDetections")


class TestVideoDetectionsProperties(unittest.TestCase):
    def _make_vdet(self, n=3):
        vdet = VideoDetections()
        for i in range(n):
            fd = FrameDetections()
            fd.add(
                DetectionFromImage(
                    frame_id=i,
                    source=f"frame{i}.jpg",
                    score=0.9,
                    bb_xywh=np.array([10, 10, 50, 50]),
                    landmarks=np.zeros((5, 2)),
                )
            )
            vdet.add(fd)
        return vdet

    def test_frame_ids(self):
        vdet = self._make_vdet(3)
        self.assertEqual(vdet.frame_ids(), [0, 1, 2])

    def test_get_frame_detection_with_frame_id(self):
        vdet = self._make_vdet(3)
        fd = vdet.get_frame_detection_with_frame_id(1)
        self.assertEqual(fd.frame_id, 1)

    def test_equality_non_instance(self):
        vdet = self._make_vdet(2)
        self.assertNotEqual(vdet, "not a VideoDetections")


class TestTrackMiscMethods(unittest.TestCase):
    def _make_track(self, frame_ids, bb_x_offset=0):
        track = Track(track_id=0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for fid in frame_ids:
            det = DetectionFactory.create_detection(
                frame_id=fid,
                source=frame,
                score=0.9,
                bb_xywh=np.array([10 + bb_x_offset, 10, 50, 50]),
                landmarks=np.zeros((5, 2)),
            )
            track.add(det)
        return track

    def test_frame_distance_overlapping_returns_zero(self):
        t1 = self._make_track([0, 1, 2, 5])
        t2 = self._make_track([3, 4])
        # t1 ends at frame 5, t2 starts at frame 3 — overlapping
        self.assertEqual(t1.frame_distance(t2), 0)

    def test_sample_fewer_than_requested(self):
        track = self._make_track([0, 1])
        # Request more samples than available
        samples = track.sample(num=10)
        self.assertEqual(len(samples), 2)

    def test_str(self):
        track = self._make_track([0, 1, 2])
        s = str(track)
        self.assertIsInstance(s, str)
        self.assertIn("0", s)


class TestVisualizationFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = Path(tempfile.mkdtemp())
        # Build a frame dir with numbered images for save_* tests
        cls.frame_dir = cls.TMP_DIR / "frames"
        cls.frame_dir.mkdir()
        cls.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(5):
            cv2.imwrite(str(cls.frame_dir / f"{i:06d}.jpg"), cls.frame)

        # Build matching VideoDetections and Track
        cls.vdet = VideoDetections()
        cls.track = Track(track_id=0)
        landmarks = np.zeros((5, 2))
        for i in range(5):
            fd = FrameDetections()
            det = DetectionFromImage(
                frame_id=i,
                source=str(cls.frame_dir / f"{i:06d}.jpg"),
                score=0.9,
                bb_xywh=np.array([5, 5, 40, 40]),
                landmarks=landmarks,
            )
            fd.add(det)
            cls.vdet.add(fd)
            cls.track.add(det)

    @classmethod
    def tearDownClass(cls):
        if cls.TMP_DIR.exists():
            shutil.rmtree(cls.TMP_DIR)

    def test_add_detections_to_frame(self):
        fd = self.vdet[0]
        result = add_detections_to_frame(fd)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 3)

    def test_add_detections_to_frame_with_frame(self):
        fd = self.vdet[0]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = add_detections_to_frame(fd, frame=frame)
        self.assertIsInstance(result, np.ndarray)

    def test_save_detections_to_video(self):
        output_dir = self.TMP_DIR / "vdet_output"
        save_detections_to_video(self.vdet, self.frame_dir, output_dir)
        self.assertTrue(output_dir.exists())

    def test_save_track_target_to_images(self):
        output_dir = self.TMP_DIR / "track_output"
        save_track_target_to_images(self.track, output_dir, bb_size=-1)
        self.assertTrue(output_dir.exists())

    def test_visualize_detection(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = DetectionFromTensor(
            frame_id=0,
            source=frame,
            score=0.9,
            bb_xywh=np.array([5, 5, 40, 40]),
            landmarks=np.array([[10, 10], [20, 10], [15, 20], [10, 25], [20, 25]]),
        )
        result = visualize_detection(det)
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
