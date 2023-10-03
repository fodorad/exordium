import unittest
import os
import numpy as np
from exordium import DATA_DIR
from exordium.video.detection import DetectionFromImage, FrameDetections


class FrameDetectionsTestCase(unittest.TestCase):

    def setUp(self):
        self.frame_detections = FrameDetections()

    def test_add_dict_valid_detection(self):
        detection_dict = {
            'frame_id': 1,
            'source': 'path/to/frame.jpg',
            'score': 0.8,
            'bb_xywh': np.array([100, 100, 200, 200]),
            'landmarks': np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        }
        self.frame_detections.add_dict(detection_dict)
        self.assertEqual(len(self.frame_detections), 1)

    def test_add_dict_invalid_detection_missing_key(self):
        detection_dict = {
            'frame_id': 1,
            'source': 'path/to/frame.jpg',
            'score': 0.8,
            # 'bb_xywh': np.array([100, 100, 200, 200]), # Missing key
            'landmarks': np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        }
        with self.assertRaises(KeyError):
            self.frame_detections.add_dict(detection_dict)

    def test_add_detection(self):
        detection = DetectionFromImage(
            frame_id=1,
            source='path/to/frame.jpg',
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        )
        self.frame_detections.add(detection)
        self.assertEqual(len(self.frame_detections), 1)

    def test_len(self):
        detection = DetectionFromImage(
            frame_id=1,
            source='path/to/frame.jpg',
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        )
        self.frame_detections.add(detection)
        self.assertEqual(len(self.frame_detections), 1)

    def test_getitem(self):
        detection = DetectionFromImage(
            frame_id=1,
            source='path/to/frame.jpg',
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        )
        self.frame_detections.add(detection)
        retrieved_detection = self.frame_detections[0]
        self.assertEqual(retrieved_detection, detection)

    def test_getitem_index_out_of_range(self):
        with self.assertRaises(IndexError):
            retrieved_detection = self.frame_detections[0]

    def test_iteration(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source='path/to/frame1.jpg',
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source='path/to/frame2.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
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
            source='path/to/frame1.jpg',
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source='path/to/frame2.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        self.frame_detections.add(detection1)
        self.frame_detections.add(detection2)
        biggest_bb = self.frame_detections.get_detection_with_biggest_bb()
        self.assertEqual(biggest_bb, detection2)

    def test_get_highest_score(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source='path/to/frame1.jpg',
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source='path/to/frame2.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        self.frame_detections.add(detection1)
        self.frame_detections.add(detection2)
        highest_score = self.frame_detections.get_detection_with_highest_score()
        self.assertEqual(highest_score, detection2)

    def test_equal(self):
        detection1 = DetectionFromImage(
            frame_id=1,
            source='path/to/frame1.jpg',
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source='path/to/frame2.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
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
            source='path/to/frame1.jpg',
            score=0.8,
            bb_xywh=np.array([100, 100, 200, 200]),
            landmarks=np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
        )
        detection2 = DetectionFromImage(
            frame_id=2,
            source='path/to/frame2.jpg',
            score=0.9,
            bb_xywh=np.array([150, 150, 250, 250]),
            landmarks=np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
        )
        self.frame_detections.add(detection1)
        self.frame_detections.add(detection2)
        output_file = str(DATA_DIR / 'output.csv')
        self.frame_detections.save(output_file)
        loaded_detections = FrameDetections().load(output_file)
        self.assertEqual(len(loaded_detections), 2)
        self.assertEqual(loaded_detections[0], detection1)
        self.assertEqual(loaded_detections[1], detection2)
        os.remove(output_file)


if __name__ == '__main__':
    unittest.main()