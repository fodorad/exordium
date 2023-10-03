import unittest
import os
import numpy as np
from exordium import DATA_DIR
from exordium.video.detection import DetectionFromImage, Track


class TrackTestCase(unittest.TestCase):

    def setUp(self):
        self.detection1 = DetectionFromImage(frame_id=1, source='video/000001.png', score=0.9, bb_xywh=np.array([10, 20, 50, 60]), landmarks=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        self.detection2 = DetectionFromImage(frame_id=2, source='video/000002.png', score=0.8, bb_xywh=np.array([20, 30, 60, 70]), landmarks=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        self.detection3 = DetectionFromImage(frame_id=3, source='video/000003.png', score=0.7, bb_xywh=np.array([30, 40, 70, 80]), landmarks=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))

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
        output_file = str(DATA_DIR / 'output.csv')
        track.save(output_file)
        loaded_detections = Track(track_id=1).load(output_file)
        self.assertEqual(len(loaded_detections), 3)
        self.assertEqual(loaded_detections[0], self.detection1)
        self.assertEqual(loaded_detections[1], self.detection2)
        self.assertEqual(loaded_detections[2], self.detection3)
        os.remove(output_file)


if __name__ == '__main__':
    unittest.main()