import unittest
import numpy as np
from exordium.video.detection import Detection, Track


class TrackTestCase(unittest.TestCase):

    def setUp(self):
        self.detection1 = Detection(1, 'video/000001.png', 0.9, np.array([10, 20, 50, 60]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        self.detection2 = Detection(2, 'video/000002.png', 0.8, np.array([20, 30, 60, 70]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        self.detection3 = Detection(3, 'video/000003.png', 0.7, np.array([30, 40, 70, 80]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))

    def test_track_creation(self):
        # Create a track with a single detection
        track = Track(1, self.detection1)
        self.assertEqual(track.track_id, 1)
        self.assertEqual(len(track.detections), 1)
        self.assertEqual(track.detections[0], self.detection1)

    def test_frame_ids(self):
        # Create a track with multiple detections
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        frame_ids = track.frame_ids()
        self.assertListEqual(frame_ids, [1, 2, 3])

    def test_get_detection(self):
        # Create a track with multiple detections
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        detection = track.get_detection(2)
        self.assertEqual(detection, self.detection2)

    def test_add_detection(self):
        # Create two tracks with detections
        track1 = Track(1, self.detection1)
        track2 = Track(2, self.detection2)

        # Add a single detection to track1
        track1.add(self.detection3)
        self.assertEqual(len(track1.detections), 2)
        self.assertEqual(track1.detections[1], self.detection3)

        # Add track2 to track1 then sort by frame id
        track1.add(track2)
        self.assertEqual(len(track1.detections), 3)
        self.assertEqual(track1.detections[1], self.detection2)

    def test_track_frame_distance(self):
        # Create two tracks with detections
        track1 = Track(1, self.detection1)
        track2 = Track(2, self.detection2)
        # Calculate the frame distance between the two tracks
        distance = track1.track_frame_distance(track2)
        self.assertEqual(distance, 1)

        track3 = Track(3, self.detection3)
        # Calculate the frame distance between the two tracks
        distance = track1.track_frame_distance(track3)
        self.assertEqual(distance, 2)

    def test_last_detection(self):
        # Create a track with multiple detections
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        last_detection = track.last_detection()
        self.assertEqual(last_detection, self.detection3)

    def test_first_detection(self):
        # Create a track with multiple detections
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)
        first_detection = track.first_detection()
        self.assertEqual(first_detection, self.detection1)

    def test_center(self):
        # Create a track with multiple detections
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)

        # Calculate the center of the track
        center = track.center()
        expected_center = [50.0, 65.0]
        self.assertListEqual(center.tolist(), expected_center)

    def test_bb_size(self):
        # Create a track with multiple detections
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)

        # Calculate the bounding box size of the track
        bb_size = track.bb_size(extra_percent=0.2)
        self.assertEqual(bb_size, 96)

    def test_len(self):
        # Create a track with multiple detections
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)

        # Check the length of the track
        self.assertEqual(len(track), 3)

    def test_iterator(self):
        # Create a track with multiple detections
        track = Track(1, self.detection1)
        track.add(self.detection2)
        track.add(self.detection3)

        # Iterate over the track and collect the detections
        detections = [detection for i, detection in enumerate(track)]

        self.assertEqual(len(detections), 3)
        self.assertEqual(detections[0], self.detection1)
        self.assertEqual(detections[1], self.detection2)
        self.assertEqual(detections[2], self.detection3)


if __name__ == '__main__':
    unittest.main()
