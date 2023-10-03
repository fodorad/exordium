import unittest
from exordium import DATA_DIR, EXAMPLE_VIDEO_PATH
from exordium.video.facedetector import RetinaFaceDetector
from exordium.video.detection import VideoDetections
from exordium.video.tracker import IouTracker, DeepFaceTracker

CACHE_PATH = DATA_DIR / 'example_multispeaker' / 'cache'


class TrackerTestCase(unittest.TestCase):

    def setUp(self):
        self.face_detector = RetinaFaceDetector()
        self.vdet = self.face_detector.detect_video(EXAMPLE_VIDEO_PATH, output_path=CACHE_PATH)

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
        vdet_part1.detections = self.vdet.detections[:40]
        vdet_part2 = VideoDetections()
        vdet_part2.detections = self.vdet.detections[50:]
        vdet = VideoDetections()
        vdet.merge(vdet_part1)
        vdet.merge(vdet_part2)
        tracker.label(vdet, max_lost=30)
        self.assertEqual(len(tracker.tracks), 3)

    def test_merge_iou(self):
        tracker = IouTracker()
        vdet_part1 = VideoDetections()
        vdet_part1.detections = self.vdet.detections[:40]
        vdet_part2 = VideoDetections()
        vdet_part2.detections = self.vdet.detections[50:]
        vdet = VideoDetections()
        vdet.merge(vdet_part1)
        vdet.merge(vdet_part2)
        tracker.label(vdet, max_lost=5)
        self.assertEqual(len(tracker.tracks), 6)
        self.assertEqual(len(tracker.tracks[0]), 40)
        self.assertEqual(len(tracker.tracks[1]), 40)
        self.assertEqual(len(tracker.tracks[2]), 40)
        tracker.merge()
        self.assertEqual(len(tracker.tracks), 3)

    def test_merge_deepface(self):
        tracker = DeepFaceTracker()
        vdet_part1 = VideoDetections()
        vdet_part1.detections = self.vdet.detections[:40]
        vdet_part2 = VideoDetections()
        vdet_part2.detections = self.vdet.detections[50:]
        vdet = VideoDetections()
        vdet.merge(vdet_part1)
        vdet.merge(vdet_part2)
        tracker.label(vdet, max_lost=5)
        self.assertEqual(len(tracker.tracks), 6)
        self.assertEqual(len(tracker.tracks[0]), 40)
        self.assertEqual(len(tracker.tracks[1]), 40)
        self.assertEqual(len(tracker.tracks[2]), 40)
        tracker.merge()
        self.assertEqual(len(tracker.tracks), 3)

    def test_select_long_track(self):
        tracker = IouTracker()
        tracker.label(self.vdet)
        self.assertEqual(len(tracker.tracks), 3)
        self.assertEqual(len(tracker.selected_tracks), 3)
        del tracker.tracks[0].detections[19:]
        tracker.select_long_tracks(min_length=20)
        self.assertEqual(len(tracker.tracks), 3)
        self.assertEqual(len(tracker.selected_tracks), 2)

    def test_select_top_long_track(self):
        tracker = IouTracker()
        tracker.label(self.vdet)
        self.assertEqual(len(tracker.tracks), 3)
        self.assertEqual(len(tracker.selected_tracks), 3)
        del tracker.tracks[0].detections[19:]
        del tracker.tracks[1].detections[:100]
        tracker.select_topk_long_tracks(top_k=1)
        self.assertEqual(len(tracker.tracks), 3)
        self.assertEqual(len(tracker.selected_tracks), 1)
        self.assertEqual(list(tracker.selected_tracks.keys())[0], 2)

    def test_select_center_track(self):
        tracker = IouTracker()
        tracker.label(self.vdet)
        center_track = tracker.get_center_track()
        self.assertEqual(center_track.track_id, 1)


if __name__ == '__main__':
    unittest.main()