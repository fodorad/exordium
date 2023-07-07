import unittest
from pathlib import Path
from exordium.video.detection import Tracker, Track, VideoDetections, FrameDetections, Detection, FaceDetector

FRAMES_DIR = f'data/processed/frames/9KAqOrdiZ4I.001'
CACHE_PATH = f'data/processed/cache/9KAqOrdiZ4I.001.vdet'
FACE_DIR = f'data/processed/faces/9KAqOrdiZ4I.001'
assert Path(FRAMES_DIR).is_dir()


class TrackerTestCase(unittest.TestCase):

    def setUp(self):
        self.face_detector = FaceDetector(verbose=False)
        self.vdet = self.face_detector.iterate_folder(FRAMES_DIR, output_path=CACHE_PATH)

    def test_tracker_creation(self):
        tracker = Tracker()
        self.assertEqual(tracker.new_track_id, 0)
        self.assertEqual(len(tracker.tracks), 0)
        self.assertEqual(len(tracker.selected_tracks), 0)

    def test_label_tracks_iou(self):
        tracker = Tracker()
        tracker.label_tracks_iou(self.vdet, max_lost=2, verbose=False)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(len(tracker.tracks[0]), 367)

    def test_interpolate(self):
        tracker = Tracker()
        vdet_part1 = VideoDetections()
        vdet_part1.detections = self.vdet.detections[:40]
        vdet_part2 = VideoDetections()
        vdet_part2.detections = self.vdet.detections[50:]
        vdet = VideoDetections()
        vdet.merge(vdet_part1)
        vdet.merge(vdet_part2)
        tracker.label_tracks_iou(vdet, max_lost=30, verbose=False)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(len(tracker.tracks[0]), 367)
        self.assertEqual(tracker.tracks[0].get_detection(45).score, -1)

    def test_merge_iou(self):
        tracker = Tracker()
        vdet_part1 = VideoDetections()
        vdet_part1.detections = self.vdet.detections[:40]
        vdet_part2 = VideoDetections()
        vdet_part2.detections = self.vdet.detections[50:]
        vdet = VideoDetections()
        vdet.merge(vdet_part1)
        vdet.merge(vdet_part2)
        tracker.label_tracks_iou(vdet, max_lost=5, verbose=False)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(len(tracker.tracks[0]), 40)
        self.assertEqual(len(tracker.tracks[1]), 317)
        tracker.merge_iou()
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(len(tracker.tracks[0]), 357)
    
    def test_select_track(self):
        tracker = Tracker()
        vdet_part1 = VideoDetections()
        vdet_part1.detections = self.vdet.detections[:40]
        vdet_part2 = VideoDetections()
        vdet_part2.detections = self.vdet.detections[50:60]
        vdet_part3 = VideoDetections()
        vdet_part3.detections = self.vdet.detections[70:]
        vdet = VideoDetections()
        vdet.merge(vdet_part1)
        vdet.merge(vdet_part2)
        vdet.merge(vdet_part3)
        tracker.label_tracks_iou(vdet, max_lost=5, verbose=False)
        self.assertEqual(len(tracker.tracks), 3)
        self.assertEqual(len(tracker.selected_tracks), 0)
        tracker.select_long_tracks(min_length=20)
        self.assertEqual(len(tracker.tracks), 3)
        self.assertEqual(len(tracker.selected_tracks), 2)
        tracker.select_topk_long_tracks(top_k=1)
        self.assertEqual(len(tracker.selected_tracks), 1)
        center_track = tracker.get_center_track()
        self.assertEqual(center_track.track_id, 2)
    
    def test_save_track_faces(self):
        tracker = Tracker()
        tracker.label_tracks_iou(self.vdet, max_lost=30, verbose=False)
        track = tracker.get_center_track()
        output_dir = Path(FACE_DIR)
        Tracker.save_track_faces(track, output_dir)

if __name__ == '__main__':
    unittest.main()
