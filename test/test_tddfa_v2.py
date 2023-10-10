import unittest
from exordium import DATA_DIR, EXAMPLE_VIDEO_PATH
from exordium.video.facedetector import RetinaFaceDetector
from exordium.video.tracker import IouTracker
from exordium.video.tddfa_v2 import TDDFA_V2


class Tddfav2TestCase(unittest.TestCase):

    def setUp(self):
        self.model = TDDFA_V2()
        self.face_detector = RetinaFaceDetector(gpu_id=1)
        self.tracker = IouTracker()
        self.video_detections = self.face_detector.detect_video(EXAMPLE_VIDEO_PATH, output_path=DATA_DIR / 'example_multispeaker' / 'cache.vdet')
        self.face_track = self.tracker.label(self.video_detections).merge().get_center_track()

    def test_feature_extraction_from_ndarray(self):
        for det in self.face_track[:3]:
            face = det.bb_crop()
            output: dict = self.model(face)
            self.assertEqual(output['landmarks'].shape, (68,2))
            self.assertEqual(output['headpose'].shape, (3,))

    def test_feature_extraction_from_ndarray_with_eyes(self):
        for det in self.face_track[:3]:
            face = det.bb_crop()
            output: dict = self.model.face_to_eyes_crop(face, bb_size=20)
            self.assertEqual(output['landmarks'].shape, (68,2))
            self.assertEqual(output['headpose'].shape, (3,))
            self.assertEqual(output['left_eye'].shape, (20,20,3))
            self.assertEqual(output['right_eye'].shape, (20,20,3))


if __name__ == '__main__':
    unittest.main()