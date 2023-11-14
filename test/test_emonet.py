import unittest
from exordium import DATA_DIR, EXAMPLE_VIDEO_PATH
from exordium.video.facedetector import RetinaFaceDetector
from exordium.video.tracker import IouTracker
from exordium.video.emonet import EmoNetWrapper


class EmoNetTestCase(unittest.TestCase):

    def setUp(self):
        self.model = EmoNetWrapper(gpu_id=0)
        self.face_detector = RetinaFaceDetector(gpu_id=1)
        self.tracker = IouTracker()
        self.video_detections = self.face_detector.detect_video(EXAMPLE_VIDEO_PATH, output_path=DATA_DIR / 'example_multispeaker' / 'cache.vdet')
        self.face_track = self.tracker.label(self.video_detections).merge().get_center_track()

    def test_emotion_prediction_from_ndarray(self):
        faces = [det.bb_crop() for det in self.face_track][:16]
        valence, arousal, expression = self.model(faces, return_probabilities=False)
        self.assertEqual(valence.shape, (16,))
        self.assertEqual(arousal.shape, (16,))
        self.assertEqual(expression.shape, (16,))

    def test_emotion_probabilities_from_ndarray(self):
        faces = [det.bb_crop() for det in self.face_track][:16]
        valence, arousal, expression, expression_logits, expression_probabilities = self.model(faces, return_probabilities=True)
        self.assertEqual(valence.shape, (16,))
        self.assertEqual(arousal.shape, (16,))
        self.assertEqual(expression.shape, (16,))
        self.assertEqual(expression_logits.shape, (16, 8))
        self.assertEqual(expression_probabilities.shape, (16, 8))


if __name__ == '__main__':
    unittest.main()