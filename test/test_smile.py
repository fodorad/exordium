import unittest
from exordium import EXAMPLE_VIDEO_PATH
from exordium.audio.io import load_audio_from_video
from exordium.audio.smile import OpensmileWrapper


class SmileTestCase(unittest.TestCase):

    def setUp(self):
        self.smile = OpensmileWrapper()
        self.audio = load_audio_from_video(EXAMPLE_VIDEO_PATH)

    def test_extract_features(self):
        feature = self.smile(self.audio)
        self.assertEqual(feature.shape, (6116, 25))


if __name__ == '__main__':
    unittest.main()