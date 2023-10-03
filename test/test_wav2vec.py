import unittest
from exordium import EXAMPLE_VIDEO_PATH
from exordium.audio.io import load_audio_from_video
from exordium.audio.wav2vec import Wav2vec2Wrapper


class SmileTestCase(unittest.TestCase):

    def setUp(self):
        self.wav2vec2 = Wav2vec2Wrapper()
        self.audio = load_audio_from_video(EXAMPLE_VIDEO_PATH)

    def test_wav2vec2_extract_features(self):
        feature = self.wav2vec2(self.audio)
        self.assertEqual(feature.shape, (1, 3060, 768))


if __name__ == '__main__':
    unittest.main()