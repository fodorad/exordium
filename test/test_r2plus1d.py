import unittest
import numpy as np
from exordium.video.r2plus1d import R2plus1DWrapper


class R2plus1DTestCase(unittest.TestCase):

    def setUp(self):
        self.model = R2plus1DWrapper()

    def test_feature_extraction(self):
        video = np.zeros((32, 112, 112, 3))
        feature = self.model(video)
        self.assertEqual(feature.shape, (1, 512))


if __name__ == '__main__':
    unittest.main()