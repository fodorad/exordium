import unittest
from exordium.video.headpose import frames2headpose


class TestHeadPose(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_path = 'data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png'

    def test_headpose(self):
        # 3DDFA_V2: (63.8680003082534, 46.346488810999375, -52.60335207496078)
        yaw, pitch, roll = frames2headpose(self.img_path)[0]
        self.assertGreater(yaw, 50)
        self.assertLess(yaw, 70)
        self.assertGreater(pitch, 40)
        self.assertLess(pitch, 60)
        self.assertGreater(roll, -60)
        self.assertLess(roll, -40)


if __name__ == '__main__':
    unittest.main()
