import unittest
import cv2
from exordium.video.tddfa_v2 import get_3DDFA_V2_landmarks, get_faceboxes, draw_landmarks


class Test_3DDFA_V2(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_path = 'data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png'
        self.img = cv2.imread(self.img_path)
        self.bb = get_faceboxes(self.img)

    def test_3ddfa_v2(self):
        landmarks = get_3DDFA_V2_landmarks(self.img, self.bb)[0]
        draw_landmarks(self.img, landmarks)


if __name__ == '__main__':
    unittest.main()
