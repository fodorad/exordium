import unittest
from pathlib import Path

import cv2
from facenet_pytorch import MTCNN

from exordium.preprocess.video.face import face_alignment, visualize_mtcnn


class TestFaceAlignment(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        img_path = Path('data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png')
        output_dir = Path('data/processed/faces_aligned/h-jMFLm6U_Y.000')
        self.output_path_mtcnn   = output_dir / img_path.name
        self.output_path_aligned = output_dir / f'{img_path.stem}_aligned.png'
        output_dir.mkdir(parents=True, exist_ok=True)

        self.img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        face_detector = MTCNN()
        bb_xyxy, probability, landmarks = face_detector.detect(self.img, landmarks=True)
        self.bb_xyxy = bb_xyxy[0]
        self.probability = probability[0]
        self.landmarks = landmarks[0]

    def test_face_alignment(self, bb_size: int = 224):
        visualize_mtcnn(img=self.img,
                        bb_xyxy=self.bb_xyxy,
                        probability=self.probability,
                        landmarks=self.landmarks,
                        output_path=self.output_path_mtcnn)
        face_aligned = face_alignment(img=self.img, landmarks=self.landmarks, output_width=bb_size)
        cv2.imwrite(str(self.output_path_aligned), cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR))
        self.assertEqual(face_aligned.shape[:2], (bb_size,bb_size))
        self.assertTrue(self.output_path_mtcnn.exists())
        self.assertTrue(self.output_path_aligned.exists())


if __name__ == '__main__':
    unittest.main()
