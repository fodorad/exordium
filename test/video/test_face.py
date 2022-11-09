import unittest
from pathlib import Path
import cv2
from facenet_pytorch import MTCNN
from exordium.video.bb import xywh2xyxy
from exordium.video.retinaface import detect_faces
from exordium.video.face import face_alignment, visualize_detection, draw_landmarks


class TestFaceAlignment(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        img_path = Path('data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png')
        output_dir = Path('data/processed/faces_aligned/h-jMFLm6U_Y.000')
        self.output_path = output_dir / img_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    def test_mtcnn(self):
        face_detector = MTCNN()
        bb_xyxy, probability, landmarks = face_detector.detect(self.img, landmarks=True)
        self.bb_xyxy = bb_xyxy[0]
        self.probability = probability[0]
        self.landmarks = landmarks[0]
        self.visualize_face_alignment(postfix='_mtcnn')

    def test_retinaface(self):
        detection = detect_faces(self.img)[0]
        self.bb_xyxy = xywh2xyxy(detection['bb'])
        self.probability = detection['score']
        self.landmarks = detection['landmarks']
        self.visualize_face_alignment(postfix='_retinaface')

    def visualize_face_alignment(self, bb_size: int = 224, postfix: str = ''):
        output_detector = self.output_path.parent / self.output_path.name.replace('.png', f'{postfix}.png')
        output_original = self.output_path.parent / self.output_path.name.replace('.png', f'{postfix}_original.png')
        output_aligned = self.output_path.parent / self.output_path.name.replace('.png', f'{postfix}_aligned.png')

        visualize_detection(img=self.img, bb_xyxy=self.bb_xyxy, probability=self.probability, landmarks=self.landmarks, output_path=output_detector)
        face_aligned, landmarks_aligned = face_alignment(img=self.img, landmarks=self.landmarks, output_width=bb_size)

        draw_landmarks(self.img, self.landmarks, output_original, format='RGB')
        draw_landmarks(face_aligned, landmarks_aligned, output_aligned, format='RGB')

        self.assertEqual(face_aligned.shape[:2], (bb_size,bb_size))
        self.assertTrue(output_original.exists())
        self.assertTrue(output_aligned.exists())

    def test_retinaface_format(self):
        detection, landmarks = detect_faces(self.img, format='xyxys,lmks')[0]
        self.bb_xyxy = detection[:4]
        self.probability = detection[4]
        self.landmarks = landmarks
        self.visualize_face_alignment(postfix='_retinaface_xyxys')

if __name__ == '__main__':
    unittest.main()
