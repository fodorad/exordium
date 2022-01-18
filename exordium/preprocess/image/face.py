import os
import cv2
import numpy as np


def load_face_detector(name: str = 'FaceBoxes'):
    assert name in ['MTCNN', 'FaceBoxes'], 'Only MTCNN and FaceBoxes are supported right now.'
    if name == 'MTCNN':
        from mtcnn import MTCNN
        detector = MTCNN()
        print('MTCNN is loaded.')
    elif name == 'FaceBoxes':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        detector = FaceBoxes_ONNX() # xmin, ymin, w, h
        print('FaceBoxes is loaded.')
    else:
        raise NotImplementedError()
    return detector


def frame2face(input_path: str, detector_name: str, detector: object, extra_space: int = 0.0, resize_dim: int = 112) -> np.ndarray:
    """Extracts the face with highest confidence value from the given image

    Args:
        input_path (str): input image path
        extra_space (int, optional): extra space relative to the length of the bounding box. Defaults to 0.0.
        resize_dim (int, optional): resize output image. Defaults to 112.
    
    Returns:
        (np.ndarray): clipped face
    """
    assert detector_name in ['MTCNN', 'FaceBoxes'], 'Only MTCNN and FaceBoxes are supported right now.'
    if detector_name == 'MTCNN':
        raise NotImplementedError()
        # detector = load_face_detector(name=detector)
        img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        max_height, max_width, _ = img.shape
        if len(faces) > 0:
            ind = np.argmax(np.array([face['confidence'] for face in faces]), axis=0)
            bb = faces[ind]['box']
            side = max([bb[2], bb[3]])
            diff = abs(bb[2] - bb[3])
            extra = int(side * extra_space)
            w = bb[0] - extra // 2
            h = bb[1] - extra // 2
            if bb[2] < bb[3]:
                w -= diff // 2
            else:
                h -= diff // 2
            if w < 0: w = 0
            if h < 0: h = 0
            W = w + side + extra
            H = h + side + extra
            if W > max_width: W = max_width
            if H > max_height: H = max_height
            out = cv2.resize(img[h:H, w:W], (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    elif detector_name == 'FaceBoxes':
        img = cv2.imread(input_path) # BGR
        print('img shape:', img.shape, 'range:', img.min(), img.max())
        boxes = detector(img)
        print(boxes)
    else:
        raise NotImplementedError()

detector = load_face_detector()
frame2face('data/fi_processed/samples/_0bg1TLPP-I.003/frames/frame_00001.png',
           'FaceBoxes', detector)
