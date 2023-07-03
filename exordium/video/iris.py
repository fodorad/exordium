from pathlib import Path
import cv2
import traceback
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from exordium.video.face import face_crop_with_landmarks
from exordium.video.irislandmarks import IrisPredictor
from exordium.video.tddfa_v2 import TDDFA_V2, FACE_LANDMARKS


'''
3DDFA_V2
Left eye:
    37  38
36          39
    41  40

Right eye:
    43  44
42          45
    47  46

FaceMesh:
     10 11 12 13 14
  9                 15
0                      8  
  1                 7
     2  3  4  5  6

Iris:
    2
3   0   1
    4   
'''
IRIS_LANDMARKS = {
    'center': 0,
    'left': 3,
    'top': 2,
    'right': 1,
    'bottom': 4
}

FACEMESH_EYE_LANDMARKS = {
    'eye': list(range(16)),
    'bottom_all': list(range(1, 8)),
    'top_all': list(range(9, 16)),
    'bottom': 4,
    'top': 12,
    'left': 0,
    'right': 8
}


def eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Calculate Eye Aspect Ratio feature.

    Expected 3DDFA_V2 eye landmark order:
       2   3
    1         4
       6   5
    
    Expected FaceMesh eye landmark order:
        10 11 12 13 14
      9                15
    0                     8  
      1                7
        2  3  4  5  6
    
    Usage:
        ear_left = eye_aspect_ratio(xy_left)
        ear_right = eye_aspect_ratio(xy_right)
        ear_mean = (ear_left + ear_right) / 2.0

    Args:
        landmarks (np.ndarray): eye landmarks with shape=(N, 2).
            N is 6 if the landmark detector is the 3DDFA_V2, and N is 16 in the case of FaceMesh.

    Returns:
        (float): eye aspect ratio.

    """
    assert landmarks.shape in {(6, 2), (71, 2), (16, 2)}, \
        'Invalid eye landmarks. Only 3DDFA_V2 or FaceMesh is supported.' \
        f'Expected (6, 2) or (71, 2) or (16, 2), but got instead {landmarks.shape}'

    if landmarks.shape == (6, 2): # 3DDFA_V2
        p2_minus_p6 = dist.euclidean(landmarks[1], landmarks[5])
        p3_minus_p5 = dist.euclidean(landmarks[2], landmarks[4])
        p1_minus_p4 = dist.euclidean(landmarks[0], landmarks[3])
        return (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    elif landmarks.shape in [(71, 2), (16, 2)]: # FaceMesh
        p11_minus_p3 = dist.euclidean(landmarks[11], landmarks[3])
        p12_minus_p4 = dist.euclidean(landmarks[12], landmarks[4])
        p13_minus_p5 = dist.euclidean(landmarks[13], landmarks[5])
        p0_minus_p8 = dist.euclidean(landmarks[0], landmarks[8])
        return (p11_minus_p3 + p12_minus_p4 + p13_minus_p5) / (3.0 * p0_minus_p8)
    else:
        raise NotImplementedError()


class IrisLandmarkExtractor():

    def __init__(self):
        self.eye_model = IrisPredictor()
        self.face_model = TDDFA_V2()


    def get_eye(self, img: np.ndarray, point: np.ndarray | list, bb_size: int = 64) -> np.ndarray | None:
        """Get the eye crop from the face image with an eye centre point and a bounding box size.

        Args:
            img (np.ndarray): image of the face.
            point (np.ndarray | list): centre point of the eye. Example: mean of the eye landmarks.
            bb_size (int, optional): bounding box size. Note that IrisPredictor expects a 64x64 eye patch. Defaults to 64.

        Returns:
            (np.ndarray | None): cropped eye or None, if the eye crop goes out of image.
        """
        y1 = max(int(point[1]-bb_size//2), 0)
        y2 = min(int(point[1]+bb_size//2), img.shape[0])
        x1 = max(int(point[0]-bb_size//2), 0)
        x2 = min(int(point[0]+bb_size//2), img.shape[1])
        image = img[y1:y2, x1:x2, :]

        if image.shape != (bb_size, bb_size, 3):
            if image.shape[0] > 0 and image.shape[1] > 0: # resize
                image = cv2.resize(image, (bb_size, bb_size), interpolation=cv2.INTER_AREA)
            else:
                return None # invalid crop

        return image


    def get_eyes_res(self, img: np.ndarray,
                     left_eye_xy: np.ndarray | list,
                     right_eye_xy: np.ndarray | list, 
                     extra_space: float = 0.2) -> np.ndarray | None:
        """Get the eye crop from the face image preserving the resolution

        Args:
            img (np.ndarray): image of the face.
            left_eye_xy (np.ndarray | list): xy corner coordinates of the left eye. [x1, y1, x2, y2]
            right_eye_xy (np.ndarray | list): xy corner coordinates of the right eye. [x1, y1, x2, y2]
            extra_space (float, optional): percentage of extra space added to the eye crop.
                Example: 0.2 means 20%. Defaults to 0.2. 

        Returns:
            (np.ndarray | None): cropped eye or None, if the eye crop goes out of image.
        """
        bb_size_eye = int(max([np.linalg.norm(np.array(left_eye_xy[:2])-np.array(left_eye_xy[2:])),
                               np.linalg.norm(np.array(right_eye_xy[:2])-np.array(right_eye_xy[2:]))]) * (1 + extra_space))
        # centre point
        left_cx = left_eye_xy[0] + abs(left_eye_xy[0]-left_eye_xy[2]) // 2
        # face is not aligned using the manual annotation
        left_cy = min(left_eye_xy[1], left_eye_xy[3]) + abs(left_eye_xy[1]-left_eye_xy[3]) // 2
        right_cx = right_eye_xy[0] + abs(right_eye_xy[0]-right_eye_xy[2]) // 2
        right_cy = min(right_eye_xy[1], right_eye_xy[3]) + abs(right_eye_xy[1]-right_eye_xy[3]) // 2
        # normalized bounding box
        left_nx1 = max([left_cx - bb_size_eye // 2, 0])
        left_nx2 = min([left_cx + bb_size_eye // 2, img.shape[1]])
        left_ny1 = max([left_cy - bb_size_eye // 2, 0])
        left_ny2 = min([left_cy + bb_size_eye // 2, img.shape[0]])
        right_nx1 = max([right_cx - bb_size_eye // 2, 0])
        right_nx2 = min([right_cx + bb_size_eye // 2, img.shape[1]])
        right_ny1 = max([right_cy - bb_size_eye // 2, 0])
        right_ny2 = min([right_cy + bb_size_eye // 2, img.shape[0]])
        return img[int(left_ny1):int(left_ny2), int(left_nx1):int(left_nx2), :], \
               img[int(right_ny1):int(right_ny2), int(right_nx1):int(right_nx2), :]


    def iris_diameter(self, iris_landmarks: np.ndarray) -> np.ndarray:
        """Calculate iris diameter from MediaPipe IrisPredictor landmarks.

        Args:
            iris_landmarks (np.ndarray): iris landmarks with shape=(5,2).

        Returns:
            tuple[float, float]: horizontal diameter, vertical diameter

        """
        w = np.linalg.norm(iris_landmarks[3,:]-iris_landmarks[1,:])
        h = np.linalg.norm(iris_landmarks[2,:]-iris_landmarks[4,:])
        return np.array([h, w], dtype=float)


    def iris_eyelid_distance(self, iris_landmarks: np.ndarray, eye_landmarks: np.ndarray) -> np.ndarray:
        """Calculate iris eyelid distance feature

        Args:
            iris_landmarks (np.ndarray): iris landmarks with shape=(5,2).
            eye_landmarks (np.ndarray): FaceMesh landmarks with shape=(71,2) or (16,2).

        Returns:
            np.ndarray: top distance, bottom distance

        """
        top = np.linalg.norm(iris_landmarks[2,:]-eye_landmarks[12,:])
        bottom = np.linalg.norm(iris_landmarks[4,:]-eye_landmarks[4,:])
        return np.array([top, bottom], dtype=float)


    def eye_to_features(self, eye: str | np.ndarray, output_path: str | Path | None = None) -> dict:
        """Generate features from an eye patch.

        Args:
            eye (str | np.ndarray): eye patch. It can be path to an image or np.ndarray with shape=(H,W,3).
            output_path (str | Path | None, optional): If the path is given, the features are saved as a pickle file. Defaults to None.

        Returns:
            dict: features as a dictionary
                Keys:
                    'eye': eye with shape=(64,64,3), BGR channel order.
                    'landmarks': FaceMesh landmarks with shape=(71,2).
                    'iris_landmarks': MediaPipe Iris prediction with shape=(5,2).
                    'iris_diameter': MediaPipe Iris prediction with shape=(2,).
                    'iris_eyelid_distance': MediaPipe Iris prediction and top/bottom eye landmarks' distances from iris centre with shape=(2,).
                    'ear': eye aspect ratio with shape=().

        """
        if isinstance(eye, str):
            eye = cv2.imread(eye)
        
        eye_original = eye.copy()

        assert eye.ndim == 3 and eye.shape[-1] == 3, f'Invalid eye patch. Expected shape is (H,W,3), got instead {eye.shape}'

        if eye.shape != (64, 64, 3):
            eye = cv2.resize(eye, (64, 64), interpolation=cv2.INTER_AREA)

        # output: (71, 2) eye landmarks xy, (5, 2) iris landmarks xy
        eye_lmrks, iris_lmrks = self.eye_model.predict(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB))
        iris_diameter = self.iris_diameter(iris_lmrks)
        iris_eyelid_distance = self.iris_eyelid_distance(iris_lmrks, eye_lmrks)
        ear = eye_aspect_ratio(eye_lmrks)

        return {
            'eye_original': eye_original,
            'eye': eye,
            'landmarks': eye_lmrks,
            'iris_landmarks': iris_lmrks,
            'iris_diameter': iris_diameter,
            'iris_eyelid_distance': iris_eyelid_distance,
            'ear': ear
        }


    def eyes_features(self, left_eye: str | Path | np.ndarray,
                      right_eye: str | Path | np.ndarray) -> dict:

        left_id = None
        right_id = None

        if isinstance(left_eye, (str, Path)):
            left_id = int(Path(left_eye).stem)
            left_eye = cv2.imread(str(left_eye))

        if isinstance(right_eye, (str, Path)):
            right_id = int(Path(right_eye).stem)
            right_eye = cv2.imread(str(right_eye))

        if left_id is not None and right_id is not None:
            assert left_id == right_id, f'The left and right eye ids differ: {l_id} vs {r_id}'
            frame_id = left_id
        else:
            frame_id = None

        left_eye_features = self.eye_to_features(left_eye)
        right_eye_features = self.eye_to_features(right_eye)
        return {'id': frame_id, 'left_eye': left_eye_features, 'right_eye': right_eye_features}

    def eye_features(self, eye_path: str | Path) -> dict:
        eye_features = self.eye_to_features(cv2.imread(str(eye_path)))
        return {'eye': eye_features}


    def face_to_eye_crop(self, face: str | Path | np.ndarray, bb_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Cut out the left and right eye regions using a fixed bounding box size.

        Args:
            face (str | Path | np.ndarray): image of the face.
            bb_size (int, optional): bounding box size. MediaPipe Iris expects 64x64 eye patches. Defaults to 64. 

        Returns:
            np.ndarray: left eye patch, right eye patch

        """
        if isinstance(face, str | Path):
            face = cv2.imread(face)

        face_features = self.face_model.inference(face)

        xy_l = face_features['landmarks'][np.array(FACE_LANDMARKS['left_eye']),:]
        xy_r = face_features['landmarks'][np.array(FACE_LANDMARKS['right_eye']),:]

        left_eye = self.get_eye(face, np.mean(xy_l, axis=0), bb_size=bb_size)
        right_eye = self.get_eye(face, np.mean(xy_r, axis=0), bb_size=bb_size)

        return left_eye, right_eye


    def face_to_eyes_res_crop(self, face: str | Path | np.ndarray, extra_space: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Crop both left and right eye regions from a face image. Resolution is preserved.

        Args:
            face (str | Path | np.ndarray): image containing only 1 face.
            extra_space (float, optional): extra space added to the eye crop. 
            0.2 means 20% more than the distance calculated from the eye corners. Defaults to 1.0.

        Returns:
            np.ndarray: left eye image patch, right eye image patch with shape=(H,W,3)

        """
        if isinstance(face, str | Path):
            face = cv2.imread(face)

        face_features = self.face_model.inference(face)

        left_eye_xy = list(face_features['landmarks'][np.array(FACE_LANDMARKS['left_eye_left']),:]) + \
                      list(face_features['landmarks'][np.array(FACE_LANDMARKS['left_eye_right']),:])
        right_eye_xy = list(face_features['landmarks'][np.array(FACE_LANDMARKS['right_eye_left']),:]) + \
                       list(face_features['landmarks'][np.array(FACE_LANDMARKS['right_eye_right']),:])

        left_eye, right_eye = self.get_eyes_res(face, left_eye_xy, right_eye_xy, extra_space=extra_space)

        return left_eye, right_eye


    def face_to_eye_features(self, face_path: str | Path) -> dict:
        id = int(Path(face_path).stem)
        left_eye, right_eye = self.face_to_eye_crop(face_path)
        left_eye_features = self.eye_to_features(left_eye)
        right_eye_features = self.eye_to_features(right_eye)

        return {'id': id, 'left_eye': left_eye_features, 'right_eye': right_eye_features}
        
    '''
    def eye_features(self, img_paths: str | list, verbose: bool = False, output_path: str | None = None):
        img_lmrks = face_crop_with_landmarks(img_paths, output_path=output_path) # list of dicts

        eye_rgb = []
        eye_lmrks = []
        iris_lmrks = []
        for i, elem in tqdm(enumerate(img_lmrks), total=len(img_lmrks), desc='Iris landmarks and EAR extraction'):

            if elem['img'] is None or elem['landmarks'] is None:
                print('Image without face detection:', elem['frame_path'])
                eye_rgb.append((None, None))
                eye_lmrks.append((None, None))
                iris_lmrks.append((None, None))
            else:
                img = elem['img']
                xy_l = elem['landmarks'][np.array(FACE_LANDMARKS['left_eye']),:]
                xy_r = elem['landmarks'][np.array(FACE_LANDMARKS['right_eye']),:]

                if verbose:
                    Path('tmp').mkdir(parents=True, exist_ok=True)
                    self.eye_3ddfa_visualization(img, xy_l, f'tmp/l_{i:05d}.png')
                    self.eye_3ddfa_visualization(img, xy_r, f'tmp/r_{i:05d}.png')
                    self.eye_3ddfa_visualization(img, xy_r, f'tmp/r_{i:05d}.png')

                eye_l = self.get_eye(img, np.mean(xy_l, axis=0)) # (2,) -> (64, 64, 3)
                eye_r = self.get_eye(img, np.mean(xy_r, axis=0)) # (2,) -> (64, 64, 3)

                # input: 64x64 np.ndarray
                # output: (71, 2) eye landmarks xy, (5, 2) iris landmarks xy
                if eye_l is not None:
                    eye_lmrks_left, iris_lmrks_left = self.eye_model.predict(cv2.cvtColor(eye_l, cv2.COLOR_BGR2RGB))
                else:
                    print('Left eye is not detected:', elem['frame_path'])
                    eye_lmrks_left, iris_lmrks_left = None, None
                
                if eye_r is not None:
                    eye_lmrks_right, iris_lmrks_right = self.eye_model.predict(cv2.cvtColor(eye_r, cv2.COLOR_BGR2RGB))
                else:
                    print('Right eye is not detected:', elem['frame_path'])
                    eye_lmrks_right, iris_lmrks_right = None, None

                eye_rgb.append((eye_l, eye_r))
                eye_lmrks.append((eye_lmrks_left, eye_lmrks_right))
                iris_lmrks.append((iris_lmrks_left, iris_lmrks_right))

        return img_lmrks, eye_rgb, eye_lmrks, iris_lmrks
    '''

    @classmethod
    def eye_3ddfa_visualization(cls, img: np.ndarray | str, eye_lmrks: np.ndarray, output_path: str | Path, size: float = 10.0):
        assert eye_lmrks.ndim == 2 and eye_lmrks.shape[1] == 2, \
            f'Invalid eye landmarks. Expected shape: (N, 2), expected mediapipe shape: (71, 2), got instead: {eye_lmrks.shape}.'

        if isinstance(img, str):
            img = cv2.imread(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(img)
        plt.scatter(eye_lmrks[:, 0], eye_lmrks[:, 1], s=size, color='b')
        plt.savefig(str(output_path))
        plt.close()


    @classmethod
    def iris_visualization(cls, img: np.ndarray | str, eye_lmrks: np.ndarray, iris_lmrks: np.ndarray, output_path: str | Path, size: float = 10.0):
        assert eye_lmrks.ndim == 2 and eye_lmrks.shape[1] == 2, \
            f'Invalid eye landmarks. Expected shape: (N, 2), expected mediapipe shape: (71, 2), got instead: {eye_lmrks.shape}.'
        assert iris_lmrks.ndim == 2 and iris_lmrks.shape[1] == 2, \
            f'Invalid eye landmarks. Expected shape: (N, 2), expected mediapipe shape: (5, 2) got instead: {iris_lmrks.shape}.'

        if isinstance(img, str):
            img = cv2.imread(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(img)
        plt.scatter(eye_lmrks[:, 0], eye_lmrks[:, 1], s=size, color='b')
        plt.scatter(iris_lmrks[1:, 0], iris_lmrks[1:, 1], s=size, color='r')
        plt.scatter(iris_lmrks[0, 0], iris_lmrks[0, 1], s=size, color='g')
        plt.savefig(str(output_path))
        plt.close()


    @classmethod
    def facemesh_visualization(cls, img: np.ndarray | str, eye_lmrks: np.ndarray, output_path: str | Path):
        assert eye_lmrks.ndim == 2 and eye_lmrks.shape[1] == 2, \
            f'Invalid eye landmarks. Expected shape: (N, 2), expected mediapipe shape: (71, 2), got instead: {eye_lmrks.shape}.'

        if isinstance(img, str):
            img = cv2.imread(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(img)
        plt.scatter(eye_lmrks[:, 0], eye_lmrks[:, 1], s=5.0, color='b')
        for i in range(eye_lmrks.shape[0]):
            plt.text(eye_lmrks[i, 0], eye_lmrks[i, 1], str(i), fontsize=6)
        plt.savefig(str(output_path))
        plt.close()


if __name__ == "__main__":
    ie = IrisLandmarkExtractor()
    bd = BlinkDetector()

    img_paths = ['data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png',
                 '/home/fodor/dev/eye/data/talkingFace/frames/000001.png',
                 '/home/fodor/dev/eye/data/talkingFace/frames/000228.png']

    '''
    for img_path in img_paths:
        img_lmrks, eye_rgb, eye_lmrks, iris_lmrks = ie.eye_features(img_path)
        IrisLandmarkExtractor.iris_visualization(eye_rgb[0][0], eye_lmrks[0][0], iris_lmrks[0][0], f'eye_left_{Path(img_path).stem}.png')
        IrisLandmarkExtractor.iris_visualization(eye_rgb[0][1], eye_lmrks[0][1], iris_lmrks[0][1], f'eye_right_{Path(img_path).stem}.png')
        IrisLandmarkExtractor.facemesh_visualization(eye_rgb[0][0], eye_lmrks[0][0], f'eye_l_lmrks_{Path(img_path).stem}.png')
        IrisLandmarkExtractor.facemesh_visualization(eye_rgb[0][1], eye_lmrks[0][1], f'eye_r_lmrks_{Path(img_path).stem}.png')
    '''
    for img_path in img_paths:
        img_lmrks, eye_rgb, headpose = bd.rgb(img_path)
        print(headpose)
        cv2.imwrite(str(Path(img_path).stem + '_left.png'), eye_rgb[0][0])
        cv2.imwrite(str(Path(img_path).stem + '_right.png'), eye_rgb[0][1])