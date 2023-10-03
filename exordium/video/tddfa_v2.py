import sys
import yaml
from math import cos, atan2, asin
import numpy as np
from exordium import TOOL_ROOT, PathType
from exordium.video.io import image2np
TDDFA_ROOT = TOOL_ROOT / '3DDFA_V2'
if not TDDFA_ROOT in sys.path: sys.path.append(str(TDDFA_ROOT))
from TDDFA_ONNX import TDDFA_ONNX # tools/3DDFA_V2


class FaceLandmarks:
    """Defines the standard 68 face landmark indices.

    Left eye:
        37  38
    36          39
        41  40

    Right eye:
        43  44
    42          45
        47  46
    """
    FACE             = np.arange(0, 17)
    LEFT_EYEBROW     = np.arange(17, 22)
    RIGHT_EYEBROW    = np.arange(22, 27)
    NOSE             = np.arange(27, 36)
    LEFT_EYE         = np.arange(36, 42)
    LEFT_EYE_LEFT    = np.array([36])
    LEFT_EYE_TOP     = np.array([37, 38])
    LEFT_EYE_RIGHT   = np.array([39])
    LEFT_EYE_BOTTOM  = np.array([40, 41])
    RIGHT_EYE        = np.arange(42, 48)
    RIGHT_EYE_LEFT   = np.array([42])
    RIGHT_EYE_TOP    = np.array([43, 44])
    RIGHT_EYE_RIGHT  = np.array([45])
    RIGHT_EYE_BOTTOM = np.array([46, 47])
    MOUTH            = np.arange(47, 68)


def get_eye_with_center_crop(img: np.ndarray,
                             eye_center: np.ndarray,
                             bb_size: int = 64) -> np.ndarray:
    """Gets the eye crop from the face image with an eye centre point and a bounding box size.

    Args:
        img (np.ndarray): image of the face.
        eye_center (np.ndarray): xy center coordinate of the eye, e.g. mean of the eye landmarks.
        bb_size (int, optional): bounding box size. Defaults to 64.

    Returns:
        (np.ndarray): cropped eye.
    """
    y1, y2 = np.clip([int(eye_center[1] - bb_size//2), int(eye_center[1] + bb_size//2)], 0, img.shape[0])
    x1, x2 = np.clip([int(eye_center[0] - bb_size//2), int(eye_center[0] + bb_size//2)], 0, img.shape[1])
    return img[y1:y2, x1:x2, :]


class TDDFA_V2():
    """TDDFA_V2 wrapper class."""

    def __init__(self) -> None:
        """Headpose Extractor using 3DDFA_V2"""
        def_cfg = yaml.load(open(TDDFA_ROOT / 'configs' / 'mb1_120x120.yml'), Loader=yaml.SafeLoader)
        def_cfg['checkpoint_fp'] = str(TDDFA_ROOT / def_cfg["checkpoint_fp"])
        def_cfg['bfm_fp'] = str(TDDFA_ROOT / def_cfg["bfm_fp"])
        self.model = TDDFA_ONNX(**def_cfg)

    def __call__(self, image: PathType | np.ndarray,
                       boxes: list[tuple[int, int, int, int, float]] | None = None) -> dict[str, np.ndarray]:
        """Estimate landmarks and headpose from an image using face bounding boxes.

        Args:
            image (PathType | np.ndarray): image containing at least 1 face. It can be a path to an image,
                                         or np.ndarray with shape (H, W, 3) and BGR channel order.
            boxes (list | None, optional): bounding boxes. If no bounding box is provided (None),
                                           then the input image is a close face crop already. Defaults to None.

        Returns:
            dict[str, np.ndarray]: landmarks of shape (68, 2), headpose angles in degrees (yaw, pitch, roll) of shape (3,) and camera matrix
        """
        image = image2np(image, 'BGR')

        if boxes is None:
            boxes = [(0, 0, image.shape[0], image.shape[1], 1.)]

        param_lst, roi_box_lst = self.model(image, boxes) # regress 3DMM params
        camera_matrix, pose = calc_pose(param_lst[0])

        # reconstruct vertices - sparse 2d
        # list of (3, 68) tensor (x, y, z) with 68 landmarks per box
        fine_landmarks = self.model.recon_vers(param_lst, roi_box_lst, dense_flag=False) # (3, 68)
        fine_landmarks = fine_landmarks[0][:2,:].T # (3, 68) -> (68, 2)

        return {
            'landmarks': fine_landmarks, # (68, 2): xy
            'headpose': np.array([pose[0], pose[1], pose[2]]), # (3,): yaw, pitch, roll angles in degree
            'camera_matrix': camera_matrix
        }

    def face_to_eyes_crop(self, face: PathType | np.ndarray,
                                bb_size: int = 64) -> dict[str, np.ndarray]:
        """Cuts out the left and right eye regions using a fixed bounding box size.

        Args:
            face (PathType | np.ndarray): image of the face.
            bb_size (int, optional): bounding box size. Defaults to 64.

        Returns:
            dict[str, np.ndarray]: left and right eye patches among other face features from __call__
        """
        face = image2np(face, 'BGR')
        face_features = self(face)
        left_eye_center = np.mean(face_features['landmarks'][FaceLandmarks.LEFT_EYE,:], axis=0)
        right_eye_center = np.mean(face_features['landmarks'][FaceLandmarks.RIGHT_EYE,:], axis=0)
        left_eye = get_eye_with_center_crop(face, left_eye_center, bb_size)
        right_eye = get_eye_with_center_crop(face, right_eye_center, bb_size)
        return face_features | {'left_eye': left_eye, 'right_eye': right_eye}


    def face_to_xyxy_eyes_crop(self, face: PathType | np.ndarray,
                                     left_eye_landmarks_xyxy: np.ndarray,
                                     right_eye_landmarks_xyxy: np.ndarray,
                                     bb_size: int | None = None,
                                     extra_space: float = 1.) -> dict[str, np.ndarray]:
        """Cuts out the left and right eye regions using a fixed bounding box size.

        Args:
            face (PathType | np.ndarray): image of the face.
            left_eye_landmarks_xyxy (np.ndarray): left eye landmarks using xyxy format.
            right_eye_landmarks_xyxy (np.ndarray): right eye landmarks using xyxy format.
            bb_size (int | None, optional): bounding box size. None means that it is calculated from the eye landmarks. Defaults to None.
            extra_space (float, optional): extra space around the bounding box. 0.2 means an extra 20%. Defaults to 1.

        Returns:
            dict[str, np.ndarray]: left and right eye patches among other face features from __call__
        """
        face = image2np(face, 'BGR')
        face_features = self(face)
        left_eye_center = left_eye_landmarks_xyxy.reshape((2,2)).mean(axis=0)
        right_eye_center = right_eye_landmarks_xyxy.reshape((2,2)).mean(axis=0)

        if bb_size is None:
            bb_size = int(max([
                np.linalg.norm(left_eye_landmarks_xyxy[:2] - left_eye_landmarks_xyxy[2:]),
                np.linalg.norm(right_eye_landmarks_xyxy[:2] - right_eye_landmarks_xyxy[2:])
            ]) * (1 + extra_space))

        left_eye = get_eye_with_center_crop(face, left_eye_center, bb_size)
        right_eye = get_eye_with_center_crop(face, right_eye_center, bb_size)
        return face_features | {'left_eye': left_eye, 'right_eye': right_eye}


######################################################################################
#                                                                                    #
#   Code: https://github.com/cleardusk/3DDFA_V2                                      #
#   Authors: Jianzhu Guo, Xiangyu Zhu, Yang Yang, Fan Yang, Zhen Lei, Stan Z. Li     #
#   Reference: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py   #
#                                                                                    #
######################################################################################


def P2sRt(P):
    """Decompositing camera matrix P.

    Args:
        P: (3, 4). Affine Camera Matrix.

    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix.
    Reference: http://www.gregslabaugh.net/publications/euler.pdf
    Refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv

    Args:
        R: rotation matrix with shape (3,3)

    Returns:
        (float, float, float): yaw (x), pitch (y), roll (z)
    """
    if R[2, 0] > 0.998:
        z = 0.
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0.
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))
    return x, y, z


def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]
    return P, pose