from pathlib import Path
import pickle
from math import cos, atan2, asin, sqrt
import cv2
import numpy as np
import torch
from torch import nn
import onnxruntime
from exordium import WEIGHT_DIR, PathType
from exordium.utils.ckpt import download_file
from exordium.video.io import image2np


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
        checkpoint_remote_path = 'https://github.com/fodorad/exordium/releases/download/v1.0.0/3ddfa_v2_mb1_120x120.pth'
        checkpoint_local_path = WEIGHT_DIR / 'tddfa_v2' / Path(checkpoint_remote_path).name
        download_file(checkpoint_remote_path, checkpoint_local_path)
        bfm_remote_path = 'https://github.com/fodorad/exordium/releases/download/v1.0.0/3ddfa_v2_bfm_noneck_v3.pkl'
        bfm_local_path = WEIGHT_DIR / 'tddfa_v2' / Path(bfm_remote_path).name
        download_file(bfm_remote_path, bfm_local_path)
        def_cfg = {
            'arch': 'mobilenet',
            'widen_factor': 1.0,
            'checkpoint_fp': str(checkpoint_local_path),
            'bfm_fp': str(bfm_local_path),
            'size': 120,
            'num_params': 62
        }
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
        camera_matrix, pose = _calc_pose(param_lst[0])

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


def _calc_pose(param):
    P = param[:12].reshape(3, -1) # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1) # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]
    return P, pose


def _parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2
    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength
    return roi_box


def _parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size
    return roi_box


def _crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


class DepthWiseBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes, bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU() if prelu else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)
        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)
        return out


class MobileNet(nn.Module):

    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """MobileNet V1

        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        self.relu = nn.PReLU() if prelu else nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet(**kwargs):
    """Construct MobileNet.

    Versions:
        widen_factor=1.0  for mobilenet_1
        widen_factor=0.75 for mobilenet_075
        widen_factor=0.5  for mobilenet_05
        widen_factor=0.25 for mobilenet_025
    """
    model = MobileNet(
        widen_factor=kwargs.get('widen_factor', 1.0),
        num_classes=kwargs.get('num_classes', 62)
    )
    return model


def _load_model(model, checkpoint_fp):
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model_dict = model.state_dict()
    for k in checkpoint.keys():
        kc = k.replace('module.', '')
        if kc in model_dict.keys():
            model_dict[kc] = checkpoint[k]
        if kc in ['fc_param.bias', 'fc_param.weight']:
            model_dict[kc.replace('_param', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    return model


def _convert_to_onnx(**kvs):
    # 1. load model
    size = kvs.get('size', 120)
    model = mobilenet(
        num_classes=kvs.get('num_params', 62),
        widen_factor=kvs.get('widen_factor', 1),
        size=size,
        mode=kvs.get('mode', 'small')
    )
    checkpoint_fp = kvs.get('checkpoint_fp')
    model = _load_model(model, checkpoint_fp)
    model.eval()

    # 2. convert
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, size, size)
    wfp = checkpoint_fp.replace('.pth', '.onnx')
    torch.onnx.export(
        model,
        (dummy_input,),
        wfp,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True
    )
    print(f'Convert {checkpoint_fp} to {wfp} done.')
    return wfp


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)


def _parse_param(param):
    """pre-defined templates for parameter

    Args:
        param (np.ndarray): matrix pose form of shape (trans_dim+shape_dim+exp_dim,)
           i.e., 62 = 12 + 40 + 10
    """
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp


class TDDFA_ONNX(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        # torch.set_grad_enabled(False)
        bfm_remote_path = 'https://github.com/fodorad/exordium/releases/download/v1.0.0/3ddfa_v2_bfm_noneck_v3.pkl'
        bfm_local_path = WEIGHT_DIR / 'tddfa_v2' / Path(bfm_remote_path).name
        download_file(bfm_remote_path, bfm_local_path)
        param_mean_std_remote_path = 'https://github.com/fodorad/exordium/releases/download/v1.0.0/3ddfa_v2_param_mean_std_62d_120x120.pkl'
        param_mean_std_local_path = WEIGHT_DIR / 'tddfa_v2' / Path(param_mean_std_remote_path).name
        download_file(param_mean_std_remote_path, param_mean_std_local_path)

        # load onnx version of BFM
        bfm_fp = kvs.get('bfm_fp', str(bfm_local_path))
        bfm_onnx_fp = bfm_fp.replace('.pkl', '.onnx')
        if not Path(bfm_onnx_fp).exists():
            self.convert_bfm_to_onnx(
                bfm_onnx_fp,
                shape_dim=kvs.get('shape_dim', 40),
                exp_dim=kvs.get('exp_dim', 10)
            )
        self.bfm_session = onnxruntime.InferenceSession(bfm_onnx_fp, providers=['CPUExecutionProvider'])

        # load for optimization
        bfm = BFMModel(bfm_fp, shape_dim=kvs.get('shape_dim', 40), exp_dim=kvs.get('exp_dim', 10))
        self.tri = bfm.tri
        self.u_base, self.w_shp_base, self.w_exp_base = bfm.u_base, bfm.w_shp_base, bfm.w_exp_base

        # config
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)
        param_mean_std_fp = kvs.get('param_mean_std_fp', str(param_mean_std_local_path))
        onnx_fp = kvs.get('onnx_fp', kvs.get('checkpoint_fp').replace('.pth', '.onnx'))

        # convert to onnx online if not existed
        if onnx_fp is None or not Path(onnx_fp).exists():
            print(f'{onnx_fp} does not exist, try to convert the `.pth` version to `.onnx` online')
            onnx_fp = _convert_to_onnx(**kvs)

        self.session = onnxruntime.InferenceSession(onnx_fp, providers=['CPUExecutionProvider'])

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

    def __call__(self, img_ori, objs, **kvs):
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box
                roi_box = _parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = _parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            img = _crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            img = (img - 127.5) / 128.

            inp_dct = {'input': img}

            param = self.session.run(None, inp_dct)[0]
            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            if dense_flag:
                inp_dct = {
                    'R': R, 'offset': offset, 'alpha_shp': alpha_shp, 'alpha_exp': alpha_exp
                }
                pts3d = self.bfm_session.run(None, inp_dct)[0]
                pts3d = similar_transform(pts3d, roi_box, size)
            else:
                pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                pts3d = similar_transform(pts3d, roi_box, size)

            ver_lst.append(pts3d)

        return ver_lst

    def convert_bfm_to_onnx(self, bfm_onnx_fp, shape_dim=40, exp_dim=10):
        bfm_fp = bfm_onnx_fp.replace('.onnx', '.pkl')
        bfm_decoder = BFMModel_ONNX(bfm_fp=bfm_fp, shape_dim=shape_dim, exp_dim=exp_dim)
        bfm_decoder.eval()
        dummy_input = torch.randn(3, 3), torch.randn(3, 1), torch.randn(shape_dim, 1), torch.randn(exp_dim, 1)
        R, offset, alpha_shp, alpha_exp = dummy_input
        torch.onnx.export(
            bfm_decoder,
            (R, offset, alpha_shp, alpha_exp),
            bfm_onnx_fp,
            input_names=['R', 'offset', 'alpha_shp', 'alpha_exp'],
            output_names=['output'],
            dynamic_axes={
                'alpha_shp': [0],
                'alpha_exp': [0],
            },
            do_constant_folding=True
        )
        print(f'Convert {bfm_fp} to {bfm_onnx_fp} done.')


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel(object):

    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        bfm = _load(bfm_fp)
        self.u = bfm.get('u').astype(float)
        self.w_shp = bfm.get('w_shp').astype(float)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(float)[..., :exp_dim]
        if Path(bfm_fp).name == 'bfm_noneck_v3.pkl':
            tri_remote_path = 'https://github.com/fodorad/exordium/releases/download/v1.0.0/3ddfa_v2_tri.pkl'
            tri_local_path = WEIGHT_DIR / 'tddfa_v2' / Path(tri_remote_path).name
            download_file(tri_remote_path, tri_local_path)
            self.tri = _load(str(tri_local_path)) # this tri/face is re-built for bfm_noneck_v3
        else:
            self.tri = bfm.get('tri')

        self.tri = _to_ctype(self.tri.T).astype(int)
        self.keypoints = bfm.get('keypoints').astype(int)
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]


class BFMModel_ONNX(torch.nn.Module):
    """BFM serves as a decoder"""

    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        super(BFMModel_ONNX, self).__init__()
        _to_tensor = lambda x: torch.from_numpy(x)

        # load bfm
        bfm = _load(bfm_fp)

        u = _to_tensor(bfm.get('u').astype(np.float32))
        self.u = u.view(-1, 3).transpose(1, 0)
        w_shp = _to_tensor(bfm.get('w_shp').astype(np.float32)[..., :shape_dim])
        w_exp = _to_tensor(bfm.get('w_exp').astype(np.float32)[..., :exp_dim])
        w = torch.cat((w_shp, w_exp), dim=1)
        self.w = w.view(-1, 3, w.shape[-1]).contiguous().permute(1, 0, 2)

    def forward(self, *inps):
        R, offset, alpha_shp, alpha_exp = inps
        alpha = torch.cat((alpha_shp, alpha_exp))
        pts3d = R @ (self.u + self.w.matmul(alpha).squeeze()) + offset
        return pts3d