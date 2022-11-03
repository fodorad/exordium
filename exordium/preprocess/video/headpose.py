import yaml
from pathlib import Path

import cv2

from exordium.preprocess.video.retinaface import RetinaFace, detect_faces

try:
    from TDDFA import TDDFA
    from utils.pose import calc_pose
except:
    raise ImportError('3DDFA_V2 cannot be found. Build, then add "tools/3DDFA_V2" directory to PYTHONPATH.')

def_cfg = yaml.load(open(Path('tools/3DDFA_V2/configs/mb1_120x120.yml').resolve()), Loader=yaml.SafeLoader)
def_cfg['checkpoint_fp'] = f'tools/3DDFA_V2/{def_cfg["checkpoint_fp"]}'
def_cfg['bfm_fp'] = f'tools/3DDFA_V2/{def_cfg["bfm_fp"]}'


class HeadPoseEstimator(TDDFA):
    def __init__(self):
        super(HeadPoseEstimator, self).__init__(gpu_mode=True, **def_cfg)

    def estimate_headpose(self, img, boxes):
        param_lst, _ = self.__call__(img, boxes)

        param = param_lst[0]
        _, pose = calc_pose(param)
        yaw = pose[0]
        pitch = pose[1]
        roll = pose[2]

        return yaw, pitch, roll


def frames2headpose(frame_paths: str | list, model: HeadPoseEstimator = None):

    if model is None:
        model = HeadPoseEstimator()

    if isinstance(frame_paths, str):
        frame_paths = [frame_paths]

    detections = detect_faces(frame_paths=frame_paths, detector=RetinaFace(gpu_id=0, network='resnet50'))
    poses = [model.estimate_headpose(cv2.imread(frame_path), [detections[0]['bb']]) 
             for frame_path in frame_paths] # [(yaw, pitch, roll), ...]
    return poses
