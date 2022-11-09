import os
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from TDDFA_ONNX import TDDFA_ONNX # tools/3DDFA_V2
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX # tools/3DDFA_V2
from exordium.utils.shared import get_project_root

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

'''
Left eye:
    37  38
36          39
    41  40

Right eye:
    43  44
42          45
    47  46
'''
FACE_LANDMARKS = {
    'face': list(range(0, 17)),
    'left_eyebrow': list(range(17, 22)),
    'right_eyebrow': list(range(22, 27)),
    'nose': list(range(27,36)),
    'left_eye': list(range(36,42)),
    'left_eye_left': 36,
    'left_eye_top': [37, 38],
    'left_eye_right': 39,
    'left_eye_bottom': [40, 41],
    'right_eye': list(range(42,48)),
    'right_eye_left': 42,
    'right_eye_top': [43, 44],
    'right_eye_right': 45,
    'right_eye_bottom': [46, 47],
    'mouth': list(range(47, 68))
}


def get_faceboxes(img, verbose: bool = True):
    face_boxes = FaceBoxes_ONNX()
    boxes = face_boxes(img)
    if verbose: print(f'Detect {len(boxes)} faces')
    return boxes # list of [x1, y1, x2, y2, score]


def get_3DDFA_V2_landmarks(img: np.ndarray, boxes: list, tddfa: TDDFA_ONNX = None):

    if tddfa is None:
        def_cfg = yaml.load(open(get_project_root() / Path(f'tools/3DDFA_V2/configs/mb1_120x120.yml')), Loader=yaml.SafeLoader)
        def_cfg['checkpoint_fp'] = str(get_project_root() / f'tools/3DDFA_V2/{def_cfg["checkpoint_fp"]}')
        def_cfg['bfm_fp'] = str(get_project_root() / f'tools/3DDFA_V2/{def_cfg["bfm_fp"]}')
        tddfa = TDDFA_ONNX(**def_cfg)

    param_lst, roi_box_lst = tddfa(img, boxes) # regress 3DMM params

    # reconstruct vertices - sparse 2d
    # list of (3, 68) tensor (x, y, z) with 68 landmarks per box
    return tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)


def draw_landmarks(img, pts, **kwargs):
    """Draw landmarks
    """
    height, width = img.shape[:2]
    plt.imshow(img[..., ::-1])

    if not type(pts) in [tuple, list]:
        pts = [pts]

    for i in range(len(pts)):
        alpha = 0.5
        markersize = 4
        lw = 1.0
        color = kwargs.get('color', 'w')
        markeredgecolor = kwargs.get('markeredgecolor', 'black')
        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
        colors = ['w', 'b', 'b', 'w', 'w', 'r', 'r', 'w', 'w', 'w']

        # close eyes and mouths
        plot_close = lambda i1, i2, c: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=c, lw=lw, alpha=alpha)
        #plot_close(41, 36, 'r')
        #plot_close(47, 42, 'r')
        #plot_close(59, 48, 'w')
        #plot_close(67, 60, 'w')
        plot_close(37, 38, 'g')
        plot_close(40, 41, 'b')
        plot_close(43, 44, 'g')
        plot_close(46, 47, 'b')

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            color = colors[ind]
            if color == 'r' or color == 'g': print(color, l, r)
            plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)
            plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                    color=color, markeredgecolor=markeredgecolor, alpha=alpha)
    plt.savefig('test.png')


if __name__ == '__main__':
    #draw_landmarks(img, ver_lst, dense_flag=dense_flag)
    pass
