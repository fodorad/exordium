import logging
import os
import json
import cv2
from exordium import PathType


def frame2pose(input_path: PathType, output_path: PathType) -> None:
    """Sends frame to the service and saves the returned pose

    paper: https://arxiv.org/abs/1611.08050

    First, run docker image
    docker run -it -p 5000:5000 quay.io/codait/max-human-pose-estimator

    Args:
        input_path (PathType): path to the frame.
        output_path (PathType): path to the output file.
    """
    CMD = f'curl -F "file=@{str(input_path)}" -XPOST http://localhost:5000/model/predict > {str(output_path)}'
    logging.info(CMD)
    os.system(CMD)


def visualize_pose(frame_path: PathType, pose_path: PathType, output_path: PathType) -> None:
    """Visualize pose coords on the frame.

    Args:
        frame_path (PathType): path to the frame.
        pose_path (PathType): path to the pose json.
        output_path (PathType): path to the output file.
    """
    img = cv2.imread(str(frame_path))

    with open(pose_path) as f:
        pose = json.load(f)

    body_parts = pose['predictions'][0]['body_parts']
    colors = {'Nose':      [255,0,0],
              'Neck':      [0,255,0],
              'RShoulder': [0,0,255],
              'LShoulder': [0,255,255],
              'LEye':      [128, 128, 0],
              'REye':      [128, 0, 128],
              'LEar':      [128, 0, 128],
              'REar':      [0, 0, 0]}

    for element in body_parts:
        if element['part_name'] in list(colors.keys()):
            img[element['y']-2:element['y']+2, element['x']-2:element['x']+2] = colors[element['part_name']]

    cv2.imwrite(str(output_path), img)