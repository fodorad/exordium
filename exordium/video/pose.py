import os
import json
import cv2


def frame2pose(input_path: str, output_path: str) -> None:
    """Sends frame to the service and saves the returned pose
    First, run docker image
    docker run -it -p 5000:5000 quay.io/codait/max-human-pose-estimator

    Args:
        data (Tuple[str, str]): frame path, pose path
    """
    CMD = f'curl -F "file=@{input_path}" -XPOST http://localhost:5000/model/predict > {output_path}'
    print(CMD)
    os.system(CMD)


def visualize_pose(frame_path: str, pose_path: str, output_path: str) -> None:
    """Visualize pose coords on the frame

    Args:
        frame_path (str): frame path
        pose_path (str): pose path
        output_path (str): output path
    """
    img = cv2.imread(frame_path)

    with open(pose_path) as f:
        pose = json.load(f)

    body_parts = pose['predictions'][0]['body_parts']
    colors = {'Nose': [255,0,0],
              'Neck': [0,255,0],
              'RShoulder': [0,0,255],
              'LShoulder': [0,255,255],
              'LEye': [128, 128, 0],
              'REye': [128, 0, 128],
              'LEar': [128, 0, 128],
              'REar': [0, 0, 0]}

    for element in body_parts:
        if element['part_name'] in list(colors.keys()):
            img[element['y']-2:element['y']+2, element['x']-2:element['x']+2] = colors[element['part_name']]

    cv2.imwrite(output_path, img)
