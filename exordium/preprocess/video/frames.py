import os
import cv2
from pathlib import Path


def video2frames(input_path: str, output_dir: str, fps: int = 30, smallest_dim: int = 256) -> None:
    """Extracts the frames from a video

    Args:
        input_path (str): input video path
        output_dir (str): output dir path
        fps (int, optional): frame per sec. Defaults to 30.
        smallest_dim (int): smallest dimension, height or width. Defaults to 256.
    """
    output_dir = Path(output_dir).resolve()
    if output_dir.exists(): return
    output_dir.mkdir(parents=True, exist_ok=True)
    vid = cv2.VideoCapture(str(input_path))
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    if height > width:
        h, w = -1, smallest_dim
    else:
        h, w = smallest_dim, -1
    CMD = f'ffmpeg -loglevel panic -i {str(input_path)} -r {int(fps)} -vf scale={h}:{w} {str(output_dir)}/frame_%05d.png -nostdin -vf -an -hide_banner'
    os.system(CMD)

