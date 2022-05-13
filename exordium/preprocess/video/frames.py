import os
import cv2
from pathlib import Path
from typing import Tuple

import moviepy.editor as mpy
#import mmcv.io as io

def frames2video(input_dir: str, output_path: str, fps: int = 25):
    #CMD_STR = f'cd {input_dir} & ffmpeg -y -loglevel panic -r {fps} -pattern_type glob -i frame_*.png {output_path}'
    #print(CMD_STR)
    #os.system(CMD_STR)
    if Path(output_path).exists(): return
    images = sorted([str(Path(input_dir) / elem) for elem in os.listdir(input_dir) if elem[-4:] == '.png'])
    movie_clip = mpy.ImageSequenceClip(images, fps)
    movie_clip.write_videofile(str(output_path), logger=None)


def video2frames(input_path: str, output_dir: str, fps: int = 25, smallest_dim: int = 360, crop: Tuple[int, int, int, int] = None, verbose: bool = True) -> None:
    """Extracts the frames from a video

    Note:
        if crop is given, then first the video will be scaled, then cropped.

    Args:
        input_path (str): input video path
        output_dir (str): output dir path
        fps (int): frame per sec. Defaults to 30.
        smallest_dim (int): smallest dimension, height or width. Defaults to 256.
        crop (tuple): Crop bounding box defined by (x, y, h, w). Defaults to None.
            [0,0]----[0,w]
              |        |
              |        |
            [h,0]----[h,w]
    """
    output_dir = Path(output_dir).resolve() 
    if output_dir.exists(): return
    output_dir.mkdir(parents=True, exist_ok=True)
    vid = cv2.VideoCapture(str(input_path))
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    if height > width:
        w, h = -1, smallest_dim
    else:
        w, h = smallest_dim, -1

    if crop is None:
        CMD = f'ffmpeg -loglevel panic -i {str(input_path)} -r {int(fps)} -vf scale={h}:{w} {str(output_dir)}/frame_%05d.png -nostdin -vf -an -hide_banner'
    else:
        cy, cx, cw, ch = crop
        CMD = f'ffmpeg -loglevel panic -i {str(input_path)} -r {int(fps)} -vf "scale={h}:{w}:flags=neighbor,crop={ch}:{cw}:{cx}:{cy}" {str(output_dir)}/frame_%05d.png -nostdin -vf -an -hide_banner'
    if verbose: print(CMD)
    os.system(CMD)


if __name__ == '__main__':
    #video2frames(input_path='data/videos/9KAqOrdiZ4I.001.mp4', output_dir='data/processed/frames/9KAqOrdiZ4I.001', crop=(100,0,100,500)) 
    #video2frames(input_path='data/videos/multispeaker_720p.mp4', output_dir='data/processed/frames/multispeaker_720p', smallest_dim=720)
    #video2frames(input_path='data/videos/multispeaker_360p.mp4', output_dir='data/processed/frames/multispeaker_360p', smallest_dim=360)
    #video2frames(input_path='data/videos/multispeaker_360p.mp4', output_dir='data/processed/frames/multispeaker_180p', smallest_dim=180)

    #video2frames(input_path='data/videos/002003_FC2_A.mp4', output_dir='data/processed/frames/002003_FC2_A_180p', smallest_dim=180)
    #video2frames(input_path='data/videos/002003_FC1_A.mp4', output_dir='data/processed/frames/002003_FC1_A_360p', smallest_dim=360)
    #video2frames(input_path='data/videos/002003_FC1_A.mp4', output_dir='data/processed/frames/002003_FC1_A_180p', smallest_dim=180)
    
    pass