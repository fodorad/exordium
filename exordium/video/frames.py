import os
import sh
from pathlib import Path
import cv2
import numpy as np
import librosa
import ffmpeg
import soundfile as sf
import psutil


def video2frames(
        input_path: str | Path,
        output_dir: str | Path,
        fps: int | float = -1,
        start_number: int = 0,
        smallest_dim: int | None = None,  # 360
        crop: tuple[int, int, int, int] | None = None,
        overwrite: bool = False,
        verbose: bool = True) -> None:
    """Extracts the frames from a video

    Note:
        start_number is preferred to be 0 as more functionalities assumes it now.
        e.g.: 000000.png -> frame_id 0 -> 0. index of extracted features in (num_frames, feature_dim) tensor.

    Args:
        input_path (str): input video path
        output_dir (str): output dir path
        fps (int): frame per sec. -1 means that the fps of the given video is used. Defaults to -1.
        smallest_dim (int): smallest dimension, height or width. None means that the frames is not resized. Defaults to None.
        crop (tuple): Crop bounding box defined by (x, y, h, w). If crop is given, then first the video will be scaled, then cropped. Defaults to None.
            [0,0]----[0,w]
              |        |
              |        |
            [h,0]----[h,w]
    """
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not overwrite: return
    output_dir.mkdir(parents=True, exist_ok=True)

    vid = cv2.VideoCapture(str(input_path))
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    if smallest_dim is not None:
        w, h = (-1, smallest_dim) if height > width else (smallest_dim, -1)
    else:
        w, h = height, width

    if crop is None:
        crop_str = f'scale={h}:{w}'
    else:
        cy, cx, cw, ch = crop
        crop_str = f"scale={h}:{w}:flags=neighbor,crop={ch}:{cw}:{cx}:{cy}"

    if fps == -1:
        CMD = f'ffmpeg -loglevel panic -i {str(input_path)} -vf {crop_str} -start_number {start_number} {str(output_dir)}/%06d.png -nostdin -vf -an -hide_banner'
    else:
        CMD = f'ffmpeg -loglevel panic -i {str(input_path)} -r {fps} -vf {crop_str} -start_number {start_number} {str(output_dir)}/%06d.png -nostdin -vf -an -hide_banner'

    if verbose:
        print(CMD)

    os.system(CMD)


def frames2video(input_path: str | list[str] | list[np.ndarray],
                 output_path: str | Path,
                 fps: float | int = 25,
                 extension: str = '.png',
                 overwrite: bool = False,
                 verbose: bool = False):
    import moviepy.editor as mpy

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(input_path, (str, Path)):
        input_path = sorted([str(elem) for elem in list(Path(input_path).iterdir())
                             if elem.suffix == extension])

    # input_path = input_path[:1000]
    # output_path = 'test.mp4'

    print(f'Found {len(input_path)} frames.')
    movie_clip = mpy.ImageSequenceClip(input_path, fps)
    movie_clip.write_videofile(str(output_path),
                               logger="bar" if verbose else None)
    movie_clip.close()
    print(f'Video is done: {str(output_path)}')



def load_frames(video_path: str):
    print(f'Load video ({video_path})')
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = True
    i = 0
    while ret:
        ret, img = cap.read()  # (H, W, C)
        if ret:
            h = 240
            r = h / float(img.shape[0])  # H # 426x240
            dim = (int(img.shape[1] * r), h)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            i += 1
            if i % (fps * 60) == 0:
                print(f'{int(i // (fps*60))} mins are processed')
    video = np.stack(frames, axis=0)  # (T, H, W, C)
    print(
        f'Video shape: {video.shape}, fps: {fps}, length: {np.round(video.shape[0]/fps, decimals=2)} sec'
    )
    return video, fps


def load_audio(video_path: str, sr: int = 16000):
    print(f'Load audio ({video_path})')
    audio, sr = librosa.load(video_path, sr=sr)
    print(
        f'Audio shape: {audio.shape}, sr: {sr}, length: {np.round(audio.shape[0]/sr, decimals=2)} sec'
    )
    return audio, sr


def save_audio(audio: np.ndarray, output_path: str, sr: int = 16000):
    print(f'Save audio ({output_path})')
    sf.write(output_path, audio, sr, 'PCM_24')


class VideoWriter:

    def __init__(self,
                 fn,
                 vcodec='libx264',
                 fps=30,
                 in_pix_fmt='rgb24',
                 out_pix_fmt='yuv420p',
                 input_args=None,
                 output_args=None):
        self.fn = fn
        self.process = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args['framerate'] = fps
        self.input_args['pix_fmt'] = in_pix_fmt
        self.output_args['pix_fmt'] = out_pix_fmt
        self.output_args['vcodec'] = vcodec

    def add(self, frame):
        if self.process is None:
            h, w = frame.shape[:2]
            self.process = (ffmpeg.input(
                'pipe:',
                format='rawvideo',
                s='{}x{}'.format(w, h),
                **self.input_args).output(
                    self.fn, **self.output_args).overwrite_output().run_async(
                        pipe_stdin=True))
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()


def vidwrite(fn, images, **kwargs):
    writer = VideoWriter(fn, **kwargs)
    print(fn)
    for image in images:
        writer.add(image)
    writer.close()


def save_frames(video: np.ndarray, output_path: str):
    print(f'Save frames ({output_path})')
    images = [video[i, ...] for i in range(video.shape[0])]
    vidwrite(output_path, images)


def save_video(video: np.ndarray, audio: np.ndarray, output_path: str):
    assert video.ndim == 4 and video.shape[-1] == 3
    assert audio.ndim == 1
    id = Path(output_path).stem
    temp_video = str(Path(output_path).parent / f'_temp_{id}.mp4')
    temp_audio = str(Path(output_path).parent / f'_temp_{id}.wav')
    save_frames(video, temp_video)
    save_audio(audio, temp_audio)
    CMD_STR = f'ffmpeg -y -i {temp_video} -i {temp_audio} -c:v copy -c:a aac {output_path}'
    os.system(CMD_STR)
    sh.rm('-r', '-f', temp_video)
    sh.rm('-r', '-f', temp_audio)


if __name__ == '__main__':
    video2frames(input_path='data/videos/9KAqOrdiZ4I.001.mp4',
                 output_dir='data/processed/frames/9KAqOrdiZ4I.001',
                 crop=(100, 0, 100, 500))
    video2frames(input_path='data/videos/multispeaker_720p.mp4',
                 output_dir='data/processed/frames/multispeaker_720p',
                 smallest_dim=720)
    video2frames(input_path='data/videos/multispeaker_360p.mp4',
                 output_dir='data/processed/frames/multispeaker_360p',
                 smallest_dim=360)
