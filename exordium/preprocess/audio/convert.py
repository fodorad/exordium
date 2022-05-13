import os
from pathlib import Path
import numpy as np


def video2audio(input_path: str, output_path: str, sr: int) -> None:
    """Extract audio from video and save

    Args:
        input_path (str): video path
        output_path (str): output path
        sr (int): sampling rate
    """
    output_path = Path(output_path).resolve()
    if output_path.exists(): return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    CMD = f'ffmpeg -loglevel 0 -i {str(input_path)} -nostdin -ab 320k -ac 1 -ar {sr} -c:a pcm_s16le -vn -hide_banner {str(output_path)}'
    os.system(CMD)


def audio2segments(input_path: str, output_dir: str, sr: int, n_segments: int) -> None:
    """Split audio file to N segments

    Args:
        input_path (str): input audio
        output_dir (str): output dir
        sr (int): sampling rate
        n_segments (int): number of segments
    """
    from pydub import AudioSegment
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(str(input_path), frame_rate=sr, format="wav")
    max_length = len(audio) # ms
    segment_length = max_length / n_segments
    steps = [int(np.floor(segment_length * i)) for i in range(n_segments)] + [max_length]
    for i in range(len(steps)-1):
        output_path = output_dir / (input_path.stem + '_{:02d}'.format(i) + '.wav')
        if output_path.exists(): return
        segment = audio[steps[i]:steps[i+1]]
        segment.export(str(output_path), format="wav")


