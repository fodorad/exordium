from pathlib import Path
from exordium.video.frames import video2frames

RESOLUTION = '720p' # 360p
VIDEO_PATH = Path(f'data/videos/multispeaker_{RESOLUTION}.mp4')
assert VIDEO_PATH.exists(), f'Video is not available at {VIDEO_PATH}'

def extract_frames(video_path: str | Path):
    print('Extract frames...')
    output_dir = Path(f'data/processed/frames/{video_path.stem}')
    video2frames(input_path=VIDEO_PATH, output_dir=output_dir, verbose=True)

if __name__ == '__main__':
    extract_frames(VIDEO_PATH)