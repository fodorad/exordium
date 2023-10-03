from exordium import DATA_DIR, EXAMPLE_VIDEO_PATH, PathType
from exordium.video.io import video2frames


def extract_frames(video_path: PathType):
    output_dir = DATA_DIR / 'processed' / video_path.stem / 'frames'
    video2frames(input_path=video_path, output_dir=output_dir, overwrite=True)


if __name__ == '__main__':
    extract_frames(EXAMPLE_VIDEO_PATH)