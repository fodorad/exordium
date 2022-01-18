import os
from pathlib import Path


def video2openface(input_path: str, output_dir: str, single_person: bool = True):
    if Path(input_path).suffix.lower() in ['.jpeg', '.jpg', '.png']:
        binary = 'FaceLandmarkImg'
    elif not single_person:
        binary = 'FaceLandmarkVidMulti'
    else:
        binary = 'FeatureExtraction'
    parent_dir = Path(input_path).resolve().parent
    output_dir = Path(output_dir).resolve()
    #if output_dir.exists(): return
    CMD = f'docker run --entrypoint /home/openface-build/build/bin/{binary} -it -v {str(parent_dir)}:/input_dir -v {str(output_dir)}:/output_dir ' \
          f'--rm algebr/openface:latest -f /{str(Path("/input_dir") / Path(input_path).name)} -out_dir /output_dir'
    print(CMD)
    os.system(CMD)


