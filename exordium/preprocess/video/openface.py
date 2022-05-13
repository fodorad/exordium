import os
from pathlib import Path


def extract_openface_singularity(input_path: str,
                                 output_dir: str,
                                 single_person: bool = True,
                                 singularity_container: str = 'tools/OpenFace/openface_latest.sif',
                                 singularity_args: str = ''):
    """OpenFace feature extractor

    Args:
        input_path (str): path to a video clip (.mp4)
        output_dir (str): path to an output directory.
                          a new directory is created with the name of the video
        single_person (bool, optional): Number of partipants in the video. Defaults to True.
    """
    # if this silently fails to generate the output, add write permission to the output directory
    if Path(input_path).suffix.lower() in ['.jpeg', '.jpg', '.png']:
        binary = 'FaceLandmarkImg'
    elif not single_person:
        binary = 'FaceLandmarkVidMulti'
    else:
        binary = 'FeatureExtraction'
    output_dir = Path(output_dir).resolve()
    if output_dir.exists(): return
    CMD = f'singularity exec {singularity_args} {singularity_container} tools/OpenFace/build/bin/{binary} -f {input_path} -nomask -out_dir {output_dir}'
    print(CMD)
    os.system(CMD)


def extract_openface_docker(input_path: str,
                            output_dir: str,
                            single_person: bool = True):
    """OpenFace feature extractor with Docker

    Args:
        input_path (str): path to a video clip (.mp4)
        output_dir (str): path to an output directory.
                          a new directory is created with the name of the video
        single_person (bool, optional): Number of partipants in the video. Defaults to True.
    """
    # if this silently fails to generate the output, add write permission to the output directory
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