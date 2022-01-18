import os
from pathlib import Path


def text2speech_festival(input_path: str, output_dir: str) -> None:
    # build image: docker build -t docker_festival .
    # check it: docker run -it docker_festival
    parent_dir = Path(input_path).resolve().parent
    output_dir = Path(output_dir).resolve()
    CMD = f'docker run --entrypoint /festival/festival/bin/text2wave -it -v {str(parent_dir)}:/input_dir -v {str(output_dir)}:/output_dir ' \
          f'--rm docker_festival -o /output_dir/{Path(input_path).stem + "_TTS.WAV"} /{str(Path("/input_dir") / Path(input_path).name)}' # -eval "voice_cmu_us_rms_cg"'
    print(CMD)
    os.system(CMD)