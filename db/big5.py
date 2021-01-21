import os, sys
import pickle
from pathlib import Path
from itertools import repeat

import parmap
from tqdm import tqdm
import multiprocessing as mp
from utils.preprocess import audio as a
from utils.preprocess import video as v
from utils.preprocess import image as i
from utils.preprocess import text as t
from utils.preprocess.shared import parallel_eval

fps = 30
sr = 44100
n_segments = 10
subsets = ['test', 'valid', 'train']
db_dir = Path.home() / 'db' / 'Big5'
pm_dict = {'pm_processes': mp.cpu_count(), 'pm_pbar': True}

for subset in subsets:
    print(f'Subset: {subset}')
    output_dir = Path.home() / 'db' / 'Big5_preproc' / subset
    output_dir.mkdir(parents=True, exist_ok=True)
    input_list = list(map(str, (db_dir / subset).glob('*.mp4')))
    ids = [Path(p).stem for p in input_list]
    frames_dirs = [str(output_dir / id / 'frames') for id in ids]
    audio_paths = [str(output_dir / id / (id + '.wav')) for id in ids]

    print('[Big5] video to frames')
    output_list = [str(output_dir / Path(vid).stem / 'frames') for vid in input_list]
    parmap.starmap(v.video2frames, list(zip(input_list, output_list)), fps=fps, **pm_dict)

    print('[Big5] video to audio')
    output_list = [output_dir / Path(vid).stem / (Path(vid).stem + '.wav') for vid in input_list]
    parmap.starmap(a.video2audio, list(zip(input_list, map(str, output_list))), sr=sr, **pm_dict)

    print('[Big5] audio to audio segments')
    output_list = [output_dir / Path(vid).stem / 'audio_segments' for vid in input_list]
    parmap.starmap(a.audio2segments, list(zip(audio_paths, map(str, output_list))), sr=sr, n_segments=n_segments, **pm_dict)

    print('[Big5] audio to mfcc and log melspec with delta features')
    output_list = [str(output_dir / Path(audio).stem / 'spectrograms') for audio in audio_paths]
    parmap.starmap(a.audio2logmelspec, list(zip(audio_paths, output_list)), sr=sr, **pm_dict)

    print('[Big5] audio to eGeMAPS features')
    config = Path(__file__).resolve().parents[1] / 'tools' / 'opensmile_wrapper' / 'opensmile_config' / 'gemaps' / 'eGeMAPSv01a.conf'
    assert config.exists(), f'[OpenSmile] Missing config file: {str(config)}'
    output_lld_list = [str(Path(output_dir).resolve() / Path(audio).stem / 'egemaps' / 'lld.csv') for audio in audio_paths]
    output_summary_list = [str(Path(output_dir).resolve() / Path(audio).stem / 'egemaps' / 'summary.csv') for audio in audio_paths]
    parmap.starmap(a.audio2egemaps, list(zip(audio_paths, output_lld_list, output_summary_list)), config=str(config), **pm_dict)
    os.system('rm -rfd smile.log')

    print('[Big5] audio to pyAudioAnalysis features')
    output_list = [str(output_dir / Path(audio).stem / 'pyaa' / Path(audio).stem) for audio in audio_paths]
    parmap.starmap(a.audio2pyaa, list(zip(audio_paths, output_list)), **pm_dict)

    print('[Big5] video to openface features')
    for input_path in tqdm(input_list):
        v.video2openface(input_path, str(output_dir / Path(input_path).stem / 'openface'), single_person=True)

    print('[Big5] frames to dynimg')
    output_list = [str(output_dir / Path(path).parent.name / 'dynimgs') for path in frames_dirs]
    parmap.starmap(v.frames2dynimgs, list(zip(frames_dirs, output_list)), **pm_dict)

