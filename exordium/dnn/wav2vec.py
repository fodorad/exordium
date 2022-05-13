import os
import warnings
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from fairseq import checkpoint_utils

from exordium.shared import get_weight_location, threads_eval


def get_weights():
    '''Downloads and wav2vec model weights if required
    '''
    weights_path = get_weight_location() / 'wav2vec' / 'wav2vec_large.pt'
    if not (weights_path).exists():
        pretrained_weights = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt'
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        os.system(f'wget {pretrained_weights} -P {weights_path.parent}')
    assert weights_path.exists()
    return weights_path


def build_wav2vec():
    """Builds pretrained Wav2Vec model

    Returns:
        nn.Module: pretrained wav2vec model instance
    """
    return checkpoint_utils.load_model_ensemble_and_task([str(get_weights())])[0][0]


def job_wav2vec(device: str, audio_paths: list, output_dir: str = 'tmp'):
    # FIXME
    warnings.warn("PyTorch do not move the data to GPU, " \
                  "Full samples take too much memory." \
                  "The calculations are enforced to CPU this time. " \
                  "The results will be the same, but the procedure might be much slower.")
    device = 'cpu'

    print(f'[Wav2Vec] Job started on {device}.')
    wav2vec = build_wav2vec()
    wav2vec.eval()
    #wav2vec.to(device)

    for audio_path in tqdm(audio_paths):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        outfile = str(Path(output_dir) / f'{Path(audio_path).stem}.npy')
        print(audio_path)
        if Path(outfile).exists(): continue
        waveform, sample_rate = torchaudio.load(audio_path)
        print(waveform.shape, sample_rate)
        # wav2vec is trained with 16k sr
        downsample_resample = torchaudio.transforms.Resample(sample_rate, 16000)
        down_sampled = downsample_resample(waveform)
        audio_features = torch.clamp(down_sampled, -1, 1)
        print(audio_features.shape)
        #audio_features.to(device)

        # Audio SSL feature extraction
        wav2vec_z =  wav2vec.feature_extractor(audio_features)
        print(wav2vec_z.shape)
        wav2vec_c =  wav2vec.feature_aggregator(wav2vec_z)
        print(wav2vec_c.shape)

        wav2vec_feature = wav2vec_c.transpose(1, 2).squeeze() # (B, C, T) -> (B, T, C) -> (T, C)
        if wav2vec_feature.shape[0] == 2: # stereo to mono
            wav2vec_feature = wav2vec_feature.mean(axis=0)
        print(wav2vec_feature.shape)

        wav2vec_feature = wav2vec_feature.detach().cpu().numpy().squeeze()
        print(wav2vec_feature.shape)
        np.save(outfile, wav2vec_feature)
        del wav2vec_feature, wav2vec_c, wav2vec_z
        print('done:', audio_path)
    print(f'[Wav2Vec] Job done on {device}.')


def get_wav2vec(audio_paths: list, device_ids: str = 'all', output_dir: str = 'tmp'):
    # FIXME
    warnings.warn("The calculations are enforced to CPU this time, using only 1 thread." \
                  "The results will be the same, but the procedure might be much slower.")
    threads_eval(job_wav2vec, audio_paths, device_ids='0', output_dir=output_dir)


if __name__ == '__main__':

    video_paths = [
        'data/videos/9KAqOrdiZ4I.001.mp4',
        'data/videos/h-jMFLm6U_Y.000.mp4',
        'data/videos/nEm44UpCKmA.002.mp4'
    ]

    audio_paths = [f'data/processed/audio/{Path(video_path).stem}.wav' for video_path in video_paths]

    from exordium.preprocess.audio.convert import video2audio
    for video_path, audio_path in zip(video_paths, audio_paths):
        video2audio(video_path, audio_path, 16000)

    get_wav2vec(audio_paths, output_dir='data/processed/wav2vec', device_ids='0')