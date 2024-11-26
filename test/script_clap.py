from exordium.audio.clap import ClapWrapper
from exordium.audio.io import AudioLoader
import exordium.utils.decorator as D

D.TIMING_ENABLED = True
test_video = 'data/videos/example_multispeaker.mp4'
clap = ClapWrapper(use_cuda=False)
al = AudioLoader()
waveform, sr = al.load_audio(test_video, start_time_sec=0, end_time_sec=3, sr=44100, mono=True, squeeze=True, batch_dim=True)
feature = clap(waveform) # (B,C) == (1,1024)