# exordium
Collection of preprocessing functions and deep learning methods.

# Setup
## Environment
* Python 3.8
* PyTorch 1.8

## Install dependencies
```bash
pip install -r requirements.txt --user
```

## Initialize submodules:
```bash
git submodule update --init --recursive
```

# Modalities
## Audio
Install pyAudioAnalysis package
```bash
mv ./tools/pyAudioAnalysis/requirements.txt ./tools/pyAudioAnalysis/_requirements.txt 
pip install -e ./tools/pyAudioAnalysis
```

## Pose estimation
Docker image for pose estimation
paper: https://arxiv.org/abs/1611.08050
```bash
docker run -it -p 5000:5000 quay.io/codait/max-human-pose-estimator
```
