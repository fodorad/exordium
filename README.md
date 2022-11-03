# exordium
Collection of preprocessing functions and deep learning methods.

# Setup
## Environment
* Python 3.10
* PyTorch 1.11

## Install dependencies
```bash
pip install -r requirements.txt --user
```

## Initialize submodules
```bash
git submodule update --init --recursive
```

## Run tests
```bash
python -m unittest
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

## Head pose estimation
Build dependency: 3DDFA_V2
```bash
cd tools/3DDFA_V2
sh build.sh
cd ../..
export PYTHONPATH="${PYTHONPATH}:${PWD}/tools/3DDFA_V2"
```
