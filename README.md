# exordium
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://www.mypy-lang.org/)

Collection of preprocessing functions and deep learning methods.

# Supported features
## Audio
* frequently used io for audio files
* openSMILE feature extraction
* spectrogram calculation
* Wav2Vec2 feature extraction

## Video
* frequently used io for videos and frames
* bounding box manipulation methods
* face detection with RetinaFace
* face landmarks and head pose with 3DDFA_V2
* body pose estimation with max-human-pose-estimator
* categorical and dimensional emotion estimation with EmoNet
* iris and pupil landmark estimation with MediaPipe Iris
* fine eye landmark estimation with MediaPipe FaceMesh
* tracking using IoU and DeepFace
* FAb-Net feature extraction
* OpenFace feature extraction
* R2+1D feature extraction

## Text
* BERT feature extraction
* RoBERTa feature extraction

## Utils
* parallel processing
* io decorators
* loss functions
* normalization

## Visualization
* graphs
* 3D headpose
* 2D landmarks
* dataframes to images

# Setup
### Install package with pip+git
```
pip install -U git+https://github.com/fodorad/exordium.git
```

### Install package from repository root
```
git clone https://github.com/fodorad/exordium
cd exordium
pip install -e .
pip install -U -r requirements.txt
```

### Initialize submodules
```bash
git submodule update --init --recursive
```

### Run unittests
```bash
python -m unittest discover -s test
```

# Projects using exordium

### (2023) BlinkLinMulT
LinMulT is trained for blink presence detection and eye state recognition tasks.
Our results demonstrate comparable or superior performance compared to state-of-the-art models on 2 tasks, using 7 public benchmark databases.
* paper: BlinkLinMulT: Transformer-based Eye Blink Detection (accepted, available soon)
* code: https://github.com/fodorad/BlinkLinMulT

### (2022) PersonalityLinMulT
LinMulT is trained for Big Five personality trait estimation using the First Impressions V2 dataset and sentiment estimation using the MOSI and MOSEI datasets.
* paper: Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures ([pdf](https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf), [website](https://proceedings.mlr.press/v173/fodor22a.html))
* code: https://github.com/fodorad/PersonalityLinMulT

# What's next
* Add support for Action Unit detection (OpenGraphAU)
* Add support for Gaze estimation (L2CS-Net)
* Add support for Blink estimation (DenseNet121, LinT, BlinkLinMulT)
* Add support for Personality trait estimation (PersonalityLinMulT)

# Contact
* Ádám Fodor (foauaai@inf.elte.hu)