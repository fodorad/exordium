# exordium
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

[![CI](https://github.com/fodorad/exordium/workflows/CI/badge.svg)](https://github.com/fodorad/exordium/actions)
[![Coverage](https://codecov.io/gh/fodorad/exordium/branch/main/graph/badge.svg)](https://codecov.io/gh/fodorad/exordium)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Collection of preprocessing functions and deep learning methods.

# Supported features
## Audio
* frequently used io for audio files
* openSMILE feature extraction
* spectrogram calculation
* Wav2Vec2 feature extraction
* CLAP feature extraction
* WavLM feature extraction

## Video
* frequently used io for videos and frames
* bounding box manipulation methods
* face detection with RetinaFace
* face landmarks and head pose with 3DDFA_V2
* iris and pupil landmark estimation with MediaPipe Iris
* fine eye landmark estimation with MediaPipe FaceMesh
* eye gaze vector estimation with L2CS-Net
* tracking using IoU and DeepFace
* FAb-Net feature extraction
* OpenFace feature extraction
* R2+1D feature extraction
* Robust Video Matting background removal
* SWIN transformer feature extraction
* FaceMesh landmark estimation
* CLIP feature extraction
* OpenGraphAU action unit estimation

## Text
* BERT feature extraction
* RoBERTa feature extraction
* XML-RoBERTa feature extraction

## Utils
* parallel processing
* io decorators
* loss functions
* normalization
* padding/truncating
* graphs
* 3D headpose
* 2D landmarks
* gaze
* saliency maps
* dataframes to images

# Setup
### Install package with all base and optional dependencies from PyPI
```
pip install exordium[all]
```
### Install package with base dependencies from PyPI
```
pip install exordium
```
### Install optional dependencies for specific modules
The following extras will install the base and specific dependencies for using TDDFA_V2.
```
pip install exordium[tddfa]
```
You can install multiple optional dependencies as well.
```
pip install exordium[tddfa,audio]
```

#### Supported extras definitions:
| extras tag | description |
| --- | --- |
| audio | dependencies to process audio data |
| text | dependency to process textual data |
| tddfa | dependencies of TDDFA_V2 for landmark and headpose estimation, or related transformations |
| detection | dependencies for automatic face detection and tracking in videos |
| video | dependencies for various video feature extraction methods |
| all | all previously described extras will be installed |

Note: If you are not sure which tag should be used, just go with the all-mighty "all".

### Install package for development
```
git clone https://github.com/fodorad/exordium
cd exordium
pip install -e .[all]
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


# Contact
* Ádám Fodor (fodorad201@gmail.com)