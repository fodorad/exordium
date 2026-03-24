<div align="center">

<img src="https://raw.githubusercontent.com/fodorad/exordium/main/docs/assets/logo.svg" alt="Exordium" width="320"/>

<br/>

**Collection of Preprocessing Functions and Deep Learning Methods for Multimodal Feature Extraction**

[![CI](https://github.com/fodorad/exordium/workflows/CI/badge.svg)](https://github.com/fodorad/exordium/actions)
[![Coverage](https://codecov.io/gh/fodorad/exordium/branch/main/graph/badge.svg)](https://codecov.io/gh/fodorad/exordium)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=githubpages)](https://fodorad.github.io/exordium/)
[![GitHub Release](https://img.shields.io/github/v/release/fodorad/exordium?color=purple)](https://github.com/fodorad/exordium/releases)
[![PyPI](https://img.shields.io/pypi/v/exordium?color=purple)](https://pypi.org/project/exordium/)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

</div>

---

Exordium is a comprehensive toolkit for **multimodal feature extraction** across audio, video, and text modalities. It provides preprocessing functions, utility tools, and deep learning wrappers for processing and analyzing multimodal data.

## Features

### Audio

| Functionality | Model / Method | Output |
|---|---|---|
| I/O | load, save, resample | waveform |
| Spectral features | MFCC, Mel-spectrogram (with pre-emphasis) | spectrogram |
| Prosody | pitch, energy, voice activity, engagement | low-level descriptors |
| Low-level descriptors | [OpenSMILE](https://github.com/audeering/opensmile) — eGeMAPSv02 | 88-d vector |
| Audio–language embeddings | [CLAP](https://github.com/LAION-AI/CLAP) (laion/larger_clap_music_and_speech) | 512-d vector |
| Speech representations | [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) (facebook/wav2vec2-base-960h) | (T, 768) |
| Speech representations | [WavLM](https://huggingface.co/microsoft/wavlm-base-plus) (microsoft/wavlm-base/base+/large) | (T, 768/1024) per layer |

### Video

#### Face Detection & Tracking

| Functionality | Model / Method | Output |
|---|---|---|
| Face detection | [YOLOv8-Face](https://github.com/akanametov/yolo-face) (arnabdhar/YOLOv8-Face-Detection) | bounding boxes |
| Face detection + keypoints | [YOLO11-pose](https://github.com/zjykzj/YOLO11Face) (yolo11n/s-pose_widerface) | bounding boxes + 5-pt keypoints |
| Multi-face tracking | IoU-based tracker | track IDs across frames |

#### Face Analysis

| Functionality | Model / Method | Output |
|---|---|---|
| Dense facial landmarks | [MediaPipe FaceMesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (face_landmarker.task) | 478 × (x, y) |
| Iris landmarks | [MediaPipe Iris](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | 71 eye pts + 5 iris pts, EAR, diameters |
| Head pose | [6DRepNet](https://github.com/thohemp/6DRepNet) (300W-LP + AFLW2000) | yaw, pitch, roll (degrees) |
| Gaze estimation | [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) (ResNet-50, MPIIFaceGaze) | pitch, yaw (radians) |
| Gaze estimation | [UniGaze](https://github.com/darijakre/unigaze) (ViT-based) | pitch, yaw (radians) |
| Eye blink detection | [BlinkDenseNet121](https://github.com/fodorad/BlinkLinMulT) (DenseNet-121) | per-eye open/closed probability |
| Facial action units | [OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU) (Swin-T backbone) | 41-dim AU intensity vector |

#### Deep Visual Features

| Functionality | Model / Method | Output |
|---|---|---|
| Video features | [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (tiny/small/base) | 768-d / 768-d / 1024-d |
| Face identity features | [FAb-Net](https://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/index.html) | 256-d |
| Vision–language embeddings | [CLIP](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) (ViT-H/14, laion2B) | 1024-d |

### Text

| Functionality | Model / Method | Output |
|---|---|---|
| Speech-to-text | [Whisper](https://github.com/openai/whisper) (OpenAI) | transcript |
| Contextual embeddings | [BERT](https://huggingface.co/bert-base-uncased) (bert-base-uncased) | (T, 768) |
| Contextual embeddings | [RoBERTa](https://huggingface.co/roberta-large) (roberta-large) | (T, 1024) |
| Multilingual embeddings | [XML-RoBERTa](https://huggingface.co/xlm-roberta-base) (xlm-roberta-base) | (T, 768) |

### Utilities

- **Device management** — GPU/CPU selection via `get_torch_device`
- **Caching** — `@load_or_create` decorator (safetensors, npy, pkl, fdet, vdet, track)
- **Normalization** — global, per-feature, sliding-window
- **Padding** — fixed-length sequence padding and masking
- **Loss functions** — Bell, ecl1 losses
- **Concurrency** — thread- and process-pool helpers

---

## Installation

> **Requires [uv](https://docs.astral.sh/uv/).**
> The `video` extras include `unigaze`, which pins `timm==0.3.2` (broken with modern PyTorch).
> `uv`'s `override-dependencies` in `pyproject.toml` silently upgrades it to `timm>=1.0`.
> Plain `pip` has no equivalent override mechanism and will fail to resolve this conflict.

```bash
uv pip install exordium          # base only
uv pip install exordium[all]     # all optional dependencies
uv pip install exordium[audio]   # audio extras only
uv pip install exordium[video]   # video extras only
uv pip install exordium[text]    # text extras only
```

Install uv if you don't have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Extras

| Extra | Dependencies |
|---|---|
| `audio` | OpenSMILE, torchaudio — audio feature extraction |
| `text` | transformers, torchaudio — text and speech models |
| `video` | MediaPipe, Ultralytics, blinklinmult, unigaze, timm — face & video models |
| `all` | all previously described extras |

---

## Development

```bash
git clone https://github.com/fodorad/exordium
cd exordium
uv pip install -e ".[all,dev]"
make check   # lint + type-check + test + docs
```

---

## Documentation

- [API Reference](https://fodorad.github.io/exordium/)
- [Demo Notebooks](https://github.com/fodorad/exordium/tree/main/examples)

---

## Related Projects

### EmotionLinMulT (202X)

Efficient, transformer-based, multi-task emotion detection system.

- Paper: not published yet
- Code: [github.com/fodorad/EmotionLinMulT](https://github.com/fodorad/EmotionLinMulT)

### BlinkLinMulT (2023)

Transformer-based eye blink detection and eye state recognition across 7 public benchmark databases.

- Paper: [BlinkLinMulT: Transformer-based Eye Blink Detection](https://www.mdpi.com/2313-433X/9/10/196)
- Code: [github.com/fodorad/BlinkLinMulT](https://github.com/fodorad/BlinkLinMulT)

### PersonalityLinMulT (2022)

LinMulT trained for Big Five personality trait estimation and sentiment analysis.

- Paper: [Multimodal Sentiment and Personality Perception Under Speech](https://proceedings.mlr.press/v173/fodor22a.html)
- Code: [github.com/fodorad/PersonalityLinMulT](https://github.com/fodorad/PersonalityLinMulT)

### LinMulT

General-purpose multimodal transformer with linear-complexity attention mechanisms.

- Website: [adamfodor.com/LinMulT](https://adamfodor.com/LinMulT/)
- Code: [github.com/fodorad/LinMulT](https://github.com/fodorad/LinMulT)

---

## Contact

**Ádám Fodor** — [adamfodor.com](https://adamfodor.com) · fodorad201@gmail.com
