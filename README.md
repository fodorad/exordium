<div align="center">

<img src="https://via.placeholder.com/300x200?text=Exordium" alt="Exordium" width="260"/>

<br/>

**Collection of Preprocessing Functions and Deep Learning Methods for Multimodal Feature Extraction**

[![CI](https://github.com/fodorad/exordium/workflows/CI/badge.svg)](https://github.com/fodorad/exordium/actions)
[![Coverage](https://codecov.io/gh/fodorad/exordium/branch/main/graph/badge.svg)](https://codecov.io/gh/fodorad/exordium)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=githubpages)](https://fodorad.github.io/exordium/)
[![PyPI](https://img.shields.io/pypi/v/exordium?color=orange)](https://pypi.org/project/exordium/)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

</div>

---

Exordium is a comprehensive toolkit for **multimodal feature extraction** across audio, video, and text modalities. It provides preprocessing functions, utility tools, and deep learning methods for processing and analyzing multimodal data.

## Features

| | |
|---|---|
| **Audio Processing** | I/O, OpenSMILE, spectrograms, Wav2Vec2, CLAP, WavLM |
| **Video Analysis** | Face detection, landmarks, head pose, gaze, iris tracking, action units, feature extraction |
| **Text Processing** | BERT, RoBERTa, XML-RoBERTa |
| **Utilities** | Parallel processing, I/O helpers, loss functions, normalization, padding, visualization |

---

## Installation

```bash
pip install exordium          # base only
pip install exordium[all]     # all optional dependencies
pip install exordium[audio]   # audio extras only
pip install exordium[video]   # video extras only
pip install exordium[text]    # text extras only
```

### Extras

| extras tag | description |
|---|---|
| `audio` | dependencies to process audio data |
| `text` | dependency to process textual data |
| `face` | dependencies for face detection, landmarks, and head pose estimation |
| `video` | dependencies for various video feature extraction methods |
| `all` | all previously described extras will be installed |

---

## Quick start

### Audio feature extraction (WavLM)

```python
import numpy as np
from exordium.audio.wavlm import WavlmWrapper

model = WavlmWrapper(device_id=-1, model_name="base+")
waveform = np.random.rand(16000).astype(np.float32)
features = model.audio_to_feature(waveform)
# list of 12 numpy arrays, each (T, 768)
```

### Text feature extraction (BERT)

```python
from exordium.text.bert import BertWrapper

model = BertWrapper(device_id=-1)
features = model("Hello, world!", pool=True)
# torch.Tensor of shape (1, 768)
```

### Video face detection

```python
from exordium.video.face import RetinaFaceDetector
from exordium.video.io import images_to_np

detector = RetinaFaceDetector()
frames = images_to_np(["frame.jpg"], "RGB")
detections = detector.detect_image(frames[0])
```

---

## Development

For development with all dependencies:

```bash
git clone https://github.com/fodorad/exordium
cd exordium
pip install -e ".[all,dev]"
make check
```

---

## Documentation

> [API reference](https://fodorad.github.io/exordium/)

---

## Related Projects

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

- Paper: [LinMulT: Efficient Multimodal Transformers via Linear-Complexity Attention](https://adamfodor.com/LinMulT/)
- Code: [github.com/fodorad/LinMulT](https://github.com/fodorad/LinMulT)

---

## Contact

**Ádám Fodor** — [adamfodor.com](https://adamfodor.com) · fodorad201@gmail.com