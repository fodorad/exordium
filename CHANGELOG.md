# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.0](https://github.com/fodorad/exordium/compare/v2.1.0...v2.2.0) (2026-03-30)


### Features

* support pretrained EmotiEffNet ([3180453](https://github.com/fodorad/exordium/commit/3180453752428f3e43bf69c45502fe6e980dbbb5))
* support pretrained emotion2vec+ seed ([6df3848](https://github.com/fodorad/exordium/commit/6df38481bce32813ccbf087b220dda0f8a6f395d))
* support pretrained wav2vec2 on emotion classification (iemocap, 4-class) ([e625e8b](https://github.com/fodorad/exordium/commit/e625e8b31b41b0576992f0781d0cd21851fba371))

## [2.1.0](https://github.com/fodorad/exordium/compare/v2.0.0...v2.1.0) (2026-03-26)


### Features

* add DINOv2 as deep feature extractor, update demo and docs ([c7ccc27](https://github.com/fodorad/exordium/commit/c7ccc27550da2af4b238675d996fdabb1aafff83))

## [2.1.0] — 2026-03-26

### Added
- DINOv2 wrapper (`DINOv2Wrapper`) — HuggingFace Transformers-based ViT encoder with four variants: small (384-d), base (768-d), large (1024-d), giant (1536-d); L2-normalised CLS-token embeddings

---

## [2.0.0] — 2026-03-23

### Added
- YOLO11-pose face detector (`YoloFace11Detector`) — bounding boxes + 5-point facial keypoints from [zjykzj/YOLO11Face](https://github.com/zjykzj/YOLO11Face), weights auto-downloaded
- YOLOv8 face detector (`YoloFaceV8Detector`) — bounding-box-only detection via [arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
- `FaceDetector` abstract base class unifying all detector subclasses with shared `detect_image`, `detect_frame_dir`, `detect_video` pipeline
- `crop_eye_regions` supporting both YOLO11 5-pt coarse keypoints and FaceMesh 478-pt dense landmarks
- 6DRepNet head pose wrapper (`SixDRepNetWrapper`) — yaw, pitch, roll in degrees (300W-LP + AFLW2000)
- UniGaze wrapper (`UnigazeWrapper`) — ViT-based gaze estimation with roll correction support
- `GazeWrapper` abstract base class for interchangeable gaze models
- `build_facemesh_region_colors` and `FACEMESH_REGION_COLORS` for regional landmark visualization
- `FaceMesh478Regions` landmark index groups (eyes, eyebrows, nose, mouth, face oval)
- Swin Transformer wrapper (`SwinTWrapper`) — tiny/small/base variants, 768-d / 1024-d features
- CLIP wrapper (`ClipWrapper`) — ViT-H/14 laion2B, 1024-d L2-normalised embeddings
- `load_or_create` decorator extended with safetensors (`st`) format alongside npy, pkl, fdet, vdet, track
- `batch_iterator` and `to_uint8_tensor` utilities in `video.core.io`
- `Video` context manager and `iter_batches` for memory-efficient video frame streaming
- GitHub Actions CI/CD pipelines (ci.yml, cd.yml, docs.yml) with hatch-vcs auto-versioning
- Sphinx + sphinx-autoapi + Furo documentation deployed to GitHub Pages
- PEP 257 attribute docstrings across all modules — Sphinx autoapi renders full descriptions for every module-level constant, class attribute, and dataclass field
- SVG logo (`docs/assets/logo.svg`) and absolute raw GitHub URL in README for correct PyPI rendering
- Codecov integration in CI pipeline with explicit `coverage.xml` upload and single-matrix upload guard
- Example notebooks for every module with consistent `| Stage | Model | Output |` pipeline tables
- Python 3.13 support; `py.typed` PEP 561 marker

### Changed
- `FaceMeshWrapper` upgraded to MediaPipe Tasks API (`FaceLandmarker`) — replaces deprecated `mp.solutions.face_mesh`; model auto-downloaded to `~/.cache/mediapipe_models/`
- Face detection API unified: all detectors return `FrameDetections` / `VideoDetections` with `Detection` dataclass objects
- `Detection` dataclass made frozen and `kw_only`; subclasses `DetectionFromImage`, `DetectionFromVideo`, `DetectionFromNp`, `DetectionFromTorchTensor` route based on source type via `DetectionFactory`
- `visualize_landmarks` made type-preserving: tensor input returns tensor, numpy returns numpy
- Bounding box format standardised to `bb_xywh` (long tensor) throughout; `bb_xyxy` is a computed property
- `Track` and `VideoDetections` serialisation uses safetensors by default
- `VisualModelWrapper` and `AudioModelWrapper` base classes aligned: `__call__` returns tensors, `track_to_feature` / `audio_to_feature` returns numpy via `@load_or_create`
- Package build system migrated from setuptools to Hatch + hatch-vcs (PEP 517/518)
- Sphinx docs front page rewritten: feature bullet lists, example notebooks table, Related Projects

### Fixed
- All 157 ruff errors (D100, E501, F401, E741, D417, TC006) resolved across the codebase
- `ty` type-check errors in `iris.py`, `facemesh.py`, `detection.py`, `base.py`
- Sphinx docs build: 0 warnings

### Removed
- `RetinaFaceDetector` (replaced by YOLO11 / YOLOv8 detectors)
- MediaPipe face detector (replaced by YOLO11 / BlazeFace)
- `TDDFA_V2` head pose module (replaced by `SixDRepNetWrapper`)
- Blur detection and IoU tracker standalone modules (functionality absorbed into core)
- `OpenFaceWrapper` action unit module (replaced by `OpenGraphAuWrapper`)
- ResNet visual feature wrapper (replaced by CLIP, SwinT, FAb-Net)
- Submodule dependencies

---

## [1.4.0] — 2024-03-01

### Added
- WavLM wrapper (`WavlmWrapper`) — base / base+ / large variants, layer-wise hidden states (768-d / 1024-d)
- `batch_audio_to_features` for variable-length batched inference in Wav2Vec2 and WavLM

---

## [1.3.0] — 2023-10-01

### Added
- OpenGraphAU wrapper (`OpenGraphAuWrapper`) — 41-dim AU intensity vector using Swin-T backbone
- `AU_REGISTRY` with 41 bilateral action unit definitions

---

## [1.2.0] — 2023-06-01

### Added
- 3DDFA_V2 head pose estimation integrated as core feature (pitch, yaw, roll in degrees)
- CLAP wrapper (`ClapWrapper`) — laion/larger_clap_music_and_speech, 512-d audio–language embeddings
- BlinkDenseNet121 wrapper — per-eye open/closed probability with `predict_frame` and `predict_pipeline`
- L2CS-Net gaze wrapper (`L2csNetWrapper`) — ResNet-50, MPIIFaceGaze, pitch/yaw in radians
- Blur detection module

---

## [1.1.0] — 2023-01-01

### Added
- Wav2Vec2 wrapper (`Wav2vec2Wrapper`) — facebook/wav2vec2-base-960h, (T, 768) representations
- RoBERTa wrapper — roberta-large, token-level and sentence-level embeddings (1024-d)
- IoU-based multi-face tracker for temporal track assignment and merging
- R2Plus1D video feature extractor

---

## [1.0.0] — 2022-09-01

### Added
- Initial public release
- Audio: I/O, MFCC / Mel-spectrogram, prosody (pitch, energy, voice ratio), OpenSMILE eGeMAPSv02
- Video: FaceMesh 478-pt dense landmarks, MediaPipe Iris, FAb-Net (256-d face features), CLIP
- Text: BERT (bert-base-uncased, 768-d), Whisper speech-to-text, XML-RoBERTa (xlm-roberta-base, 768-d)
- Utilities: device management, `@load_or_create` caching, normalization, padding, loss functions (MSE, CCC)
- PyPI package with optional extras: audio, video, text, all

---

[2.0.0]: https://github.com/fodorad/exordium/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/fodorad/exordium/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/fodorad/exordium/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/fodorad/exordium/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/fodorad/exordium/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/fodorad/exordium/releases/tag/v1.0.0
