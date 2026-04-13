Exordium
========

Collection of preprocessing functions and deep learning methods for **multimodal feature extraction** across audio, video, and text modalities.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   autoapi/index

Features
--------

**Audio**

- I/O utilities — load, save, resample waveforms
- Spectrograms — MFCC and Mel-spectrogram (with pre-emphasis)
- OpenSMILE — eGeMAPSv02 low-level descriptors
- CLAP — contrastive language–audio pre-training embeddings (512-d)
- Wav2Vec2 — self-supervised speech representations (base-960h / emotion-iemocap, 768-d)
- WavLM — masked speech modelling, layer-wise hidden states (768-d)
- emotion2vec+ — speech emotion features at ~50 Hz (768-d)

**Video**

- Face detection — YOLOv8-Face, YOLO11-pose, BlazeFace
- Facial landmarks — YOLO11 5-pt coarse keypoints, FaceMesh 478-pt dense mesh
- Head pose — SixDRepNet (yaw, pitch, roll in degrees)
- Gaze direction — L2CS-Net (ResNet-50), UniGaze (ViT), roll correction
- Iris landmarks — MediaPipe Iris 71 eye pts + 5 iris pts, EAR, iris diameters
- Blink detection — BlinkDenseNet121 per-eye open/closed probability
- Action units — OpenGraphAU 41-dim intensity vector
- Deep visual features — Swin Transformer (768-d), AdaFace IResNet-18/50/101 (512-d identity embeddings), FAb-Net (256-d), CLIP ViT-H/14 (1024-d), DINOv2 (384 / 768 / 1024 / 1536-d), EmotiEffNet (1280 / 1408-d), MARLIN (384 / 768 / 1024-d, 16-frame clips)
- Tracking — IoU-based multi-face tracker; face-ID tracker (AdaFace embeddings + IoU gating) with occlusion recovery and identity-aware merge

**Text**

- Whisper — speech-to-text transcription (OpenAI Whisper)
- BERT — token-level and sentence-level embeddings (768-d)
- RoBERTa — token-level and sentence-level embeddings (1024-d)
- XML-RoBERTa — multilingual sentence embeddings (768-d)

**Utilities**

- Concurrent processing — thread- and process-pool helpers
- Decorators — ``@load_or_create`` caching, retry, timing
- Normalization — global, per-feature, sliding-window
- Padding — fixed-length sequence padding and masking
- Loss functions — MSE, CCC, combined losses

Installation
------------

.. code-block:: bash

   pip install exordium          # base only
   pip install exordium[all]     # all optional dependencies
   pip install exordium[audio]   # audio extras only
   pip install exordium[video]   # video extras only
   pip install exordium[text]    # text extras only

Extras
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Extra
     - Dependencies
   * - ``audio``
     - OpenSMILE, torchaudio — audio feature extraction
   * - ``text``
     - transformers, torchaudio — text and speech models
   * - ``video``
     - MediaPipe, Ultralytics, blinklinmult, unigaze, timm — face & video models
   * - ``all``
     - all previously described extras

Example Notebooks
-----------------

Interactive demos for every module are in the ``examples/`` directory.
All notebooks use fixture files from ``tests/fixtures/`` — model weights are
downloaded automatically on first run.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Notebook
     - Description
   * - ``demo_video_io.ipynb``
     - Video loading, frame iteration, batch access, FPS resampling
   * - ``demo_video_deep.ipynb``
     - Deep visual features: SwinT (768-d), AdaFace (512-d identity), FabNet (256-d), CLIP ViT-H/14 (1024-d), DINOv2 (384–1536-d), EmotiEffNet (1280/1408-d), MARLIN (384–1024-d, 16-frame clips)
   * - ``demo_video_face_bb.ipynb``
     - Face detection: YOLOv8 vs YOLO11 on easy and hard (extreme-pose) images
   * - ``demo_video_face_landmarks.ipynb``
     - Landmarks: YOLO11 5-pt keypoints + FaceMesh 478-pt dense mesh
   * - ``demo_video_face_headpose.ipynb``
     - Head pose (SixDRepNet) — yaw/pitch/roll with axis and cube overlays
   * - ``demo_video_face_gaze.ipynb``
     - Gaze direction: L2CS-Net (ResNet-50) and UniGaze (ViT), roll correction
   * - ``demo_video_face_blink.ipynb``
     - Per-eye blink detection on video — frame-wise score plot, patch examples
   * - ``demo_video_face_iris.ipynb``
     - Iris landmarks: 71 eye pts + 5 iris pts, EAR, iris diameters (YOLO11 + FaceMesh pipeline)
   * - ``demo_video_face_action_units.ipynb``
     - Facial action units with OpenGraphAU — 41-dim AU intensity vector
   * - ``demo_video_tracking.ipynb``
     - Multi-face tracking: IoU tracker and face-ID tracker (AdaFace) with occlusion recovery
   * - ``demo_audio.ipynb``
     - Audio features: spectrogram, OpenSMILE, CLAP, Wav2Vec2, WavLM, emotion2vec+
   * - ``demo_text.ipynb``
     - Text features: Whisper transcription, BERT, RoBERTa, XML-RoBERTa

Development
-----------

.. code-block:: bash

   git clone https://github.com/fodorad/exordium
   cd exordium
   uv pip install -e ".[all,dev]"
   make check   # lint + type-check + test + docs

Related Projects
----------------

EmotionLinMulT (202X)
^^^^^^^^^^^^^^^^^^^^^

Efficient, transformer-based, multi-task emotion detection system.

- Paper: not published yet
- `Code <https://github.com/fodorad/EmotionLinMulT>`_

BlinkLinMulT (2023)
^^^^^^^^^^^^^^^^^^^

Transformer-based eye blink detection and eye state recognition across 7 public benchmark databases.

- `Paper <https://www.mdpi.com/2313-433X/9/10/196>`_
- `Code <https://github.com/fodorad/BlinkLinMulT>`_

PersonalityLinMulT (2022)
^^^^^^^^^^^^^^^^^^^^^^^^^

LinMulT trained for Big Five personality trait estimation and sentiment analysis.

- `Paper <https://proceedings.mlr.press/v173/fodor22a.html>`_
- `Code <https://github.com/fodorad/PersonalityLinMulT>`_

LinMulT
^^^^^^^

General-purpose multimodal transformer with linear-complexity attention mechanisms.

- `Website <https://adamfodor.com/LinMulT/>`_
- `Code <https://github.com/fodorad/LinMulT>`_

Contact
-------

**Ádám Fodor** — `adamfodor.com <https://adamfodor.com>`_ · fodorad201@gmail.com
