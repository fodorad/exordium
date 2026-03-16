Exordium
========

Collection of preprocessing functions and deep learning methods for **multimodal feature extraction** across audio, video, and text modalities.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   autoapi/index

Features
--------

.. raw:: html

   <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
     <div>
       <h4>Audio Processing</h4>
       <p>I/O, OpenSMILE, spectrograms, Wav2Vec2, CLAP, WavLM</p>
     </div>
     <div>
       <h4>Video Analysis</h4>
       <p>Face detection, landmarks, head pose, gaze, iris tracking, action units, feature extraction</p>
     </div>
     <div>
       <h4>Text Processing</h4>
       <p>BERT, RoBERTa, XML-RoBERTa</p>
     </div>
     <div>
       <h4>Utilities</h4>
       <p>Parallel processing, I/O helpers, loss functions, normalization, padding, visualization</p>
     </div>
   </div>

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

| ``audio`` | dependencies to process audio data |
| ``text`` | dependency to process textual data |
| ``face`` | dependencies for face detection, landmarks, and head pose estimation |
| ``video`` | dependencies for various video feature extraction methods |
| ``all`` | all previously described extras will be installed |

Quick start
-----------

Audio feature extraction (WavLM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from exordium.audio.wavlm import WavlmWrapper

   model = WavlmWrapper(device_id=-1, model_name="base+")
   waveform = np.random.rand(16000).astype(np.float32)
   features = model.audio_to_feature(waveform)
   # list of 12 numpy arrays, each (T, 768)

Text feature extraction (BERT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from exordium.text.bert import BertWrapper

   model = BertWrapper(device_id=-1)
   features = model("Hello, world!", pool=True)
   # torch.Tensor of shape (1, 768)

Video face detection
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from exordium.video.face import RetinaFaceDetector
   from exordium.video.io import images_to_np

   detector = RetinaFaceDetector()
   frames = images_to_np(["frame.jpg"], "RGB")
   detections = detector.detect_image(frames[0])

Development
-----------

For development with all dependencies:

.. code-block:: bash

   git clone https://github.com/fodorad/exordium
   cd exordium
   pip install -e ".[all,dev]"
   make check

Related Projects
----------------

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
