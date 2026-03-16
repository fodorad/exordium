Exordium
========

.. toctree::
   :maxdepth: 2
   :caption: Contents

   autoapi/index

Collection of utility tools and deep learning methods for multimodal feature extraction (audio, video, text).

Installation
------------

.. code-block:: bash

   pip install exordium          # base only
   pip install exordium[all]     # all optional dependencies
   pip install exordium[audio]   # audio extras only
   pip install exordium[video]   # video extras only
   pip install exordium[text]    # text extras only

Quick Start
-----------

**Audio feature extraction (WavLM):**

.. code-block:: python

   import numpy as np
   from exordium.audio.wavlm import WavlmWrapper

   model = WavlmWrapper(device_id=-1, model_name="base+")
   waveform = np.random.rand(16000).astype(np.float32)
   features = model.audio_to_feature(waveform)
   # list of 12 numpy arrays, each (T, 768)

**Text feature extraction (BERT):**

.. code-block:: python

   from exordium.text.bert import BertWrapper

   model = BertWrapper(device_id=-1)
   features = model("Hello, world!", pool=True)
   # torch.Tensor of shape (1, 768)

**Video face detection:**

.. code-block:: python

   from exordium.video.face import RetinaFaceDetector
   from exordium.video.io import images_to_np

   detector = RetinaFaceDetector()
   frames = images_to_np(["frame.jpg"], "RGB")
   detections = detector.detect_image(frames[0])

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
