"""Test fixture paths and lightweight weight-availability helpers."""

import contextlib
import gc
import logging
import os
import pathlib
import time
import types
import unittest
from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

# Force anonymous HuggingFace Hub access for the whole test process.
#
# Every test module imports this package, so setting the flag here runs before
# any Hub call is made. It stops huggingface_hub from attaching a locally stored
# token to requests. A *stale or invalid* token is worse than none: the Hub 401s
# the whole request instead of falling back to anonymous access, which mass-fails
# the weight-availability and wrapper setUpClass tests even though every repo they
# touch is public. Setting it with ``setdefault`` lets CI still opt into an
# explicit token by exporting HF_HUB_DISABLE_IMPLICIT_TOKEN=0 if ever needed.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

FIXTURES_ROOT = pathlib.Path(__file__).parent

TEST_CLIP_MODEL = "openai/clip-vit-base-patch32"
"""Smallest CLIP config used by the tests (88M params, vs 632M for the H/14 default)."""

TEST_DINOV2_MODEL = "small"
"""Smallest DINOv2 variant used by the tests."""

TEST_MARLIN_MODEL = "small"
"""Smallest MARLIN variant used by the tests."""

TEST_WHISPER_MODEL = "openai/whisper-tiny"
"""Smallest Whisper config used by the tests (39M params, vs 756M for distil-large-v3).

Since :data:`PRETRAINED` is ``False``, a model id only selects an *architecture* — no
checkpoint is fetched either way. Building the production-sized variants meant randomly
initialising hundreds of millions of parameters just to assert a tensor shape, which
dominated the suite's runtime. The smallest config in each family exercises exactly the
same wrappers, shapes-by-config and call paths, for a fraction of the setup cost.
"""

PRETRAINED = False
"""Whether tests build wrappers with real pretrained weights. Always ``False``.

The suite downloads **no checkpoints**. Wrappers are constructed with
``pretrained=False``, which fetches a few KB of ``config.json`` and instantiates the
architecture with random weights instead of pulling gigabytes — the whole suite went
from ~12 GB of downloads to ~17 MB.

Random weights still exercise everything a unit test should: input/output shapes, device
placement, batching, preprocessing, and error paths. What they *cannot* check is whether
a prediction is **correct** — that is a property of the weights, not the code. Those
correctness demonstrations live in the example notebooks, run client-side, deliberately
out of CI.

Weights are still verified to be *reachable* — see :func:`hf_file_exists` and
:func:`hf_repo_exists`, which issue a HEAD request and download nothing.
"""


def best_anchored_word(words: list) -> object:
    """Pick the word whose own alignment is trustworthy enough to test against.

    The speech fixtures' reference transcripts cover only part of a long multi-speaker
    recording, so an aligner smears the unmatched tail across huge spans (whole "words"
    lasting 12 s). Only tightly-bounded, high-confidence words are usable as ground truth.

    Args:
        words: Word stream from a forced aligner.

    Returns:
        The highest-scoring word with a plausible duration.

    """
    anchored = [w for w in words if 0.05 <= (w.end - w.start) <= 0.4]
    return max(anchored, key=lambda w: w.score)


@contextlib.contextmanager
def logging_enabled() -> Iterator[None]:
    """Undo the suite-wide ``logging.disable(WARNING)`` for one block.

    ``tests/__init__`` silences warnings so model-loading noise stays out of the
    test output, but that also stops :meth:`~unittest.TestCase.assertLogs` from
    ever seeing a record. Wrap a block in this to assert on log output::

        with logging_enabled(), self.assertLogs("exordium.text.base", "WARNING"):
            ...
    """
    logging.disable(logging.NOTSET)
    try:
        yield
    finally:
        logging.disable(logging.WARNING)


def free_torch_memory() -> None:
    """Run garbage collection and empty the CUDA/MPS caching allocators.

    The full test suite runs in a single process (``unittest discover``), so
    models loaded in one test class stay resident unless released. Calling this
    after dropping model references keeps peak memory bounded.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.empty_cache()


class ModelTestCase(unittest.TestCase):
    """``TestCase`` that frees class-level models after all its tests run.

    Test classes that load a model once in :meth:`setUpClass` (e.g.
    ``cls.model = SomeWrapper(...)``) should subclass this instead of
    :class:`unittest.TestCase`. Its :meth:`tearDownClass` deletes the data
    attributes assigned to the class and empties the torch allocator caches, so
    each class's models do not accumulate across the whole suite.
    """

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete class-level model attributes and free torch caches."""
        # Drop data attributes set in setUpClass (models, loaders, tensors),
        # keeping methods/classmethods/etc. so the class stays well-formed.
        for name, value in list(cls.__dict__.items()):
            if name.startswith("__"):
                continue
            if isinstance(value, (types.FunctionType, classmethod, staticmethod, property)):
                continue
            try:
                delattr(cls, name)
            except AttributeError:
                pass
        free_torch_memory()


class PooledTokenWrapperContract:
    """Shared real-model contract for the multilingual token/pooled wrappers.

    Mix into a :class:`ModelTestCase` whose ``setUpClass`` sets ``cls.model`` to a
    :class:`~exordium.text.base.PooledTokenTextWrapper` subclass. Asserts the
    ``pooling`` API on the concrete class without duplicating the body across the
    per-model test files (``test_xlm_roberta`` / ``test_mmbert`` / ``test_eurobert``)::

        class TestMmbertWrapper(PooledTokenWrapperContract, ModelTestCase):
            @classmethod
            def setUpClass(cls):
                from exordium.text.mmbert import MmbertWrapper
                cls.model = MmbertWrapper(device_id=None, pretrained=PRETRAINED)
    """

    def test_call_returns_tensor(self):
        """Calling the wrapper on a string returns a torch tensor."""
        import torch

        out = self.model("hello world")
        self.assertIsInstance(out, torch.Tensor)

    def test_default_output_is_pooled(self):
        """The default output is a pooled ``(B, H)`` vector — rank 2."""
        out = self.model("hello world")
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[-1], self.model.hidden_size)

    def test_pooling_none_is_token_level(self):
        """``pooling='none'`` returns a token-level ``(B, T, H)`` sequence — rank 3."""
        out = self.model("hello world", pooling="none")
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[-1], self.model.hidden_size)

    def test_hidden_size_matches_config(self):
        """``hidden_size`` mirrors the backbone's ``config.hidden_size``."""
        self.assertEqual(self.model.hidden_size, self.model.model.config.hidden_size)

    def test_predict_returns_numpy(self):
        """``predict`` returns a rank-2 numpy array by default (pooled)."""
        import numpy as np

        out = self.model.predict("hello world")
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.ndim, 2)

    def test_batch_input_pooled(self):
        """A batch of two pooled vectors is rank 2 with a leading batch of 2."""
        out = self.model(["hello world", "a slightly longer second sentence"])
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (2, self.model.hidden_size))

    def test_batch_input_token_level(self):
        """A batch of two token sequences is rank 3; padded to a shared length T."""
        out = self.model(["hello world", "a slightly longer second sentence"], pooling="none")
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[-1], self.model.hidden_size)


_HF_PROBE_RETRY_PAUSES_S = (0.0, 1.0)
"""Pause before each retry after a transient failure, in seconds.

One initial attempt plus one retry per entry: the first retry is instant, the
second waits 1s. Three attempts total.
"""


def _hf_probe(probe: Callable[[], bool]) -> bool:
    """Run a Hub availability *probe*, retrying only on transient failures.

    A ``RepositoryNotFoundError`` (or a probe returning ``False``) is a
    definitive answer and is returned immediately -- retrying would only waste
    time. Any other exception is treated as transient (network blip, momentary
    5xx, presigned-URL hiccup) and retried, so a single flaky call among
    hundreds does not fail an otherwise-green suite. Retries follow
    :data:`_HF_PROBE_RETRY_PAUSES_S`: the first is instant, the second waits 1s.
    If every attempt raises, the probe is reported as unreachable (``False``).
    """
    from huggingface_hub.utils import RepositoryNotFoundError

    attempts = len(_HF_PROBE_RETRY_PAUSES_S) + 1
    for attempt in range(attempts):
        try:
            return probe()
        except RepositoryNotFoundError:
            return False
        except Exception as error:  # noqa: BLE001 - transient Hub/network failure, retry
            if attempt == attempts - 1:
                logger.warning(f"Hub probe failed after {attempts} attempts: {error}")
                return False
            time.sleep(_HF_PROBE_RETRY_PAUSES_S[attempt])
    return False


def hf_repo_exists(repo_id: str, timeout: int = 15) -> bool:
    """Return True if a HuggingFace Hub model repo exists and is reachable.

    ``token=False`` forces an anonymous request so a stale or invalid token in
    the local HuggingFace cache cannot poison the check: the repos under test are
    public, and an anonymous call succeeds where an invalid credential would 401.
    Transient failures are retried; see :func:`_hf_probe`.
    """
    from huggingface_hub import model_info

    def probe() -> bool:
        model_info(repo_id, timeout=timeout, token=False)
        return True

    return _hf_probe(probe)


def hf_file_exists(repo_id: str, filename: str) -> bool:
    """Return True if *filename* exists in a HuggingFace Hub model repo.

    ``token=False`` forces an anonymous request so a stale or invalid token in
    the local HuggingFace cache cannot poison the check: the repos under test are
    public, and an anonymous call succeeds where an invalid credential would 401.
    Transient failures are retried; see :func:`_hf_probe`.
    """
    from huggingface_hub import list_repo_files

    def probe() -> bool:
        return filename in list(list_repo_files(repo_id, token=False))

    return _hf_probe(probe)


AUDIO_MULTISPEAKER = FIXTURES_ROOT / "audio" / "multispeaker.wav"
AUDIO_FI_PROTAGONIST = FIXTURES_ROOT / "audio" / "fi_protagonist.wav"
IMAGE_CAT_TIE = FIXTURES_ROOT / "image" / "cat_tie.jpg"
IMAGE_MULTISPEAKER = FIXTURES_ROOT / "image" / "multispeaker.png"
IMAGE_FACE = FIXTURES_ROOT / "image" / "face.jpg"
IMAGE_EMMA = FIXTURES_ROOT / "image" / "emma.jpg"
VIDEO_MULTISPEAKER = FIXTURES_ROOT / "video" / "multispeaker.mp4"
VIDEO_MULTISPEAKER_SHORT = FIXTURES_ROOT / "video" / "multispeaker_short.mp4"
VIDEO_FI_PROTAGONIST = FIXTURES_ROOT / "video" / "fi_protagonist.mp4"
