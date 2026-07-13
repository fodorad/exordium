"""Test fixture paths and lightweight weight-availability helpers."""

import contextlib
import gc
import logging
import pathlib
import types
import unittest
import urllib.request
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.client import HTTPResponse

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


def head_ok(url: str, timeout: int = 15) -> bool:
    """Return True if a HEAD request to *url* returns HTTP 2xx or 3xx."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        resp: HTTPResponse = urllib.request.urlopen(req, timeout=timeout)
        return resp.status < 400
    except Exception:
        return False


def hf_repo_exists(repo_id: str, timeout: int = 15) -> bool:
    """Return True if a HuggingFace Hub model repo exists and is reachable."""
    from huggingface_hub import model_info
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        model_info(repo_id, timeout=timeout)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception:
        return False


def hf_file_exists(repo_id: str, filename: str) -> bool:
    """Return True if *filename* exists in a HuggingFace Hub model repo."""
    from huggingface_hub import list_repo_files
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        files = list(list_repo_files(repo_id))
        return filename in files
    except RepositoryNotFoundError:
        return False
    except Exception:
        return False


AUDIO_MULTISPEAKER = FIXTURES_ROOT / "audio" / "multispeaker.wav"
AUDIO_FI_PROTAGONIST = FIXTURES_ROOT / "audio" / "fi_protagonist.wav"
IMAGE_CAT_TIE = FIXTURES_ROOT / "image" / "cat_tie.jpg"
IMAGE_MULTISPEAKER = FIXTURES_ROOT / "image" / "multispeaker.png"
IMAGE_FACE = FIXTURES_ROOT / "image" / "face.jpg"
IMAGE_EMMA = FIXTURES_ROOT / "image" / "emma.jpg"
VIDEO_MULTISPEAKER = FIXTURES_ROOT / "video" / "multispeaker.mp4"
VIDEO_MULTISPEAKER_SHORT = FIXTURES_ROOT / "video" / "multispeaker_short.mp4"
