"""Test fixture paths and lightweight weight-availability helpers."""

import gc
import pathlib
import types
import unittest
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.client import HTTPResponse

FIXTURES_ROOT = pathlib.Path(__file__).parent


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
