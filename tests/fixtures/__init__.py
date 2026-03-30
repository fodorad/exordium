"""Test fixture paths and lightweight weight-availability helpers."""

import pathlib
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.client import HTTPResponse

FIXTURES_ROOT = pathlib.Path(__file__).parent


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
IMAGE_CAT_TIE = FIXTURES_ROOT / "image" / "cat_tie.jpg"
IMAGE_MULTISPEAKER = FIXTURES_ROOT / "image" / "multispeaker.png"
IMAGE_FACE = FIXTURES_ROOT / "image" / "face.jpg"
IMAGE_EMMA = FIXTURES_ROOT / "image" / "emma.jpg"
VIDEO_MULTISPEAKER = FIXTURES_ROOT / "video" / "multispeaker.mp4"
VIDEO_MULTISPEAKER_SHORT = FIXTURES_ROOT / "video" / "multispeaker_short.mp4"
