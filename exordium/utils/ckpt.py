"""Checkpoint utility functions."""

import logging
import urllib.request
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
"""Module-level logger."""

_HF_REPO_ID = "fodorad/exordium-weights"
"""HuggingFace Hub repository ID for exordium model weights."""


def download_file(remote_path: str, local_path: Path | str, overwrite: bool = False) -> None:
    """Downloads a file to a given local path using urllib.

    Args:
        remote_path (str): URL of the remote file.
        local_path (Path | str): Destination path for the downloaded file.
        overwrite (bool, optional): If True, re-downloads even if the file
            already exists. Defaults to False.

    Raises:
        FileNotFoundError: If the file is not present after downloading.

    """
    local_path = Path(local_path)
    if not local_path.exists() or overwrite:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {remote_path} → {local_path}")
        urllib.request.urlretrieve(str(remote_path), str(local_path))

    if not local_path.exists():
        raise FileNotFoundError(f"Downloaded file is missing at {local_path}.")


def download_weight(
    filename: str,
    local_dir: Path | str,
    repo_id: str = _HF_REPO_ID,
) -> Path:
    """Download a model weight file from Hugging Face Hub if not already cached.

    Uses ``huggingface_hub.hf_hub_download`` to fetch ``filename`` from
    ``repo_id`` and place it directly in ``local_dir``.  Subsequent calls
    with the same arguments are no-ops (file already present).

    Args:
        filename: Name of the file in the HF Hub repository (e.g.
            ``"fabnet_weights.pth"``).
        local_dir: Local directory where the file will be stored.  The file
            will be at ``local_dir / filename``.
        repo_id: Hugging Face Hub repository ID.  Defaults to
            ``"fodorad/exordium-weights"``.

    Returns:
        Path to the downloaded (or already cached) local file.

    Raises:
        FileNotFoundError: If the file is missing after the download attempt.

    """
    from huggingface_hub import hf_hub_download

    local_dir = Path(local_dir)
    local_path = local_dir / filename
    if not local_path.exists():
        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {repo_id}/{filename} → {local_path}")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))

    if not local_path.exists():
        raise FileNotFoundError(f"Weight file missing after download: {local_path}")

    return local_path


def remove_token(weights: dict, token: str = "_model.") -> dict:
    """Removes a token from the keys of the weight dictionary.

    The ckpt file saves the pytorch_lightning module which includes its child
    members.  Loading the state_dict with the child called ``_model.`` leads
    to an error; removing the unnecessary token fixes checkpoint loading.

    Args:
        weights (dict): Loaded torch weights.
        token (str, optional): String to be removed. Defaults to ``"_model."``.

    Returns:
        dict: Updated weight dictionary.

    """
    return {k.replace(token, ""): v for k, v in weights.items()}


def load_checkpoint(path: Path | str, strip_prefix: str = "module.") -> dict:
    """Load a torch checkpoint and strip DDP/Lightning key prefixes.

    Handles both raw ``state_dict`` files and Lightning checkpoint files that
    store the state dict under a ``"state_dict"`` key.  Keys starting with
    ``strip_prefix`` have that prefix removed so the dict can be passed
    directly to ``model.load_state_dict()``.

    Args:
        path (Path | str): Path to the ``.pth`` or ``.ckpt`` file.
        strip_prefix (str, optional): Prefix to remove from each key.
            Defaults to ``"module."`` (DDP convention).

    Returns:
        dict: Cleaned state dict ready for ``model.load_state_dict()``.

    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state: dict = ckpt.get("state_dict", ckpt)
    n = len(strip_prefix)
    return {(k[n:] if k.startswith(strip_prefix) else k): v for k, v in state.items()}
