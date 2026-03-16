"""Checkpoint utility functions."""

import logging
import urllib.request
from pathlib import Path

import torch


def get_logger(name: str, path: Path | str) -> logging.Logger:
    """Creates and configures a logger with file handler.

    Args:
        name (str): Name of the logger.
        path (Path | str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_handler = logging.FileHandler(path)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)
    return logger


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
        logging.info(f"Downloading {remote_path} → {local_path}")
        urllib.request.urlretrieve(str(remote_path), str(local_path))

    if not local_path.exists():
        raise FileNotFoundError(f"Downloaded file is missing at {local_path}.")


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
