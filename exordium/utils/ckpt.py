"""Checkpoint utility functions."""

import logging
import urllib.request
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)
"""Module-level logger."""

_HF_REPO_ID = "fodorad/exordium-weights"
"""HuggingFace Hub repository ID for exordium model weights."""


def build_hf_model(
    model_cls: Any,
    model_id: str,
    pretrained: bool = True,
    config_cls: Any = None,
    **kwargs: Any,
) -> Any:
    """Build a HuggingFace model, with or without its pretrained weights.

    With ``pretrained=False`` only the model's ``config.json`` is fetched (kilobytes)
    and the architecture is instantiated with **random weights** — no checkpoint is
    downloaded.  The result has the same shapes and interface as the real model, so it
    is enough to exercise input/output contracts, inspect the architecture, or work
    offline; its *predictions are meaningless*.

    For reference, a CLIP ViT-H/14 costs ~15 KB this way instead of ~3.7 GB.

    Args:
        model_cls: The HuggingFace model class (e.g. ``CLIPVisionModelWithProjection``).
        model_id: HuggingFace Hub model identifier.
        pretrained: ``True`` (default) downloads the real weights. ``False`` builds the
            architecture with random weights.
        config_cls: Config class to load with. Defaults to ``AutoConfig``; pass a
            specific one when the model needs a sub-config (e.g. ``CLIPVisionConfig``
            for a vision tower inside a full CLIP checkpoint).
        **kwargs: Forwarded to ``from_pretrained`` when *pretrained* is ``True``.

    Returns:
        The instantiated model. Typed as ``Any`` because the concrete class comes from
        *model_cls*, mirroring ``from_pretrained``'s own dynamic return.

    """
    if pretrained:
        return model_cls.from_pretrained(model_id, **kwargs)

    import transformers as tfm

    logger.info(f"Building {model_id} architecture with random weights (no checkpoint).")
    config = (config_cls or tfm.AutoConfig).from_pretrained(model_id)
    # ``Auto*`` classes are factories and refuse direct construction; concrete model
    # classes take the config positionally.
    if hasattr(model_cls, "from_config"):
        return model_cls.from_config(config)
    return model_cls(config)


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
    upstream_repo_id: str | None = None,
    upstream_filename: str | None = None,
) -> Path:
    """Download a model weight file from Hugging Face Hub if not already cached.

    Fetches *filename* from :data:`_HF_REPO_ID` — our own mirror — and places it in
    *local_dir*.  Subsequent calls with the same arguments are no-ops.

    The mirror is preferred because upstream sources move or vanish (the 6DRepNet
    author's original link already did).  When *upstream_repo_id* is given it is used as
    a **fallback**: if the mirror is unreachable or missing the file, the original source
    is tried before giving up, so neither location is a single point of failure.

    Args:
        filename: Name of the file in *repo_id* (e.g. ``"fabnet_weights.pth"``).
        local_dir: Directory to store the file in; it lands at ``local_dir / filename``.
        repo_id: Hub repo to fetch from. Defaults to our mirror,
            ``"fodorad/exordium-weights"``.
        upstream_repo_id: Original Hub repo, used only if *repo_id* fails.
        upstream_filename: Filename within *upstream_repo_id*, if it differs from
            *filename*. Defaults to *filename*.

    Returns:
        Path to the downloaded (or already cached) local file.

    Raises:
        FileNotFoundError: If the file is missing after every download attempt.

    """
    from huggingface_hub import hf_hub_download

    local_dir = Path(local_dir)
    local_path = local_dir / filename
    if local_path.exists():
        return local_path

    local_dir.mkdir(parents=True, exist_ok=True)
    sources: list[tuple[str, str]] = [(repo_id, filename)]
    if upstream_repo_id is not None:
        sources.append((upstream_repo_id, upstream_filename or filename))

    for source_repo, source_file in sources:
        try:
            logger.info(f"Downloading {source_repo}/{source_file} → {local_path}")
            hf_hub_download(repo_id=source_repo, filename=source_file, local_dir=str(local_dir))
        except Exception as error:  # noqa: BLE001 - any Hub/network failure is a fallback trigger
            logger.warning(f"Could not fetch {source_repo}/{source_file}: {error}")
            continue

        # An upstream file may be named differently from ours; normalise it so the
        # local cache key stays stable regardless of which source answered.
        fetched = local_dir / source_file
        if fetched != local_path and fetched.exists():
            fetched.replace(local_path)

        if local_path.exists():
            return local_path
        logger.warning(f"{source_repo}/{source_file} reported success but {local_path} is missing")

    raise FileNotFoundError(
        f"Could not download {filename!r} from any of: {[repo for repo, _ in sources]}"
    )


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
