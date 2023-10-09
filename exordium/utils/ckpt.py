import os
import logging
from pathlib import Path
from exordium import PathType


def download_file(remote_path: PathType, local_path: PathType, overwrite: bool = False) -> None:
    """Downloads a file to given path.

    Args:
        remote_path (PathType): path to the remote file.
        local_path (PathType): path to the local file.

    Raises:
        FileNotFoundError: if the file is not downloaded successfully.
    """
    local_path = Path(local_path)
    if not local_path.exists() or overwrite:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        CMD = f'wget -t 5 {remote_path} -O {local_path}'
        logging.info(CMD)
        os.system(CMD)

    if not local_path.exists():
        raise FileNotFoundError(f'Downloaded file is missing at {local_path}.')


def remove_token(weights: dict, token: str = '_model.') -> dict:
    """Removes a token from the keys of the weight dictionary.
    The ckpt file saves the pytorch_lightning module which includes it's child members.
    Loading the state_dict with the child called '_model.' leads to an error. Removing the
    unnecessary token fixes the checkpoint loading method.

    Args:
        weights (dict): loaded torch weights.
        token (str, optional): string to be removed. Defaults to 'model.'.

    Returns:
        dict: updated weight dictionary.
    """
    return {k.replace(token, ''): v for k, v in weights.items()}