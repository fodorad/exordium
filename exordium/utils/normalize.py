import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_mean_std(
    dataloader: DataLoader, ndim: int, verbose: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean and std values of samples for standardization using a DataLoader object.

    VAR[X] = E[X**2] - E[X]**2

    Args:
        dataloader: DataLoader of samples.
        ndim: Dimensionality of the samples.
        verbose: Show progress bar. Defaults to True.

    Raises:
        NotImplementedError: if given ndim is not supported.

    Returns:
        Tuple of mean and std tensors.

    """
    ndim_to_dim = {
        2: [0],  # vectors - (B, C)
        3: [0, 1],  # time series - (B, T, C)
        4: [0, 2, 3],  # images - (B, C, H, W)
        5: [0, 1, 3, 4],  # videos - (B, T, C, H, W)
    }

    if ndim not in ndim_to_dim:
        raise NotImplementedError(f"Given ndim {ndim} is not supported.")

    dim = ndim_to_dim[ndim]
    channel_dim = list(set(range(ndim)) - set(dim))[0]

    first_batch, _ = next(iter(dataloader))
    channel_size = first_batch.shape[channel_dim]

    channels_sum, channels_squared_sum, num_batches = (
        torch.zeros((channel_size,)),
        torch.zeros((channel_size,)),
        torch.tensor(0),
    )
    for data, _ in tqdm(
        dataloader, total=len(dataloader), desc="Calculate MEAN/STD values", disable=not verbose
    ):
        channels_sum += torch.mean(data, dim=dim)
        channels_squared_sum += torch.mean(data**2, dim=dim)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    return mean, std


def standardization(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Applies standardization to a given torch.Tensor.

    Args:
        x (torch.Tensor): input data of shape (B, F) or (B, T, F).
        mean (torch.Tensor): mean value.
        std (torch.Tensor): std value.

    Returns:
        torch.Tensor: standardized data.

    """
    return (x - mean) / (std + torch.finfo(torch.float32).eps)


def save_params_to_json(mean: torch.Tensor, std: torch.Tensor, file_path: os.PathLike):
    """Save standardization params to a JSON file.

    Args:
        mean (torch.Tensor): First vector to save.
        std (torch.Tensor): Second vector to save.
        file_path (os.PathLike): Path to the JSON file.

    """
    data = {"mean": mean.tolist(), "std": std.tolist()}

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(str(file_path), "w") as f:
        json.dump(data, f)


def load_params_from_json(file_path: os.PathLike) -> tuple[torch.Tensor, torch.Tensor]:
    """Load standardization params from a JSON file.

    Args:
        file_path (os.PathLike): Path to the JSON file.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loaded params as PyTorch tensors.

    """
    with open(str(file_path)) as f:
        data = json.load(f)

    mean = torch.FloatTensor(data["mean"])
    std = torch.FloatTensor(data["std"])

    return mean, std
