from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def get_mean_std(loader: DataLoader, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean and std values of samples for standardization using a DataLoader object.

    VAR[X] = E[X**2] - E[X]**2

    Args:
        loader (DataLoader): dataloader of samples.
        ndim (int): dimensionality of the samples.

    Raises:
        NotImplementedError: if given ndim is not supported.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: mean and std values
    """
    ndim_to_dim = {
        2: [0],                    # vectors - (B, C)
        3: [0, 1],                 # time series - (B, T, C)
        4: [0, 2, 3],              # images - (B, C, H, W)
        5: [0, 1, 3, 4],           # videos - (B, T, C, H, W)
    }

    if ndim not in ndim_to_dim:
        raise NotImplementedError(f'Given ndim {ndim} is not supported.')

    dim = ndim_to_dim[ndim]

    channels_sum, channels_squared_sum, num_batches = torch.tensor(0), torch.tensor(0), torch.tensor(0)
    for data, _ in tqdm(loader, total=len(loader)):
        channels_sum += torch.mean(data, dim=dim)
        channels_squared_sum += torch.mean(data**2, dim=dim)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
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
    return (torch.FloatTensor(x) - mean) / (std + torch.finfo(torch.float32).eps)