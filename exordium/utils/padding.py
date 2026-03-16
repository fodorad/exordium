import numpy as np
import torch
import torch.nn.functional as F


def pad_or_crop_time_dim(
    array: np.ndarray | torch.Tensor, target_size: int, pad_value: int | float = 0
):
    """Adjust the time dimension (T) of an array to a given size and generate a mask.

    Works with both NumPy arrays and PyTorch tensors. Handles arrays of shape (T, F)
    or vectors of shape (T,).

    Args:
        array: Input array with shape (T, F) or (T,).
        target_size: Desired size for the time dimension.
        pad_value: Value to use for padding if necessary. Defaults to 0.

    Returns:
        Tuple containing the adjusted array/tensor and a boolean mask of shape (T,)
        with True for original values and False for padded values.

    """
    # Check if the input is a NumPy array or a PyTorch tensor
    is_numpy = isinstance(array, np.ndarray)
    is_torch = isinstance(array, torch.Tensor)

    if not is_numpy and not is_torch:
        raise ValueError("Input must be a NumPy array or a PyTorch tensor")

    # Reshape (T,) to (T, 1) for uniform handling
    is_vector = array.ndim == 1
    if is_vector:
        array = array[:, None]

    current_size = array.shape[0]
    # feature_dim = array.shape[1]

    if current_size < target_size:
        # Padding is required
        pad_amount = target_size - current_size
        if is_numpy:
            # For NumPy arrays
            padding = ((0, pad_amount), (0, 0))  # Pad only the time dimension
            padded_array = np.pad(array, padding, mode="constant", constant_values=pad_value)
            mask = np.concatenate(
                [np.ones(current_size, dtype=bool), np.zeros(pad_amount, dtype=bool)]
            )
        elif is_torch:
            # PyTorch tensors
            padding = (
                0,  # F-right
                0,  # F-left
                0,  # T-right
                pad_amount,  # T-left
            )
            padded_array = torch.nn.functional.pad(array, padding, mode="constant", value=pad_value)
            mask = torch.cat(
                [
                    torch.ones(current_size, dtype=torch.bool),
                    torch.zeros(pad_amount, dtype=torch.bool),
                ]
            )
    elif current_size > target_size:
        # Cropping is required
        padded_array = array[:target_size, :]
        if is_numpy:
            mask = np.ones(target_size, dtype=bool)
        elif is_torch:
            mask = torch.ones(target_size, dtype=bool)
    else:
        # No padding or cropping required
        padded_array = array
        if is_numpy:
            mask = np.ones(current_size, dtype=bool)
        elif is_torch:
            mask = torch.ones(current_size, dtype=bool)

    # Squeeze back to (T,) if the input was a vector
    if is_vector:
        padded_array = padded_array.squeeze(1)

    return padded_array, mask


def pad_layered_time_dim(
    tensor: torch.Tensor, target_time_dim: int, pad_value: int | float = 0
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """Pads the time dimension of a tensor and generates a mask.

    Args:
        tensor (torch.Tensor): Input tensor of shape (L, N, F).
        target_time_dim (int): Desired size for the time dimension (N).
        pad_value (int or float): Value to use for padding (default is 0).

    Returns:
        tuple:
            - torch.Tensor: Padded tensor with shape (L, target_time_dim, F).
            - torch.BoolTensor: Mask tensor of shape (target_time_dim,) with `1` for original data
                and `0` for padding.

    """
    _, N, _ = tensor.shape

    # Create the mask
    mask = torch.ones(N, dtype=torch.bool)
    if N < target_time_dim:
        # Extend the mask for padding
        mask = torch.cat([mask, torch.zeros(target_time_dim - N, dtype=torch.bool)], dim=0)

    if N >= target_time_dim:
        # No padding needed, crop to target_time_dim
        return tensor[:, :target_time_dim, :], mask[:target_time_dim]

    # Calculate padding
    pad_amount = target_time_dim - N
    padded_tensor = F.pad(tensor, (0, 0, 0, pad_amount, 0, 0), mode="constant", value=pad_value)
    return padded_tensor, mask
