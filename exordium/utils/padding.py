"""Padding and cropping utilities."""

import torch
import torch.nn.functional as F


def pad_or_crop_time_dim(
    tensor: torch.Tensor, target_size: int, pad_value: int | float = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust the time dimension (T) of a tensor to a given size and generate a mask.

    Handles tensors of shape ``(T, F)`` or ``(T,)``.

    Args:
        tensor: Input tensor with shape ``(T, F)`` or ``(T,)``.
        target_size: Desired size for the time dimension.
        pad_value: Value to use for padding. Defaults to 0.

    Returns:
        Tuple of ``(adjusted_tensor, mask)`` where ``mask`` is a bool tensor
        of shape ``(target_size,)`` with ``True`` for original values and
        ``False`` for padded positions.

    """
    is_vector = tensor.ndim == 1
    if is_vector:
        tensor = tensor[:, None]

    current_size = tensor.shape[0]

    if current_size < target_size:
        pad_amount = target_size - current_size
        padding = (0, 0, 0, pad_amount)
        result = F.pad(tensor, padding, mode="constant", value=pad_value)
        mask = torch.cat(
            [
                torch.ones(current_size, dtype=torch.bool),
                torch.zeros(pad_amount, dtype=torch.bool),
            ]
        )
    elif current_size > target_size:
        result = tensor[:target_size, :]
        mask = torch.ones(target_size, dtype=torch.bool)
    else:
        result = tensor
        mask = torch.ones(current_size, dtype=torch.bool)

    if is_vector:
        result = result.squeeze(1)

    return result, mask


def pad_layered_time_dim(
    tensor: torch.Tensor, target_time_dim: int, pad_value: int | float = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the time dimension of a tensor and generate a mask.

    Args:
        tensor: Input tensor of shape ``(L, N, F)``.
        target_time_dim: Desired size for the time dimension ``N``.
        pad_value: Value to use for padding. Defaults to 0.

    Returns:
        Tuple of ``(padded_tensor, mask)`` where the tensor has shape
        ``(L, target_time_dim, F)`` and the mask has shape
        ``(target_time_dim,)`` with ``True`` for original data and ``False``
        for padding.

    """
    _, N, _ = tensor.shape

    mask = torch.ones(N, dtype=torch.bool)
    if N < target_time_dim:
        mask = torch.cat([mask, torch.zeros(target_time_dim - N, dtype=torch.bool)], dim=0)

    if N >= target_time_dim:
        return tensor[:, :target_time_dim, :], mask[:target_time_dim]

    pad_amount = target_time_dim - N
    padded = F.pad(tensor, (0, 0, 0, pad_amount, 0, 0), mode="constant", value=pad_value)
    return padded, mask
