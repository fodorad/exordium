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


def repeat_pad_time_dim(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad the time dimension by repeating the last frame.

    If ``target_size <= T`` the tensor is returned unchanged (no cropping).

    Args:
        tensor: Input tensor of shape ``(T, ...)`` where ``T`` is the time
            dimension (e.g. ``(T, C, H, W)`` or ``(T, D)``).
        target_size: Desired length of the time dimension.

    Returns:
        Tensor of shape ``(target_size, ...)`` where frames beyond the
        original length are copies of the last frame.

    Raises:
        ValueError: If the tensor has zero frames.

    Example::

        >>> x = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
        >>> repeat_pad_time_dim(x, 5)
        tensor([[1, 2], [3, 4], [3, 4], [3, 4], [3, 4]])

    """
    if tensor.shape[0] == 0:
        raise ValueError("Cannot repeat-pad a tensor with zero frames.")
    current = tensor.shape[0]
    if current >= target_size:
        return tensor
    pad_count = target_size - current
    last = tensor[-1:].expand(pad_count, *(-1,) * (tensor.ndim - 1))
    return torch.cat([tensor, last], dim=0)


def fill_gaps_with_repeat(
    tensor: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill zero-valued time steps by repeating the nearest earlier valid frame.

    Given a tensor of shape ``(T, ...)`` where some time steps are entirely
    zero (no data), this function fills each gap by copying the closest
    preceding valid frame.  If the tensor starts with zeros before the first
    valid frame, those leading positions are back-filled from the first valid
    frame.

    A time step is considered **valid** when any element is non-zero, unless
    an explicit ``valid_mask`` is provided.

    Args:
        tensor: Input tensor of shape ``(T, ...)``.
        valid_mask: Optional boolean tensor of shape ``(T,)``.  ``True``
            marks valid (non-gap) time steps.  When ``None``, validity is
            inferred as ``tensor.flatten(1).any(dim=1)``.

    Returns:
        Tensor of the same shape with all gaps filled.

    Raises:
        ValueError: If the tensor contains no valid frames at all.

    Example::

        >>> x = torch.zeros(6, 2)
        >>> x[1] = torch.tensor([1.0, 2.0])
        >>> x[4] = torch.tensor([3.0, 4.0])
        >>> fill_gaps_with_repeat(x)
        tensor([[1., 2.],   # back-filled from frame 1
                [1., 2.],   # valid
                [1., 2.],   # forward-filled from frame 1
                [1., 2.],   # forward-filled from frame 1
                [3., 4.],   # valid
                [3., 4.]])  # forward-filled from frame 4

    """
    t = tensor.shape[0]
    if t == 0:
        return tensor

    if valid_mask is None:
        valid_mask = tensor.reshape(t, -1).any(dim=1)

    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        raise ValueError("Tensor contains no valid frames — cannot fill gaps.")

    result = tensor.clone()

    # Back-fill leading zeros from first valid frame
    first_valid = int(valid_indices[0].item())
    if first_valid > 0:
        result[:first_valid] = tensor[first_valid]

    # Forward-fill remaining gaps
    last_valid_idx = first_valid
    for i in range(first_valid + 1, t):
        if valid_mask[i]:
            last_valid_idx = i
        else:
            result[i] = result[last_valid_idx]

    return result
