import numpy as np
import torch


def pad_or_crop_time_dim(array: np.ndarray | torch.Tensor, target_size: int, pad_value: int | float = 0):
    """
    Adjust the time dimension (T) of an array (T, F) or vector (T,) to a given size and generate a mask,
    works with both NumPy and PyTorch tensors.
    
    Args:
        array (np.ndarray or torch.Tensor): Input array with shape (T, F) or (T,).
        target_size (int): Desired size for the time dimension.
        pad_value (int or float): Value to use for padding if necessary (default is 0).
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray or torch.Tensor: The array/tensor with time dimension padded or cropped to the target size.
            - np.ndarray or torch.BoolTensor: Boolean mask of shape (T,) with `True` for original values and `False` for padded values.
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
    feature_dim = array.shape[1]

    if current_size < target_size:
        # Padding is required
        pad_amount = target_size - current_size
        if is_numpy:
            # For NumPy arrays
            padding = ((0, pad_amount), (0, 0))  # Pad only the time dimension
            padded_array = np.pad(array, padding, mode='constant', constant_values=pad_value)
            mask = np.concatenate([np.ones(current_size, dtype=bool), np.zeros(pad_amount, dtype=bool)])
        elif is_torch:
            # For PyTorch tensors
            padding = (0, 0, 0, pad_amount)  # PyTorch pad format: (F-right, F-left, T-right, T-left)
            padded_array = torch.nn.functional.pad(array, padding, mode='constant', value=pad_value)
            mask = torch.cat([torch.ones(current_size, dtype=torch.bool), torch.zeros(pad_amount, dtype=torch.bool)])
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