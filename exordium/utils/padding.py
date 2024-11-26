import numpy as np
import torch


def pad_or_crop_time_dim(array: np.ndarray | torch.Tensor, target_size: int, pad_value: int | float = 0):
    """
    Adjust the time dimension (T) of an array (T, F) to a given size, works with both NumPy and PyTorch tensors.
    
    Args:
        array (np.ndarray or torch.Tensor): Input array with shape (T, F), where T is the time dimension.
        target_size (int): Desired size for the time dimension.
        pad_value (int or float): Value to use for padding if necessary (default is 0).
    
    Returns:
        np.ndarray or torch.Tensor: The array/tensor with time dimension padded or cropped to the target size.
    """
    # Check if the input is a NumPy array or a PyTorch tensor
    is_numpy = isinstance(array, np.ndarray)
    is_torch = isinstance(array, torch.Tensor)
    
    if not is_numpy and not is_torch:
        raise ValueError("Input must be a NumPy array or a PyTorch tensor")

    current_size = array.shape[0]
    
    if current_size < target_size:
        # Padding is required
        pad_amount = target_size - current_size
        if is_numpy:
            # For NumPy arrays
            padding = ((0, pad_amount), (0, 0))  # Pad only the time dimension
            padded_array = np.pad(array, padding, mode='constant', constant_values=pad_value)
        elif is_torch:
            # For PyTorch tensors
            padding = (0, 0, 0, pad_amount)  # PyTorch pad format: (F-right, F-left, T-right, T-left)
            padded_array = torch.nn.functional.pad(array, padding, mode='constant', value=pad_value)
        return padded_array
    elif current_size > target_size:
        # Cropping is required
        cropped_array = array[:target_size, :]
        return cropped_array
    else:
        # No padding or cropping required, return the array/tensor as is
        return array