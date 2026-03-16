"""Device management utilities."""

import torch


def get_torch_device(device_id: int | None = None) -> torch.device:
    """Get a PyTorch device based on device ID.

    Args:
        device_id: Device ID. None or negative returns CPU. Defaults to None.

    Returns:
        PyTorch device (MPS > CUDA > CPU based on availability).
    """
    if device_id is None:
        return torch.device("cpu")

    if isinstance(device_id, int) and device_id < 0:
        return torch.device("cpu")

    if torch.mps.is_available():
        # Apple Silicon GPU
        return torch.device(f"mps:{device_id}")

    if torch.cuda.is_available():
        # Nvidia GPU
        return torch.device(f"cuda:{device_id}")

    return torch.device("cpu")


def get_default_device() -> torch.device:
    """Get the default device (GPU or CPU).

    Returns:
        The device with highest priority available: MPS > CUDA > CPU.
    """
    if torch.mps.is_available():
        return torch.device("mps:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_module_device(module: torch.nn.Module) -> torch.device:
    """Get the device of a PyTorch module.

    Args:
        module: PyTorch module.

    Returns:
        Device of the module's parameters.
    """
    return next(module.parameters()).device
