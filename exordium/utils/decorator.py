"""Decorator utilities for caching and timing."""

import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)
"""Module-level logger."""


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that prints the execution time of a function.

    Args:
        func (Callable[..., Any]): The function to time.

    Returns:
        Callable[..., Any]: Wrapped function that prints elapsed time on each
            call and returns the original return value.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"Function took: {round(time.time() - before, 3)} seconds.")
        return result

    return wrapper


class Loader(ABC):
    """Abstract base class for serialization helpers."""

    @abstractmethod
    def load(self, path: str | Path):
        """Loads and returns data from the given path.

        Args:
            path (str | Path): File path to load from.

        Returns:
            Any: Deserialized data.

        """

    @abstractmethod
    def save(self, data, path: str | Path):
        """Serializes data and writes it to the given path.

        Args:
            data: Data to serialize.
            path (str | Path): Destination file path.

        """


class PickleLoader(Loader):
    """Loader implementation using Python's pickle serialization format."""

    def load(self, path: str | Path):
        """Loads a pickled object from the given file.

        Args:
            path (str | Path): Path to the pickle file.

        Returns:
            Any: Deserialized Python object.

        """
        with open(path, "rb") as f:
            val = pickle.load(f)
        return val

    def save(self, data, path: str | Path):
        """Saves a Python object as a pickle file.

        Args:
            data: Object to serialize.
            path (str | Path): Destination file path.

        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)


class FrameDetLoader(Loader):
    """Loader implementation for FrameDetections objects."""

    def load(self, path: str | Path):
        """Loads a FrameDetections object from the given file.

        Args:
            path (str | Path): Path to the serialized FrameDetections file.

        Returns:
            FrameDetections: Loaded detections for a single frame.

        """
        from exordium.video.core.detection import FrameDetections  # noqa: PLC0415

        return FrameDetections().load(path)

    def save(self, data, path: str | Path):
        """Saves a FrameDetections object to the given file.

        Args:
            data: FrameDetections instance to serialize.
            path (str | Path): Destination file path.

        """
        data.save(path)


class VideoDetLoader(Loader):
    """Loader implementation for VideoDetections objects."""

    def load(self, path: str | Path):
        """Loads a VideoDetections object from the given file.

        Args:
            path (str | Path): Path to the serialized VideoDetections file.

        Returns:
            VideoDetections: Loaded detections for a video.

        """
        from exordium.video.core.detection import VideoDetections  # noqa: PLC0415

        return VideoDetections().load(path)

    def save(self, data, path: str | Path):
        """Saves a VideoDetections object to the given file.

        Args:
            data: VideoDetections instance to serialize.
            path (str | Path): Destination file path.

        """
        data.save(path)


class TrackLoader(Loader):
    """Loader implementation for Track objects."""

    def load(self, path: str | Path):
        """Loads a Track object from the given file.

        Args:
            path (str | Path): Path to the serialized Track file.

        Returns:
            Track: Loaded track.

        """
        from exordium.video.core.detection import Track  # noqa: PLC0415

        return Track().load(path)

    def save(self, data, path: str | Path):
        """Saves a Track object to the given file.

        Args:
            data: Track instance to serialize.
            path (str | Path): Destination file path.

        """
        data.save(path)


class NpyLoader(Loader):
    """Loader implementation for NumPy .npy files."""

    def load(self, path: str | Path):
        """Loads a NumPy array from the given .npy file.

        Args:
            path (str | Path): Path to the .npy file.

        Returns:
            np.ndarray: Loaded array.

        """
        return np.load(path)

    def save(self, data, path: str | Path):
        """Saves a NumPy array to the given .npy file.

        Args:
            data: Array-like data to save.
            path (str | Path): Destination file path.

        """
        np.save(path, data)


class SafetensorsLoader(Loader):
    """Loader implementation using the safetensors format for torch tensors.

    Accepts either a single ``torch.Tensor`` or a ``dict[str, torch.Tensor]``:

    * **Single tensor** — stored under the key ``"data"`` and returned as a
      plain ``torch.Tensor`` on load, so the round-trip is transparent.
    * **Dict of tensors** — stored as-is and returned as
      ``dict[str, torch.Tensor]`` on load.

    Safetensors files are memory-mappable, contain no pickle data, and store
    only the raw float bytes plus a compact JSON header — making them the
    smallest and safest option for persisting feature tensors.

    """

    _KEY = "data"

    def load(self, path: str | Path) -> torch.Tensor | dict[str, torch.Tensor]:
        """Loads tensor(s) from a safetensors file.

        Returns a plain ``torch.Tensor`` if the file was saved from a single
        tensor, or a ``dict[str, torch.Tensor]`` if saved from a dict.

        Args:
            path (str | Path): Path to the ``.st`` file.

        Returns:
            torch.Tensor or dict[str, torch.Tensor].

        """
        from safetensors.torch import load_file

        tensors = load_file(str(path))
        if list(tensors.keys()) == [self._KEY]:
            return tensors[self._KEY]
        return tensors

    def save(self, data: torch.Tensor | dict[str, torch.Tensor], path: str | Path) -> None:
        """Saves tensor(s) to a safetensors file.

        Args:
            data: A single ``torch.Tensor`` or a ``dict[str, torch.Tensor]``.
                CUDA tensors are moved to CPU automatically before saving.
            path (str | Path): Destination file path.

        Raises:
            TypeError: If ``data`` is not a tensor or dict of tensors.

        """
        from safetensors.torch import save_file

        if isinstance(data, torch.Tensor):
            tensor_dict = {self._KEY: data.cpu()}
        elif isinstance(data, dict):
            tensor_dict = {k: v.cpu() for k, v in data.items()}
        else:
            raise TypeError(
                "SafetensorsLoader expects a torch.Tensor or dict[str, torch.Tensor], "
                f"got {type(data).__name__}"
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensor_dict, str(path))


class LoaderFactory:
    """Factory that maps file format strings to Loader instances.

    Supported formats: "fdet", "vdet", "track", "npy", "pkl", "st".
    """

    LOADERS: dict[str, type[Loader]] = {
        "fdet": FrameDetLoader,
        "vdet": VideoDetLoader,
        "track": TrackLoader,
        "npy": NpyLoader,
        "pkl": PickleLoader,
        "st": SafetensorsLoader,
    }
    """Registry mapping format keys to their :class:`Loader` implementations."""

    @classmethod
    def get(cls, format: str) -> Loader:
        """Returns a Loader instance for the specified file format.

        Args:
            format (str): File format key. One of "fdet", "vdet", "track",
                "npy", or "pkl".

        Returns:
            Loader: Loader instance for the given format.

        Raises:
            NotImplementedError: If format is not a recognised key.

        """
        loader_class = LoaderFactory.LOADERS.get(format)
        if loader_class is None:
            raise NotImplementedError(f"Format {format} is not supported.")
        return loader_class()


def load_or_create(format: str):
    """Decorator factory that caches function results to disk.

    If the wrapped function is called with an ``output_path`` keyword argument
    and the file already exists (and ``overwrite`` is False), the cached value
    is loaded and returned instead of calling the function.  If ``output_path``
    is None or the file does not exist, the function is executed and its result
    is saved.

    Args:
        format (str): File format key accepted by LoaderFactory (e.g. "st",
            "npy", "pkl", "fdet", "vdet", "track").

    Returns:
        Callable: A decorator that wraps a function with load-or-create
            caching behaviour.

    """
    loader = LoaderFactory.get(format)

    def decorator(function):
        def wrapper(*args, **kwargs):
            output_path: str | Path | None = kwargs.get("output_path", None)
            overwrite: bool = kwargs.get("overwrite", False)

            if (
                output_path is None
                or not Path(output_path).exists()
                or (Path(output_path).exists() and overwrite)
            ):
                val = function(*args, **kwargs)
                if output_path is not None:
                    logger.info(f"Save to {str(output_path)}...")
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    loader.save(val, output_path)
            else:
                logger.info(f"Load from {str(output_path)}...")
                val = loader.load(output_path)

            return val

        return wrapper

    return decorator
