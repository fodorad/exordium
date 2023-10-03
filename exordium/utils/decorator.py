import logging
import time
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Type
import numpy as np
from exordium import PathType
from exordium.video.detection import FrameDetections, VideoDetections, Track


def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        func(*args, **kwargs)
        print('Function took:', np.round(time.time()-before, decimals=3), 'seconds.')
    return wrapper


def timer_with_return(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        val = func(*args, **kwargs)
        print('Function took:', np.round(time.time()-before, decimals=3), 'seconds.')
        return val
    return wrapper


class Loader(ABC):

    @abstractmethod
    def load(self, path: PathType):
        pass

    @abstractmethod
    def save(self, data, path: PathType):
        pass


class PickleLoader(Loader):

    def load(self, path: PathType):
        with open(path, 'rb') as f:
            val = pickle.load(f)
        return val

    def save(self, data, path: PathType):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)


class FrameDetLoader(Loader):

    def load(self, path: PathType):
        return FrameDetections().load(path)

    def save(self, data: FrameDetections, path: PathType):
        data.save(path)


class VideoDetLoader(Loader):

    def load(self, path: PathType):
        return VideoDetections().load(path)

    def save(self, data: VideoDetections, path: PathType):
        data.save(path)


class TrackLoader(Loader):

    def load(self, path: PathType):
        return Track().load(path)

    def save(self, data: Track, path: PathType):
        data.save(path)


class NpyLoader(Loader):

    def load(self, path: PathType):
        return np.load(path)

    def save(self, data, path: PathType):
        np.save(path, data)


class LoaderFactory:

    LOADERS: dict[str, Type[Loader]] = {
        'fdet': FrameDetLoader,
        'vdet': VideoDetLoader,
        'track': TrackLoader,
        'npy': NpyLoader,
        'pkl': PickleLoader,
    }

    @classmethod
    def get(cls, format: str) -> Loader:
        loader_class = LoaderFactory.LOADERS.get(format)
        if loader_class is None:
            raise NotImplementedError(f'Format {format} is not supported.')
        return loader_class()


def load_or_create(format: str):
    loader = LoaderFactory.get(format)

    def decorator(function):
        def wrapper(*args, **kwargs):
            output_path: PathType | None = kwargs.get('output_path', None)
            overwrite: bool = kwargs.get('overwrite', False)

            if output_path is None or not Path(output_path).exists() or (Path(output_path).exists() and overwrite):
                val = function(*args, **kwargs)
                if output_path is not None:
                    logging.info(f'Save to {str(output_path)}...')
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    loader.save(val, output_path)
            else:
                logging.info(f'Load from {str(output_path)}...')
                val = loader.load(output_path)

            return val
        return wrapper
    return decorator