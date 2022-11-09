import os
import sys
import time
import pickle
import threading
from pathlib import Path
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Union, List, Callable, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm


def processes_eval(fcn: Callable[..., Any], 
                   data: List[Tuple[Union[str, int],...]], 
                   use_mp: bool = True) -> List[Tuple[Union[str, int, None],...]]:
    """Eval function using multiple processes

    Args:
        fcn (Callable[Union[str, int],...]): function
        data (List[Tuple[Union[str, int],...]]): list of function args
        use_mp (bool, optional): use multiprocessing. Defaults to True.
        verbose (Union[str, bool]): prints information. Defaults to False.

    Returns:
        List[Tuple[Union[str, int, None],...]]: list of return values
    """
    results = []
    if use_mp:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for result in tqdm(pool.starmap(fcn, data), total=len(data)):
                results.append(result)
            pool.close()
            pool.join()
    else:
        for d in tqdm(data, total=len(data)):
            results.append(fcn(*d))
    return results


def threads_eval(fcn: Callable[..., Any], 
                 data: List[Union[str, int]],
                 device_ids: str = 'all',
                 *args,
                 **kwargs):
    # get gpu devices
    if device_ids == 'all':
        devices = [f'cuda:{id}' for id in range(torch.cuda.device_count())]
    elif device_ids == 'cpu':
        devices = ['cpu']
    else:
        devices = [f'cuda:{int(id)}' for id in device_ids.split(',')]
    print(f'Available devices: {devices}.')

    splits = np.array_split(data, len(devices))
    threads = []
    for i in range(len(devices)):
        # batch_size
        # output_dir
        threads.append(threading.Thread(target=fcn, args=(devices[i], splits[i]) + args, kwargs=kwargs))
    for thread in threads: thread.start()
    for thread in threads: thread.join()


def reload_matplotlib() -> None:
    from importlib import reload
    import matplotlib
    reload(matplotlib)
    matplotlib.use('TkAgg')


def get_idle_gpus(thr: float = 0.1) -> List[int]:
    """Returns the available GPU ids

    Args:
        thr (float, optional): VRAM usage threshold. Defaults to 0.1.

    Returns:
        List[int]: list of available gpu ids
    """
    import nvidia_smi
    nvidia_smi.nvmlInit()
    gpu_ids = []
    for gpu_id in range(3):
        try:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            usage = info.used/info.total
            if info.used/info.total < thr:
                print(f'GPU:{gpu_id} is idle: {usage}')
                gpu_ids.append(gpu_id)
            else:
                print(f'GPU:{gpu_id} is busy: {usage}')    
        except:
            print(f'GPU:{gpu_id} is busy.')
    nvidia_smi.nvmlShutdown()
    if len(gpu_ids) == 0:
        print(f'There isn\'t any idle GPUs currently...')
        sys.exit(0)
    return gpu_ids


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
    def load(self, path):
        pass

    @abstractmethod
    def save(self, data, path):
        pass

class PickleLoader(Loader):

    def load(self, path):
        with open(path, 'rb') as f:
            val = pickle.load(f)
        return val
    
    def save(self, data, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

class DetLoader(Loader):

    def load(self, path):
        from exordium.video.retinaface import RetinafaceDetections
        return RetinafaceDetections().load(path)

    def save(self, data, path):
        data.save(path)

class NpyLoader(Loader):

    def load(self, path):
        return np.load(path)

    def save(self, data, path):
        np.save(path, data)

def load_or_create(format: str):
    if format == 'det':
        loader = DetLoader()
    elif format == 'npy':
        loader = NpyLoader()
    elif format == 'pkl':
        loader = PickleLoader()
    else:
        raise NotImplementedError()

    def decorator(function):
        def wrapper(*args, **kwargs):
            output_path = None if 'output_path' not in kwargs else kwargs['output_path']
            if output_path is not None and Path(output_path).exists() and \
                (('overwrite' not in kwargs) or ('overwrite' in kwargs and not kwargs['overwrite'])):
                print(f'Load from {output_path}...')
                val = loader.load(output_path)
            else:
                val = function(*args, **kwargs)
                if output_path is not None:
                    print(f'Save to {output_path}...')
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    loader.save(val, output_path)
            return val
        return wrapper
    return decorator

def get_project_root() -> Path:
    return Path(__file__).parents[2]


def get_weight_location() -> Path:
    # default weight location
    # possible future feature: configuration file and custom cache location
    return Path().home() / '.cache' / 'torch' / 'hub' / 'checkpoints'


if __name__ == '__main__':

    data = [
        'data/processed/frames/9KAqOrdiZ4I.001',
        'data/processed/frames/h-jMFLm6U_Y.000',
        'data/processed/frames/nEm44UpCKmA.002'
    ]

    def job(device_id, data_split, batch_size=32, output_dir='tmp/'):
        print(device_id, data_split, batch_size, output_dir)

    threads_eval(job, data, batch_size=64, device_ids='0', output_dir='tmp2/')