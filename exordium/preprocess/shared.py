import sys
import time
import multiprocessing as mp
from typing import Union, List, Callable, Any, Tuple

from tqdm import tqdm


def parallel_eval(fcn: Callable[..., Any], 
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
        print('Function took:', time.time()-before, 'seconds.')
    return wrapper


def timer_with_return(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        val = func(*args, **kwargs)
        print('Function took:', time.time()-before, 'seconds.')
        return val
    return wrapper