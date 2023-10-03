import threading
import multiprocessing
from typing import Iterable, Sequence, Callable, Any
import numpy as np
import torch
from tqdm import tqdm


def processes_eval(fcn: Callable[..., Any], data: Iterable[Iterable[Any]], use_mp: bool = True) -> list:
    """Eval function using multiple processes.

    Args:
        fcn (Callable[..., Any]): function.
        data (Iterable[Iterable[Any]]): list of function args.
        use_mp (bool, optional): use multiprocessing. Defaults to True.

    Returns:
        list: return values of fcn.
    """
    total = len(list(data))
    results = []
    if use_mp:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for result in tqdm(pool.starmap(fcn, data), total=total):
                results.append(result)
            pool.close()
            pool.join()
    else:
        for d in tqdm(data, total=total):
            results.append(fcn(*d))
    return results


def threads_eval(fcn: Callable[..., Any], data: Sequence[Any], *args, **kwargs) -> None:
    """Eval function using multiple threads.

    Args:
        fcn (Callable[..., Any]): function.
        data (Sequence[Any]): data to be split between threads.
    """
    devices = [f'cuda:{id}' for id in range(torch.cuda.device_count())]
    splits = np.array_split(data, len(devices))

    threads = []
    for i in range(len(devices)):
        threads.append(threading.Thread(target=fcn, args=(splits[i], devices[i]) + args, kwargs=kwargs))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def get_idle_gpus(thr: float = 0.1) -> list[int]:
    """Returns the available GPU ids.

    Args:
        thr (float, optional): VRAM usage threshold. Defaults to 0.1.

    Returns:
        list[int]: list of available gpu ids.
    """
    import nvidia_smi
    nvidia_smi.nvmlInit()

    num_gpus = torch.cuda.device_count()
    gpu_ids_idle, gpu_ids_busy, gpu_ids_down = [], [], []
    for gpu_id in range(num_gpus):

        try:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            usage = np.round(info.used/info.total, decimals=2)

            if usage < thr:
                print(f'GPU:{gpu_id} is idle: {usage}')
                gpu_ids_idle.append(gpu_id)
            else:
                print(f'GPU:{gpu_id} is busy: {usage}')
                gpu_ids_busy.append(gpu_id)
        except:
            print(f'GPU:{gpu_id} is down.')
            gpu_ids_down.append(gpu_id)

    nvidia_smi.nvmlShutdown()

    print(f'Number of GPUs: {num_gpus}')
    print(f'Idle: {",".join(map(str, gpu_ids_idle))}')
    print(f'Busy: {",".join(map(str, gpu_ids_busy))}')
    print(f'Down: {",".join(map(str, gpu_ids_down))}')

    if len(gpu_ids_idle) > 0:
        raise Exception('No idle gpus.')

    return gpu_ids_idle


if __name__ == '__main__':

    def job(device, data_split, batch_size, verbose: bool = False):
        print(data_split, batch_size, verbose)

    data = [f'{i}th path' for i in range(100)]
    threads_eval(job, data, 16, verbose=True)