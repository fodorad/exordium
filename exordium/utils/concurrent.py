"""Concurrent execution utilities."""

import multiprocessing
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def processes_eval(
    fcn: Callable[..., Any], data: Iterable[Iterable[Any]], use_mp: bool = True
) -> list:
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
        fcn: Function to evaluate.
        data: Data to be split between threads.
        *args: Additional positional arguments passed to fcn.
        **kwargs: Additional keyword arguments passed to fcn.

    """
    devices = [f"cuda:{id}" for id in range(torch.cuda.device_count())]
    splits = np.array_split(data, len(devices))

    threads = []
    for i in range(len(devices)):
        threads.append(
            threading.Thread(target=fcn, args=(splits[i], devices[i]) + args, kwargs=kwargs)
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
