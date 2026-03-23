"""Concurrent execution utilities."""

import multiprocessing
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from tqdm import tqdm


def _split_evenly(data: Sequence, n: int) -> list:
    """Split a sequence into n roughly equal chunks."""
    k, m = divmod(len(data), n)
    return [data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def processes_eval(
    fcn: Callable[..., Any],
    data: Iterable[Iterable[Any]],
    use_mp: bool = True,
    verbose: bool = False,
) -> list:
    """Eval function using multiple processes.

    Args:
        fcn (Callable[..., Any]): function.
        data (Iterable[Iterable[Any]]): list of function args.
        use_mp (bool, optional): use multiprocessing. Defaults to True.
        verbose (bool, optional): show progress bar. Defaults to False.

    Returns:
        list: return values of fcn.

    """
    total = len(list(data))
    results = []
    if use_mp:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for result in tqdm(
                pool.starmap(fcn, data), total=total, desc="Processing", disable=not verbose
            ):
                results.append(result)
            pool.close()
            pool.join()
    else:
        for d in tqdm(data, total=total, desc="Processing", disable=not verbose):
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
    splits = _split_evenly(data, len(devices))

    threads = []
    for i in range(len(devices)):
        threads.append(
            threading.Thread(target=fcn, args=(splits[i], devices[i]) + args, kwargs=kwargs)
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
