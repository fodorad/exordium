"""Tests for exordium.utils.concurrent.processes_eval, threads_eval, _split_evenly."""

import threading
import unittest
from unittest.mock import patch

from exordium.utils.concurrent import _split_evenly, processes_eval


def _square(x):
    return x * x


def _add(a, b):
    return a + b


class TestProcessesEval(unittest.TestCase):
    def test_use_mp_false_simple_function(self):
        data = [(1,), (2,), (3,), (4,)]
        results = processes_eval(_square, data, use_mp=False)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 4)

    def test_use_mp_false_correct_values(self):
        data = [(i,) for i in range(5)]
        results = processes_eval(_square, data, use_mp=False)
        expected = [0, 1, 4, 9, 16]
        self.assertEqual(results, expected)

    def test_empty_data(self):
        results = processes_eval(_square, [], use_mp=False)
        self.assertEqual(results, [])


class TestProcessesEvalMultiprocess(unittest.TestCase):
    def test_with_multiprocessing(self):
        data = [(1, 2), (3, 4), (5, 6)]
        results = processes_eval(_add, data, use_mp=True, verbose=False)
        self.assertEqual(results, [3, 7, 11])

    def test_empty_data(self):
        results = processes_eval(_add, [], use_mp=False)
        self.assertEqual(results, [])


class TestThreadsEval(unittest.TestCase):
    def test_threads_eval_with_mocked_cuda(self):
        """threads_eval is exercised by mocking torch.cuda.device_count() to return 2."""
        from exordium.utils.concurrent import threads_eval

        results = []
        lock = threading.Lock()

        def fn(chunk, device):
            with lock:
                results.append((chunk, device))

        data = [1, 2, 3, 4]
        with patch("torch.cuda.device_count", return_value=2):
            threads_eval(fn, data)

        self.assertEqual(len(results), 2)
        all_items = [item for chunk, _ in results for item in chunk]
        self.assertEqual(sorted(all_items), [1, 2, 3, 4])

    def test_threads_eval_passes_extra_args(self):
        """threads_eval passes *args and **kwargs to the function."""
        from exordium.utils.concurrent import threads_eval

        received_kwargs = []
        lock = threading.Lock()

        def fn(chunk, device, multiplier=1):
            with lock:
                received_kwargs.append(multiplier)

        with patch("torch.cuda.device_count", return_value=1):
            threads_eval(fn, [1, 2, 3], multiplier=5)

        self.assertEqual(received_kwargs, [5])


class TestSplitEvenly(unittest.TestCase):
    def test_even_split(self):
        data = list(range(6))
        chunks = _split_evenly(data, 3)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 2)

    def test_uneven_split(self):
        data = list(range(7))
        chunks = _split_evenly(data, 3)
        self.assertEqual(len(chunks), 3)
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 7)

    def test_single_chunk(self):
        data = list(range(5))
        chunks = _split_evenly(data, 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(list(chunks[0]), data)


if __name__ == "__main__":
    unittest.main()
