"""Tests for exordium.utils.concurrent module."""

import unittest
from unittest.mock import MagicMock, patch

from exordium.utils.concurrent import processes_eval, threads_eval


class TestProcessesEval(unittest.TestCase):
    """Test processes_eval function."""

    @patch("multiprocessing.Pool")
    def test_processes_eval_with_multiprocessing(self, mock_pool_class):
        """Test processes_eval with multiprocessing enabled."""
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        mock_pool.starmap.return_value = iter([3, 7, 11])

        def add(a, b):
            return a + b

        data = [(1, 2), (3, 4), (5, 6)]
        results = processes_eval(add, data, use_mp=True)

        self.assertEqual(results, [3, 7, 11])

    def test_processes_eval_without_multiprocessing(self):
        """Test processes_eval without multiprocessing."""

        def multiply(a, b):
            return a * b

        data = [(2, 3), (4, 5), (6, 7)]
        results = processes_eval(multiply, data, use_mp=False)

        self.assertEqual(results, [6, 20, 42])

    def test_processes_eval_single_argument(self):
        """Test with functions that take single argument."""

        def square(x):
            return x**2

        data = [(2,), (3,), (4,), (5,)]
        results = processes_eval(square, data, use_mp=False)

        self.assertEqual(results, [4, 9, 16, 25])

    def test_processes_eval_multiple_arguments(self):
        """Test with functions that take multiple arguments."""

        def sum_three(a, b, c):
            return a + b + c

        data = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        results = processes_eval(sum_three, data, use_mp=False)

        self.assertEqual(results, [6, 15, 24])

    def test_processes_eval_empty_data(self):
        """Test with empty data list."""

        def dummy(x):
            return x

        data = []
        results = processes_eval(dummy, data, use_mp=False)

        self.assertEqual(results, [])

    def test_processes_eval_preserves_order(self):
        """Test that results preserve input order."""

        def identity(x):
            return x

        data = [(10,), (20,), (30,), (40,)]
        results = processes_eval(identity, data, use_mp=False)

        self.assertEqual(results, [10, 20, 30, 40])

    def test_processes_eval_with_complex_return(self):
        """Test with functions returning complex objects."""

        def create_dict(key, value):
            return {key: value}

        data = [("a", 1), ("b", 2), ("c", 3)]
        results = processes_eval(create_dict, data, use_mp=False)

        expected = [{"a": 1}, {"b": 2}, {"c": 3}]
        self.assertEqual(results, expected)

    @patch("multiprocessing.cpu_count", return_value=4)
    @patch("multiprocessing.Pool")
    def test_processes_eval_uses_all_cpus(self, mock_pool_class, mock_cpu_count):
        """Test that multiprocessing uses all available CPUs."""
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        mock_pool.starmap.return_value = iter([1, 2, 3])

        def add(a, b):
            return a + b

        data = [(1, 2), (3, 4), (5, 6)]
        processes_eval(add, data, use_mp=True)

        # Verify Pool was created with cpu_count
        mock_pool_class.assert_called_once_with(processes=4)


class TestThreadsEval(unittest.TestCase):
    """Test threads_eval function."""

    @patch("torch.cuda.device_count", return_value=2)
    @patch("threading.Thread")
    def test_threads_eval_creates_threads(self, mock_thread, mock_device_count):
        """Test that threads_eval creates threads for each GPU."""
        mock_thread_instances = [MagicMock(), MagicMock()]
        mock_thread.side_effect = mock_thread_instances

        def dummy_func(data, device):
            pass

        data = list(range(10))
        threads_eval(dummy_func, data)

        # Should create 2 threads (one per GPU)
        self.assertEqual(mock_thread.call_count, 2)

        # Verify all threads were started and joined
        for thread in mock_thread_instances:
            thread.start.assert_called_once()
            thread.join.assert_called_once()

    @patch("torch.cuda.device_count", return_value=2)
    @patch("threading.Thread")
    def test_threads_eval_splits_data(self, mock_thread, mock_device_count):
        """Test that data is split among threads."""
        thread_calls = []

        def capture_thread(*args, **kwargs):
            thread_calls.append((args, kwargs))
            mock = MagicMock()
            return mock

        mock_thread.side_effect = capture_thread

        def dummy_func(data, device):
            pass

        data = list(range(10))
        threads_eval(dummy_func, data)

        # Verify 2 threads were created
        self.assertEqual(len(thread_calls), 2)

        # Check that kwargs contains target function and args contains data splits
        for args, kwargs in thread_calls:
            # Thread is created with target=fcn, args=(split, device) + args
            self.assertIn("target", kwargs)
            self.assertEqual(kwargs["target"], dummy_func)
            # args should contain (data_split, device)
            self.assertEqual(len(kwargs.get("args", ())), 2)

    @patch("torch.cuda.device_count", return_value=1)
    @patch("threading.Thread")
    def test_threads_eval_single_gpu(self, mock_thread, mock_device_count):
        """Test threads_eval with single GPU."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        def dummy_func(data, device):
            pass

        data = list(range(5))
        threads_eval(dummy_func, data)

        # Should create 1 thread
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        mock_thread_instance.join.assert_called_once()

    @patch("torch.cuda.device_count", return_value=2)
    @patch("threading.Thread")
    def test_threads_eval_passes_extra_args(self, mock_thread, mock_device_count):
        """Test that extra args and kwargs are passed to function."""
        thread_calls = []

        def capture_thread(*args, **kwargs):
            thread_calls.append((args, kwargs))
            return MagicMock()

        mock_thread.side_effect = capture_thread

        def dummy_func(data, device, extra_arg, kwarg1=None):
            pass

        data = list(range(10))
        threads_eval(dummy_func, data, "extra", kwarg1="test")

        # Check that extra args were passed
        # Verify that 2 threads were created
        self.assertEqual(len(thread_calls), 2)
        for args, kwargs in thread_calls:
            # kwargs["args"] should contain (data_split, device, "extra")
            thread_args = kwargs.get("args", ())
            # Check that "extra" is in the tuple (last element after device)
            self.assertTrue(len(thread_args) >= 3)
            self.assertEqual(thread_args[2], "extra")
            self.assertEqual(kwargs.get("kwargs", {}).get("kwarg1"), "test")

    @patch("torch.cuda.device_count", return_value=0)
    def test_threads_eval_no_gpus(self, mock_device_count):
        """Test threads_eval when no GPUs are available."""
        # When device_count is 0, devices list will be empty
        # np.array_split with 0 splits will raise ValueError

        def dummy_func(data, device):
            pass

        data = list(range(10))

        # This should raise ValueError due to empty device list
        with self.assertRaises(ValueError):
            threads_eval(dummy_func, data)


if __name__ == "__main__":
    unittest.main()
