"""Tests for exordium.utils.ckpt module."""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exordium.utils.ckpt import download_file, get_logger, remove_token


class TestGetLogger(unittest.TestCase):
    """Test get_logger function."""

    def test_creates_logger_with_name(self):
        """Test that logger is created with correct name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = get_logger("test_logger", log_path)

            self.assertEqual(logger.name, "test_logger")
            self.assertEqual(logger.level, logging.DEBUG)

    def test_creates_log_file(self):
        """Test that log file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = get_logger("test_logger", log_path)

            # Write a log message
            logger.debug("Test message")

            # Check file exists and contains message
            self.assertTrue(log_path.exists())
            with open(log_path) as f:
                content = f.read()
            self.assertIn("Test message", content)
            self.assertIn("DEBUG", content)

    def test_logger_format(self):
        """Test that logger uses correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = get_logger("test_logger", log_path)

            logger.info("Info message")

            with open(log_path) as f:
                content = f.read()

            # Check format: "timestamp - level - message"
            self.assertIn(" - INFO - Info message", content)

    def test_logger_accepts_string_path(self):
        """Test that logger accepts string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "test.log")
            logger = get_logger("test_logger", log_path)

            logger.warning("Warning message")

            self.assertTrue(Path(log_path).exists())

    def test_logger_accepts_path_object(self):
        """Test that logger accepts Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = get_logger("test_logger", log_path)

            logger.error("Error message")

            self.assertTrue(log_path.exists())

    def test_multiple_loggers(self):
        """Test creating multiple loggers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path1 = Path(tmpdir) / "log1.log"
            log_path2 = Path(tmpdir) / "log2.log"

            logger1 = get_logger("logger1", log_path1)
            logger2 = get_logger("logger2", log_path2)

            logger1.info("Message 1")
            logger2.info("Message 2")

            with open(log_path1) as f:
                content1 = f.read()
            with open(log_path2) as f:
                content2 = f.read()

            self.assertIn("Message 1", content1)
            self.assertIn("Message 2", content2)


class TestDownloadFile(unittest.TestCase):
    """Test download_file function."""

    @patch("urllib.request.urlretrieve")
    def test_downloads_file(self, mock_retrieve):
        """Test that file is downloaded using urllib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "downloaded.txt"
            remote_path = "http://example.com/file.txt"

            # Simulate download creating the file
            local_path.touch()
            mock_retrieve.return_value = (str(local_path), {})

            download_file(remote_path, local_path, overwrite=True)

            mock_retrieve.assert_called_once_with(remote_path, str(local_path))

    @patch("urllib.request.urlretrieve")
    def test_creates_parent_directory(self, mock_retrieve):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "subdir" / "downloaded.txt"
            remote_path = "http://example.com/file.txt"

            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.touch()
            mock_retrieve.return_value = (str(local_path), {})

            download_file(remote_path, local_path)

            self.assertTrue(local_path.parent.exists())

    @patch("urllib.request.urlretrieve")
    def test_skips_download_if_exists(self, mock_retrieve):
        """Test that download is skipped if file exists and overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "existing.txt"
            local_path.touch()
            remote_path = "http://example.com/file.txt"

            download_file(remote_path, local_path, overwrite=False)

            mock_retrieve.assert_not_called()

    @patch("urllib.request.urlretrieve")
    def test_overwrites_if_overwrite_true(self, mock_retrieve):
        """Test that file is re-downloaded if overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "existing.txt"
            local_path.touch()
            remote_path = "http://example.com/file.txt"
            mock_retrieve.return_value = (str(local_path), {})

            download_file(remote_path, local_path, overwrite=True)

            mock_retrieve.assert_called_once()

    @patch("urllib.request.urlretrieve")
    def test_raises_error_if_download_fails(self, mock_retrieve):
        """Test that FileNotFoundError is raised if download fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "failed.txt"
            remote_path = "http://example.com/file.txt"

            # Simulate urlretrieve not actually creating the file
            mock_retrieve.return_value = (str(local_path), {})

            with self.assertRaises(FileNotFoundError):
                download_file(remote_path, local_path)

    @patch("urllib.request.urlretrieve")
    def test_uses_correct_urlretrieve_args(self, mock_retrieve):
        """Test that urlretrieve is called with correct arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "file.txt"
            remote_path = "http://example.com/file.txt"

            local_path.touch()
            mock_retrieve.return_value = (str(local_path), {})

            download_file(remote_path, local_path, overwrite=True)

            mock_retrieve.assert_called_once_with(remote_path, str(local_path))


class TestRemoveToken(unittest.TestCase):
    """Test remove_token function."""

    def test_removes_default_token(self):
        """Test removing default '_model.' token."""
        weights = {
            "_model.layer1.weight": "value1",
            "_model.layer2.bias": "value2",
            "_model.layer3.weight": "value3",
        }

        result = remove_token(weights)

        expected = {
            "layer1.weight": "value1",
            "layer2.bias": "value2",
            "layer3.weight": "value3",
        }
        self.assertEqual(result, expected)

    def test_removes_custom_token(self):
        """Test removing custom token."""
        weights = {
            "encoder.layer1.weight": "value1",
            "encoder.layer2.bias": "value2",
        }

        result = remove_token(weights, token="encoder.")

        expected = {
            "layer1.weight": "value1",
            "layer2.bias": "value2",
        }
        self.assertEqual(result, expected)

    def test_handles_no_token_match(self):
        """Test when token doesn't exist in keys."""
        weights = {
            "layer1.weight": "value1",
            "layer2.bias": "value2",
        }

        result = remove_token(weights, token="_model.")

        # Should return unchanged
        self.assertEqual(result, weights)

    def test_handles_partial_matches(self):
        """Test with partial token matches."""
        weights = {
            "_model.layer1.weight": "value1",
            "layer2.bias": "value2",
            "_model.layer3.weight": "value3",
        }

        result = remove_token(weights, token="_model.")

        expected = {
            "layer1.weight": "value1",
            "layer2.bias": "value2",
            "layer3.weight": "value3",
        }
        self.assertEqual(result, expected)

    def test_preserves_values(self):
        """Test that values are preserved correctly."""
        import torch

        weights = {
            "_model.layer1": torch.tensor([1, 2, 3]),
            "_model.layer2": torch.tensor([4, 5, 6]),
        }

        result = remove_token(weights)

        self.assertTrue(torch.equal(result["layer1"], torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(result["layer2"], torch.tensor([4, 5, 6])))

    def test_empty_weights(self):
        """Test with empty weights dictionary."""
        weights = {}
        result = remove_token(weights)
        self.assertEqual(result, {})

    def test_multiple_token_occurrences(self):
        """Test removing token that appears multiple times in a key."""
        weights = {
            "_model._model.layer1": "value1",
        }

        result = remove_token(weights, token="_model.")

        # Should remove all occurrences
        expected = {"layer1": "value1"}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
