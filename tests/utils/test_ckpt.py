"""Tests for exordium.utils.ckpt: remove_token, load_checkpoint, download_file, download_weight."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from exordium.utils.ckpt import download_file, load_checkpoint, remove_token


class TestRemoveToken(unittest.TestCase):
    def test_strips_prefix_from_keys(self):
        weights = {"_model.layer.weight": torch.zeros(4), "_model.layer.bias": torch.zeros(4)}
        cleaned = remove_token(weights, token="_model.")
        self.assertIn("layer.weight", cleaned)
        self.assertIn("layer.bias", cleaned)
        self.assertNotIn("_model.layer.weight", cleaned)

    def test_keys_without_token_unchanged(self):
        weights = {"other_key": torch.zeros(4), "_model.key": torch.zeros(4)}
        cleaned = remove_token(weights, token="_model.")
        self.assertIn("other_key", cleaned)

    def test_empty_dict(self):
        result = remove_token({})
        self.assertEqual(result, {})

    def test_custom_token(self):
        weights = {"module.conv.weight": torch.zeros(4), "module.bn.bias": torch.zeros(4)}
        cleaned = remove_token(weights, token="module.")
        self.assertIn("conv.weight", cleaned)
        self.assertIn("bn.bias", cleaned)

    def test_strips_model_prefix(self):
        weights = {"_model.layer.weight": torch.zeros(3), "other.bias": torch.zeros(2)}
        cleaned = remove_token(weights, token="_model.")
        self.assertIn("layer.weight", cleaned)
        self.assertNotIn("_model.layer.weight", cleaned)

    def test_non_matching_keys_unchanged(self):
        weights = {"layer.weight": torch.zeros(3)}
        cleaned = remove_token(weights, token="_model.")
        self.assertIn("layer.weight", cleaned)


class TestLoadCheckpoint(unittest.TestCase):
    def test_strips_module_prefix(self):
        state = {"module.fc.weight": torch.zeros(4, 4), "module.fc.bias": torch.zeros(4)}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        try:
            torch.save(state, path)
            loaded = load_checkpoint(path, strip_prefix="module.")
            self.assertIn("fc.weight", loaded)
            self.assertIn("fc.bias", loaded)
            self.assertNotIn("module.fc.weight", loaded)
        finally:
            path.unlink(missing_ok=True)

    def test_handles_state_dict_key(self):
        state = {"fc.weight": torch.zeros(4, 4)}
        ckpt = {"state_dict": state}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        try:
            torch.save(ckpt, path)
            loaded = load_checkpoint(path, strip_prefix="module.")
            self.assertIn("fc.weight", loaded)
        finally:
            path.unlink(missing_ok=True)

    def test_keys_without_prefix_unchanged(self):
        state = {"fc.weight": torch.zeros(4, 4), "fc.bias": torch.zeros(4)}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        try:
            torch.save(state, path)
            loaded = load_checkpoint(path, strip_prefix="module.")
            self.assertIn("fc.weight", loaded)
            self.assertIn("fc.bias", loaded)
        finally:
            path.unlink(missing_ok=True)

    def test_strips_module_prefix_from_state_dict(self):
        state = {"module.layer.weight": torch.zeros(3), "module.bias": torch.zeros(2)}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            p = Path(f.name)
        try:
            torch.save({"state_dict": state}, p)
            loaded = load_checkpoint(p)
            self.assertIn("layer.weight", loaded)
            self.assertNotIn("module.layer.weight", loaded)
        finally:
            p.unlink(missing_ok=True)


class TestDownloadFile(unittest.TestCase):
    def test_skips_download_when_file_exists(self):
        """When the file already exists, urlretrieve should NOT be called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "existing.bin"
            p.write_bytes(b"data")
            with patch("urllib.request.urlretrieve") as mock_retrieve:
                download_file("http://example.com/file.bin", p)
                mock_retrieve.assert_not_called()

    def test_downloads_when_file_missing(self):
        """When the file is absent, urlretrieve is called and creates the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "new_file.bin"

            def fake_retrieve(url, dest):
                Path(dest).write_bytes(b"downloaded")

            with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
                download_file("http://example.com/file.bin", p)
            self.assertTrue(p.exists())

    def test_overwrite_forces_redownload(self):
        """overwrite=True should re-download even if file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "existing.bin"
            p.write_bytes(b"old")
            call_count = [0]

            def fake_retrieve(url, dest):
                call_count[0] += 1
                Path(dest).write_bytes(b"new")

            with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
                download_file("http://example.com/file.bin", p, overwrite=True)
            self.assertEqual(call_count[0], 1)
            self.assertEqual(p.read_bytes(), b"new")

    def test_raises_file_not_found_when_download_fails(self):
        """If urlretrieve is called but the file is still missing, raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "fail.bin"
            with patch("urllib.request.urlretrieve"):
                with self.assertRaises(FileNotFoundError):
                    download_file("http://example.com/fail.bin", p)

    def test_creates_parent_directories(self):
        """download_file creates parent dirs as needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "deep" / "nested" / "file.bin"

            def fake_retrieve(url, dest):
                Path(dest).write_bytes(b"ok")

            with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
                download_file("http://example.com/file.bin", p)
            self.assertTrue(p.exists())


class TestDownloadWeight(unittest.TestCase):
    def test_downloads_when_file_missing(self):
        """When local file doesn't exist → lines 68-70: mkdir, log, hf_hub_download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "weights"
            filename = "test_weight.pth"
            local_path = local_dir / filename

            def fake_hf_download(repo_id, filename, local_dir):
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                (Path(local_dir) / filename).write_bytes(b"fake")

            with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_download):
                from exordium.utils.ckpt import download_weight

                result = download_weight(filename, local_dir=local_dir, repo_id="fake/repo")

            self.assertEqual(result, local_path)
            self.assertTrue(result.exists())

    def test_raises_when_download_fails(self):
        """File missing after hf_hub_download → line 73: raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "weights"
            filename = "missing.pth"

            def fake_hf_download(repo_id, filename, local_dir):
                Path(local_dir).mkdir(parents=True, exist_ok=True)

            with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_download):
                from exordium.utils.ckpt import download_weight

                with self.assertRaises(FileNotFoundError):
                    download_weight(filename, local_dir=local_dir, repo_id="fake/repo")

    def test_skips_download_when_file_exists(self):
        """File already exists → lines 68-70 NOT executed, returns existing path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "weights"
            local_dir.mkdir(parents=True)
            filename = "existing.pth"
            (local_dir / filename).write_bytes(b"data")

            with patch("huggingface_hub.hf_hub_download") as mock_dl:
                from exordium.utils.ckpt import download_weight

                result = download_weight(filename, local_dir=local_dir, repo_id="fake/repo")
                mock_dl.assert_not_called()

            self.assertEqual(result, local_dir / filename)


if __name__ == "__main__":
    unittest.main()
