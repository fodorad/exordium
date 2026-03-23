"""Coverage tests for bb.py: visualize_bb with output_path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.core.bb import visualize_bb


class TestVisualizeBbOutputPath(unittest.TestCase):
    def test_numpy_image_with_output_path_saves_file(self):
        """output_path is not None → lines 448-449: parent.mkdir + imwrite."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bb = np.array([10, 10, 80, 80], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sub" / "bb_vis.jpg"
            result = visualize_bb(img, bb, probability=0.9, output_path=out)
            self.assertTrue(out.exists())
        self.assertIsInstance(result, np.ndarray)

    def test_tensor_image_with_output_path(self):
        """Tensor (C, H, W) + output_path → convert and save, return tensor."""
        img_t = torch.zeros(3, 100, 100, dtype=torch.uint8)
        bb = np.array([10, 10, 80, 80], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "bb_tensor.jpg"
            result = visualize_bb(img_t, bb, probability=0.75, output_path=out)
            self.assertTrue(out.exists())
        self.assertIsInstance(result, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
