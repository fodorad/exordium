"""Tests for exordium.video.deep.swin module (renamed + improved with SwinWrapper tests)."""

import unittest

import numpy as np
import torch

from exordium.video.deep.swin import (
    BasicLayer,
    PatchEmbed,
    PatchMerging,
    SwinTransformer,
    SwinTransformerBlock,
    SwinWrapper,
    WindowAttention,
    swin_transformer_base,
    swin_transformer_small,
    swin_transformer_tiny,
)


class TestWindowAttention(unittest.TestCase):
    """Tests for WindowAttention."""

    def setUp(self):
        self.wa = WindowAttention(dim=96, window_size=(7, 7), num_heads=3)

    def test_extra_repr(self):
        s = self.wa.extra_repr()
        self.assertIn("96", s)
        self.assertIn("num_heads=3", s)

    def test_flops(self):
        n = 49  # 7*7
        flops = self.wa.flops(n)
        self.assertIsInstance(flops, (int, float))
        self.assertGreater(flops, 0)

    def test_forward(self):
        x = torch.randn(8, 49, 96)  # (B*num_windows, window_size^2, dim)
        out = self.wa(x)
        self.assertEqual(out.shape, x.shape)


class TestSwinTransformerBlock(unittest.TestCase):
    """Tests for SwinTransformerBlock."""

    def setUp(self):
        self.stb = SwinTransformerBlock(
            dim=96, input_resolution=(56, 56), num_heads=3, window_size=7, shift_size=0
        )

    def test_extra_repr(self):
        s = self.stb.extra_repr()
        self.assertIn("96", s)
        self.assertIn("num_heads=3", s)

    def test_flops(self):
        flops = self.stb.flops()
        self.assertIsInstance(flops, float)
        self.assertGreater(flops, 0)


class TestPatchMerging(unittest.TestCase):
    """Tests for PatchMerging."""

    def setUp(self):
        self.pm = PatchMerging(input_resolution=(56, 56), dim=96)

    def test_extra_repr(self):
        s = self.pm.extra_repr()
        self.assertIn("96", s)

    def test_flops(self):
        flops = self.pm.flops()
        self.assertIsInstance(flops, (int, float))
        self.assertGreater(flops, 0)

    def test_forward(self):
        x = torch.randn(1, 56 * 56, 96)
        out = self.pm(x)
        self.assertEqual(out.shape, (1, 56 * 56 // 4, 96 * 2))


class TestBasicLayer(unittest.TestCase):
    """Tests for BasicLayer."""

    def setUp(self):
        self.bl = BasicLayer(dim=96, input_resolution=(56, 56), depth=2, num_heads=3, window_size=7)

    def test_extra_repr(self):
        s = self.bl.extra_repr()
        self.assertIn("96", s)

    def test_flops(self):
        flops = self.bl.flops()
        self.assertIsInstance(flops, float)
        self.assertGreater(flops, 0)

    def test_forward(self):
        x = torch.randn(1, 56 * 56, 96)
        out = self.bl(x)
        self.assertEqual(out.shape, x.shape)


class TestPatchEmbed(unittest.TestCase):
    """Tests for PatchEmbed."""

    def setUp(self):
        self.pe = PatchEmbed()

    def test_forward(self):
        x = torch.randn(1, 3, 224, 224)
        out = self.pe(x)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[-1], 96)

    def test_flops(self):
        flops = self.pe.flops()
        self.assertIsInstance(flops, (int, float))
        self.assertGreater(flops, 0)


class TestBasicLayerWithCheckpoint(unittest.TestCase):
    """Tests for BasicLayer with use_checkpoint=True."""

    def test_forward_with_checkpoint(self):
        bl = BasicLayer(
            dim=96,
            input_resolution=(56, 56),
            depth=2,
            num_heads=3,
            window_size=7,
            use_checkpoint=True,
        )
        x = torch.randn(1, 56 * 56, 96)
        out = bl(x)
        self.assertEqual(out.shape, x.shape)


class TestSwinTransformerWithApe(unittest.TestCase):
    """Tests for SwinTransformer with absolute position embedding."""

    def test_forward_with_ape(self):
        model = SwinTransformer(
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, ape=True
        )
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape[0], 1)


class TestSwinTransformerTiny(unittest.TestCase):
    """Tests for swin_transformer_tiny factory."""

    @classmethod
    def setUpClass(cls):
        cls.model = swin_transformer_tiny(pretrained=False)

    def test_creates_model(self):
        self.assertIsNotNone(self.model)

    def test_no_weight_decay(self):
        params = self.model.no_weight_decay()
        self.assertIsInstance(params, set)
        self.assertIn("absolute_pos_embed", params)

    def test_no_weight_decay_keywords(self):
        keywords = self.model.no_weight_decay_keywords()
        self.assertIsInstance(keywords, set)
        self.assertIn("relative_position_bias_table", keywords)

    def test_flops(self):
        flops = self.model.flops()
        self.assertIsInstance(flops, (int, float))
        self.assertGreater(flops, 0)

    def test_forward(self):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape[0], 1)


class TestSwinTransformerSmall(unittest.TestCase):
    """Tests for swin_transformer_small factory."""

    def test_creates_model(self):
        model = swin_transformer_small(pretrained=False)
        self.assertIsNotNone(model)

    def test_forward(self):
        model = swin_transformer_small(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape[0], 1)


class TestSwinTransformerBase(unittest.TestCase):
    """Tests for swin_transformer_base factory."""

    def test_creates_model(self):
        model = swin_transformer_base(pretrained=False)
        self.assertIsNotNone(model)

    def test_forward(self):
        model = swin_transformer_base(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape[0], 1)


class TestSwinWrapper(unittest.TestCase):
    """Tests for SwinWrapper feature extractor."""

    def test_tiny_predict_single_image(self):
        model = SwinWrapper(arch="tiny", pretrained=False, device_id=None)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = model.predict(img)
        self.assertEqual(result.shape, (1, 768))

    def test_invalid_arch_raises_value_error(self):
        with self.assertRaises(ValueError):
            SwinWrapper(arch="huge", pretrained=False, device_id=None)

    def test_result_dtype(self):
        model = SwinWrapper(arch="tiny", pretrained=False, device_id=None)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = model.predict(img)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
