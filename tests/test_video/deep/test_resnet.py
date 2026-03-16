"""Tests for exordium.video.deep.resnet module (improved with ResNetWrapper tests)."""

import unittest

import numpy as np
import torch

from exordium.video.deep.resnet import (
    BasicBlock,
    Bottleneck,
    ResNetWrapper,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


class TestBasicBlock(unittest.TestCase):
    """Tests for BasicBlock."""

    def test_forward_no_downsample(self):
        block = BasicBlock(64, 64)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 28, 28))

    def test_forward_with_downsample(self):
        downsample = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(128),
        )
        block = BasicBlock(64, 128, stride=2, downsample=downsample)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 14, 14))

    def test_expansion_is_one(self):
        self.assertEqual(BasicBlock.expansion, 1)


class TestBottleneck(unittest.TestCase):
    """Tests for Bottleneck."""

    def test_forward_no_downsample(self):
        block = Bottleneck(256, 64)
        x = torch.randn(2, 256, 14, 14)
        out = block(x)
        self.assertEqual(out.shape, (2, 256, 14, 14))

    def test_forward_with_downsample(self):
        downsample = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(256),
        )
        block = Bottleneck(64, 64, downsample=downsample)
        x = torch.randn(2, 64, 14, 14)
        out = block(x)
        self.assertEqual(out.shape, (2, 256, 14, 14))

    def test_expansion_is_four(self):
        self.assertEqual(Bottleneck.expansion, 4)


class TestResNet(unittest.TestCase):
    """Tests for ResNet model."""

    @classmethod
    def setUpClass(cls):
        cls.model = resnet18(pretrained=False)
        cls.model.eval()

    def test_forward_output_shape(self):
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x)
        # ResNet returns spatial features: (B, C, H*W) then permuted to (B, H*W, C)
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[0], 2)

    def test_resnet18_no_pretrained(self):
        model = resnet18(pretrained=False)
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.ndim, 3)

    def test_resnet34_no_pretrained(self):
        model = resnet34(pretrained=False)
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.ndim, 3)

    def test_resnet50_no_pretrained(self):
        model = resnet50(pretrained=False)
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.ndim, 3)

    def test_resnet101_no_pretrained(self):
        model = resnet101(pretrained=False)
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.ndim, 3)

    def test_resnet152_no_pretrained(self):
        model = resnet152(pretrained=False)
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.ndim, 3)

    def test_make_layer_creates_sequential(self):
        model = resnet18(pretrained=False)
        self.assertIsInstance(model.layer1, torch.nn.Sequential)
        self.assertIsInstance(model.layer2, torch.nn.Sequential)

    def test_batch_size_two(self):
        model = resnet18(pretrained=False)
        x = torch.randn(2, 3, 56, 56)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape[0], 2)


class TestResNetWrapper(unittest.TestCase):
    """Tests for ResNetWrapper feature extractor."""

    def test_resnet18_predict_single_image(self):
        model = ResNetWrapper(arch="resnet18", pretrained=False, device_id=None)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = model.predict(img)
        self.assertEqual(result.shape, (1, 512))

    def test_resnet50_predict_batch(self):
        model = ResNetWrapper(arch="resnet50", pretrained=False, device_id=None)
        imgs = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
        result = model.predict(imgs)
        self.assertEqual(result.shape, (2, 2048))

    def test_invalid_arch_raises_value_error(self):
        with self.assertRaises(ValueError):
            ResNetWrapper(arch="resnet999", pretrained=False, device_id=None)

    def test_call_3d_tensor(self):
        model = ResNetWrapper(arch="resnet18", pretrained=False, device_id=None)
        tensor = torch.rand(3, 224, 224)
        result = model(tensor)
        self.assertEqual(result.shape, (1, 512))

    def test_call_4d_tensor(self):
        model = ResNetWrapper(arch="resnet18", pretrained=False, device_id=None)
        tensor = torch.rand(2, 3, 224, 224)
        result = model(tensor)
        self.assertEqual(result.shape, (2, 512))

    def test_result_dtype(self):
        model = ResNetWrapper(arch="resnet18", pretrained=False, device_id=None)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = model.predict(img)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
