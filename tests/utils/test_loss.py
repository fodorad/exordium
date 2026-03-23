"""Tests for exordium.utils.loss: bell_l2_l1_loss, ecl1."""

import unittest

import torch

from exordium.utils.loss import bell_l2_l1_loss, ecl1


class TestBellL2L1Loss(unittest.TestCase):
    def test_identical_inputs_near_zero(self):
        y = torch.zeros(4, 5)
        loss = bell_l2_l1_loss(y, y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_wrong_predictions_higher_loss(self):
        y_true = torch.zeros(4, 5)
        y_pred = torch.ones(4, 5) * 5.0
        loss = bell_l2_l1_loss(y_pred, y_true)
        self.assertGreater(loss.item(), 0.0)

    def test_scalar_output(self):
        y_pred = torch.randn(8, 5)
        y_true = torch.randn(8, 5)
        loss = bell_l2_l1_loss(y_pred, y_true)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_nonnegative(self):
        y_pred = torch.randn(4, 3)
        y_true = torch.randn(4, 3)
        loss = bell_l2_l1_loss(y_pred, y_true)
        self.assertGreaterEqual(loss.item(), 0.0)


class TestEcl1(unittest.TestCase):
    def test_identical_inputs_near_zero(self):
        y = torch.zeros(4, 5)
        loss = ecl1(y, y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_wrong_predictions_higher_loss(self):
        y_true = torch.zeros(4, 5)
        y_pred = torch.ones(4, 5) * 3.0
        loss = ecl1(y_pred, y_true)
        self.assertGreater(loss.item(), 0.0)

    def test_scalar_output(self):
        y_pred = torch.randn(8, 5)
        y_true = torch.randn(8, 5)
        loss = ecl1(y_pred, y_true)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_nonnegative(self):
        y_pred = torch.randn(4, 3)
        y_true = torch.randn(4, 3)
        loss = ecl1(y_pred, y_true)
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
