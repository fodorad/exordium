import unittest

import torch

from exordium.utils.loss import bell_l2_l1_loss, ecl1


class LossTestCase(unittest.TestCase):
    def test_bell_l2_l1_loss(self):
        """Test bell_l2_l1_loss function."""
        y_pred = torch.randn(10, 5)
        y_true = torch.randn(10, 5)
        loss = bell_l2_l1_loss(y_pred, y_true)
        self.assertEqual(loss.shape, ())
        self.assertIsInstance(loss.item(), float)

    def test_ecl1(self):
        """Test ecl1 (Error Consistent L1) loss function."""
        loss = ecl1(torch.zeros(10, 5), torch.zeros(10, 5))
        self.assertEqual(loss.shape, ())


if __name__ == "__main__":
    unittest.main()
