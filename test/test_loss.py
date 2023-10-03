import unittest
import torch
import numpy as np
from exordium.utils.loss import InvKDEWeightedLoss, ecl1


class LossTestCase(unittest.TestCase):

    def setUp(self):
        self.gt = np.stack([
            np.random.normal(0.3, 0.15, 6000).clip(0,1),
            np.random.normal(0.5, 0.12, 6000).clip(0,1),
            np.random.normal(0.4, 0.13, 6000).clip(0,1),
            np.random.normal(0.7, 0.11, 6000).clip(0,1),
            np.random.normal(0.6, 0.12, 6000).clip(0,1)
        ], axis=1) # (6000,5)

    def test_inv_kde_bell(self):
        loss_fcn = InvKDEWeightedLoss(self.gt).bell()
        loss = loss_fcn(torch.zeros(10, 5), torch.zeros(10, 5))
        self.assertEqual(loss.shape, ())

    def test_ecl1(self):
        loss = ecl1(torch.zeros(10, 5), torch.zeros(10, 5))
        self.assertEqual(loss.shape, ())


if __name__ == '__main__':
    unittest.main()