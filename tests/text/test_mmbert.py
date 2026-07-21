"""Tests for the mmBERT multilingual text encoder wrapper.

Exercises the shared token/pooled ``pooling`` contract against the real
:class:`~exordium.text.mmbert.MmbertWrapper` (with random weights when
``PRETRAINED`` is ``False``, so no checkpoint is downloaded), and confirms the
backbone repo is reachable on the Hub.
"""

import unittest

from tests.fixtures import (
    PRETRAINED,
    ModelTestCase,
    PooledTokenWrapperContract,
    hf_repo_exists,
)


class TestMmbertWrapper(PooledTokenWrapperContract, ModelTestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.text.mmbert import MmbertWrapper

        cls.model = MmbertWrapper(device_id=None, pretrained=PRETRAINED)


class TestMmbertWeightAvailability(unittest.TestCase):
    def test_mmbert_base_repo(self):
        self.assertTrue(hf_repo_exists("jhu-clsp/mmBERT-base"))


if __name__ == "__main__":
    unittest.main()
