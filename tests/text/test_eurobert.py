"""Tests for the EuroBERT multilingual text encoder wrapper.

Exercises the shared token/pooled ``pooling`` contract against the real
:class:`~exordium.text.eurobert.EurobertWrapper` (with random weights when
``PRETRAINED`` is ``False``, so no checkpoint is downloaded), and confirms the
backbone repo is reachable on the Hub. EuroBERT's hidden size is 1152, not 768;
the contract reads it from the model config rather than assuming a value.
"""

import unittest

from tests.fixtures import (
    PRETRAINED,
    ModelTestCase,
    PooledTokenWrapperContract,
    hf_repo_exists,
)


class TestEurobertWrapper(PooledTokenWrapperContract, ModelTestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.text.eurobert import EurobertWrapper

        cls.model = EurobertWrapper(device_id=None, pretrained=PRETRAINED)

    def test_hidden_size_is_1152(self):
        """EuroBERT-610m has hidden size 1152 — the non-uniform case."""
        self.assertEqual(self.model.hidden_size, 1152)


class TestEurobertWeightAvailability(unittest.TestCase):
    def test_eurobert_610m_repo(self):
        self.assertTrue(hf_repo_exists("EuroBERT/EuroBERT-610m"))


if __name__ == "__main__":
    unittest.main()
