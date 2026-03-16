import logging
import os
import unittest
import warnings

import torch

from exordium.text.base import TextModelWrapper
from exordium.text.roberta import RobertaWrapper

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

HIDDEN = 1024


class RobertaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobertaWrapper()

    # --- Initialization ---

    def test_inherits_text_model_wrapper(self):
        self.assertIsInstance(self.model, TextModelWrapper)

    def test_model_name(self):
        self.assertEqual(self.model.model_name, "roberta-large")

    def test_model_loaded(self):
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.tokenizer)

    # --- __call__ default (pool=False) → (B, T, H) ---

    def test_single_string(self):
        feature = self.model("Welcome, this is an example")
        self.assertEqual(feature.shape, (1, 8, HIDDEN))

    def test_multiple_strings(self):
        feature = self.model(
            ["Welcome, this is an example", "An another, longer example. I mean a lot longer."]
        )
        self.assertEqual(feature.shape, (2, 14, HIDDEN))

    def test_padding_max_length(self):
        feature = self.model("Welcome, this is an example", padding="max_length", max_length=20)
        self.assertEqual(feature.shape, (1, 20, HIDDEN))

    def test_truncation(self):
        long_text = " ".join(["This is a very long text that should be truncated."] * 20)
        feature = self.model(long_text, max_length=30)
        self.assertEqual(feature.shape[1], 30)

    def test_batch_processing(self):
        texts = ["First sentence.", "Second sentence here.", "Third one is longer than the others."]
        feature = self.model(texts)
        self.assertEqual(feature.shape[0], 3)
        self.assertEqual(feature.shape[2], HIDDEN)

    def test_empty_string(self):
        feature = self.model("")
        self.assertIsNotNone(feature)
        self.assertEqual(feature.shape[2], HIDDEN)

    def test_returns_torch_tensor(self):
        feature = self.model("Hello")
        self.assertIsInstance(feature, torch.Tensor)

    # --- __call__ with pool=True → (B, H) ---

    def test_pool_false_is_default(self):
        seq = self.model("Hello world", pool=False)
        self.assertEqual(len(seq.shape), 3)

    def test_pool_true_single(self):
        feature = self.model("Hello world", pool=True)
        self.assertEqual(feature.shape, (1, HIDDEN))

    def test_pool_true_batch(self):
        texts = ["First sentence.", "Second sentence."]
        feature = self.model(texts, pool=True)
        self.assertEqual(feature.shape, (2, HIDDEN))

    def test_pool_collapses_sequence_dim(self):
        seq = self.model("Hello world", pool=False)
        pooled = self.model("Hello world", pool=True)
        self.assertEqual(seq.shape[0], pooled.shape[0])
        self.assertEqual(seq.shape[2], pooled.shape[1])
        self.assertEqual(len(pooled.shape), 2)

    # --- inference() → (T, H) or (H,) ---

    def test_inference_sequence_shape(self):
        feature = self.model.inference("Hello world")
        self.assertEqual(len(feature.shape), 2)
        self.assertEqual(feature.shape[1], HIDDEN)

    def test_inference_pooled_shape(self):
        feature = self.model.inference("Hello world", pool=True)
        self.assertEqual(feature.shape, (HIDDEN,))

    def test_inference_returns_torch_tensor(self):
        feature = self.model.inference("Hello")
        self.assertIsInstance(feature, torch.Tensor)

    def test_inference_consistent_with_call(self):
        """inference(text) and __call__([text]) should have the same hidden dim."""
        inf = self.model.inference("Test sentence")
        call = self.model("Test sentence")
        self.assertEqual(inf.shape[-1], call.shape[-1])


if __name__ == "__main__":
    unittest.main()
