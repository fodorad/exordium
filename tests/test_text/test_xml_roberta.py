import logging
import os
import unittest
import warnings

import torch

from exordium.text.base import TextModelWrapper
from exordium.text.xml_roberta import XmlRobertaWrapper

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

HIDDEN = 768


class XmlRobertaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = XmlRobertaWrapper()

    # --- Initialization ---

    def test_inherits_text_model_wrapper(self):
        self.assertIsInstance(self.model, TextModelWrapper)

    def test_model_name(self):
        self.assertIn("paraphrase-multilingual", self.model.model_name)

    def test_model_loaded(self):
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.tokenizer)

    # --- __call__ default (pool=False) → (B, T, H) ---

    def test_single_string_sequence(self):
        feature = self.model("Welcome, this is an example")
        self.assertEqual(len(feature.shape), 3)
        self.assertEqual(feature.shape[2], HIDDEN)

    def test_multiple_strings_sequence(self):
        feature = self.model(
            ["Welcome, this is an example", "An another, longer example. I mean a lot longer."]
        )
        self.assertEqual(len(feature.shape), 3)
        self.assertEqual(feature.shape[0], 2)
        self.assertEqual(feature.shape[2], HIDDEN)

    def test_returns_torch_tensor(self):
        feature = self.model("Hello")
        self.assertIsInstance(feature, torch.Tensor)

    # --- __call__ with pool=True → (B, H) ---

    def test_single_string_pooled(self):
        feature = self.model("Welcome, this is an example", pool=True)
        self.assertEqual(feature.shape, (1, HIDDEN))

    def test_multiple_strings_pooled(self):
        feature = self.model(
            ["Welcome, this is an example", "An another, longer example. I mean a lot longer."],
            pool=True,
        )
        self.assertEqual(feature.shape, (2, HIDDEN))

    def test_pool_collapses_sequence_dim(self):
        seq = self.model("Hello world", pool=False)
        pooled = self.model("Hello world", pool=True)
        self.assertEqual(seq.shape[0], pooled.shape[0])
        self.assertEqual(seq.shape[2], pooled.shape[1])
        self.assertEqual(len(pooled.shape), 2)

    def test_batch_processing_shapes_pooled(self):
        texts = [
            "First sentence.",
            "Second sentence here.",
            "Third one is longer than the others.",
            "Fourth sentence is even longer and contains more words.",
        ]
        feature = self.model(texts, pool=True)
        self.assertEqual(feature.shape, (4, HIDDEN))

    def test_single_vs_batch_pooled(self):
        text = "This is a test sentence."
        single = self.model(text, pool=True)
        batch = self.model([text], pool=True)
        self.assertEqual(single.shape, batch.shape)
        torch.testing.assert_close(single, batch, rtol=1e-4, atol=1e-6)

    def test_empty_string_pooled(self):
        feature = self.model("", pool=True)
        self.assertIsNotNone(feature)
        self.assertEqual(feature.shape[1], HIDDEN)

    # --- _mean_pool static method ---

    def test_mean_pool_output_shape(self):
        batch_size, seq_len = 2, 5
        last_hidden = torch.randn(batch_size, seq_len, HIDDEN)
        attention_mask = torch.ones(batch_size, seq_len)
        pooled = TextModelWrapper._mean_pool(last_hidden, attention_mask)
        self.assertEqual(pooled.shape, (batch_size, HIDDEN))

    def test_mean_pool_with_partial_mask(self):
        batch_size, seq_len = 2, 5
        last_hidden = torch.randn(batch_size, seq_len, HIDDEN)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 3:] = 0  # mask last 2 tokens in first sequence
        pooled = TextModelWrapper._mean_pool(last_hidden, attention_mask)
        self.assertEqual(pooled.shape, (batch_size, HIDDEN))

    def test_mean_pool_all_zeros_mask_no_nan(self):
        """All-zero mask should not produce NaN (clamped denominator)."""
        last_hidden = torch.randn(1, 4, HIDDEN)
        attention_mask = torch.zeros(1, 4)
        pooled = TextModelWrapper._mean_pool(last_hidden, attention_mask)
        self.assertFalse(torch.isnan(pooled).any())

    # --- Sentence similarity (validates pooled embeddings are meaningful) ---

    def test_sentence_similarity(self):
        sentence1 = "The cat is sleeping on the couch."
        sentence2 = "A cat is resting on the sofa."
        sentence3 = "The weather is very nice today."
        emb1 = self.model(sentence1, pool=True)
        emb2 = self.model(sentence2, pool=True)
        emb3 = self.model(sentence3, pool=True)
        cos_sim_12 = torch.nn.functional.cosine_similarity(emb1, emb2)
        cos_sim_13 = torch.nn.functional.cosine_similarity(emb1, emb3)
        self.assertGreater(cos_sim_12.item(), cos_sim_13.item())
        self.assertGreater(cos_sim_12.item(), 0.5)

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
        inf = self.model.inference("Test sentence")
        call = self.model("Test sentence")
        self.assertEqual(inf.shape[-1], call.shape[-1])


if __name__ == "__main__":
    unittest.main()
