"""Shape-contract tests for the pooling API and the multilingual wrapper registries.

The pooling-mode tests exercise the ``pooling`` branch logic of
:class:`~exordium.text.base.PooledTokenTextWrapper` on a fake ``last_hidden``
tensor and a stub model, so no checkpoint download or network access is needed.
They lock in the contract that ``pooling="mean"`` (default) returns a pooled
``(B, H)`` vector and ``pooling="none"`` returns the raw token sequence
``(B, T, H)``. ``H`` is parametrised over ``{768, 1152}`` so the differing-hidden
case (EuroBERT) is covered without a download.
"""

import inspect
import unittest

import torch

from exordium.text.base import PooledTokenTextWrapper, TextModelWrapper
from exordium.text.eurobert import EurobertWrapper
from exordium.text.mmbert import MmbertWrapper
from exordium.text.xlm_roberta import _BACKBONES, XlmRobertaWrapper


class _StubOutput:
    """Mimics a HuggingFace model output exposing ``last_hidden_state``."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


def _wrapper_with_fake_model(last_hidden: torch.Tensor) -> PooledTokenTextWrapper:
    """Build a PooledTokenTextWrapper whose forward pass returns *last_hidden*.

    ``__new__`` skips ``__init__`` (which would download a checkpoint); the model
    and device are stubbed so :meth:`inference` runs offline. The stub model
    ignores its inputs and always returns *last_hidden*, which is all the pooling
    shape logic under test needs.
    """
    wrapper = PooledTokenTextWrapper.__new__(PooledTokenTextWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.model = lambda **_: _StubOutput(last_hidden)
    return wrapper


class TestPoolingShape(unittest.TestCase):
    """PooledTokenTextWrapper.inference — token-level vs. pooled output shape."""

    def _inputs(self, batch: int, seq_len: int, hidden: int = 768):
        last_hidden = torch.randn(batch, seq_len, hidden)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        return last_hidden, {"attention_mask": attention_mask}

    def test_default_is_pooled_rank2(self):
        """Default (``pooling='mean'``) output is pooled ``(B, H)`` — no time axis."""
        for hidden in (768, 1152):
            with self.subTest(hidden=hidden):
                last_hidden, inputs = self._inputs(batch=4, seq_len=7, hidden=hidden)
                wrapper = _wrapper_with_fake_model(last_hidden)

                out = wrapper.inference(inputs)

                self.assertEqual(out.ndim, 2)
                self.assertEqual(out.shape, (4, hidden))

    def test_pooled_equals_mean_pool(self):
        """``pooling='mean'`` equals the shared :meth:`TextModelWrapper._mean_pool`."""
        last_hidden, inputs = self._inputs(batch=3, seq_len=5)
        wrapper = _wrapper_with_fake_model(last_hidden)

        out = wrapper.inference(inputs, pooling="mean")
        expected = TextModelWrapper._mean_pool(last_hidden, inputs["attention_mask"])

        self.assertTrue(torch.allclose(out, expected))

    def test_none_is_rank3_token_sequence(self):
        """``pooling='none'`` returns token-level ``(B, T, H)`` — rank 3, T preserved."""
        for hidden in (768, 1152):
            with self.subTest(hidden=hidden):
                last_hidden, inputs = self._inputs(batch=4, seq_len=7, hidden=hidden)
                wrapper = _wrapper_with_fake_model(last_hidden)

                out = wrapper.inference(inputs, pooling="none")

                self.assertEqual(out.ndim, 3)
                self.assertEqual(out.shape, (4, 7, hidden))

    def test_none_equals_raw_last_hidden(self):
        """``pooling='none'`` returns the raw ``last_hidden_state`` untouched."""
        last_hidden, inputs = self._inputs(batch=3, seq_len=5)
        wrapper = _wrapper_with_fake_model(last_hidden)

        out = wrapper.inference(inputs, pooling="none")

        self.assertTrue(torch.equal(out, last_hidden))

    def test_batch_of_one_pooled_stays_rank2(self):
        """A batch of one keeps rank 2 ``(1, H)`` — not collapsed to ``(H,)``."""
        last_hidden, inputs = self._inputs(batch=1, seq_len=6)
        wrapper = _wrapper_with_fake_model(last_hidden)

        out = wrapper.inference(inputs)

        self.assertEqual(out.shape, (1, 768))

    def test_batch_of_one_token_stays_rank3(self):
        """A batch of one keeps rank 3 ``(1, T, H)`` for the token-level mode."""
        last_hidden, inputs = self._inputs(batch=1, seq_len=6)
        wrapper = _wrapper_with_fake_model(last_hidden)

        out = wrapper.inference(inputs, pooling="none")

        self.assertEqual(out.shape, (1, 6, 768))

    def test_per_sample_views(self):
        """``[0]`` drops the batch axis: token → ``(T, H)``, pooled → ``(H,)``."""
        last_hidden, inputs = self._inputs(batch=2, seq_len=5)
        wrapper = _wrapper_with_fake_model(last_hidden)

        token = wrapper.inference(inputs, pooling="none")
        pooled = wrapper.inference(inputs, pooling="mean")

        self.assertEqual(token[0].shape, (5, 768))
        self.assertEqual(pooled[0].shape, (768,))

    def test_invalid_pooling_raises(self):
        """An unknown ``pooling`` value raises ``ValueError``."""
        last_hidden, inputs = self._inputs(batch=2, seq_len=4)
        wrapper = _wrapper_with_fake_model(last_hidden)

        with self.assertRaises(ValueError):
            wrapper.inference(inputs, pooling="cls")

    def test_signature_defaults_to_pooled(self):
        """``pooling`` defaults to 'mean', so existing callers get pooled output."""
        sig = inspect.signature(PooledTokenTextWrapper.inference)
        self.assertIn("pooling", sig.parameters)
        self.assertEqual(sig.parameters["pooling"].default, "mean")


class TestXlmRobertaBackboneRegistry(unittest.TestCase):
    """XlmRobertaWrapper carries only genuine XLM-RoBERTa backbones."""

    def test_only_xlm_roberta_backbones_registered(self):
        """Only the two XLM-RoBERTa-architecture aliases are registered here."""
        self.assertEqual(set(_BACKBONES), {"mpnet", "e5"})

    def test_default_backbone_is_mpnet(self):
        """The constructor default is the aligned mpnet backbone."""
        sig = inspect.signature(XlmRobertaWrapper.__init__)
        self.assertEqual(sig.parameters["backbone"].default, "mpnet")

    def test_aliases_map_to_expected_repo_ids(self):
        """Each alias resolves to its expected XLM-RoBERTa HuggingFace repo id."""
        self.assertEqual(
            _BACKBONES["mpnet"],
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
        self.assertEqual(_BACKBONES["e5"], "intfloat/multilingual-e5-base")


class TestStandaloneWrapperClasses(unittest.TestCase):
    """mmBERT and EuroBERT are standalone classes, not XlmRobertaWrapper backbones."""

    def test_all_share_the_pooling_base(self):
        """Every multilingual wrapper inherits the shared pooling API."""
        for cls in (XlmRobertaWrapper, MmbertWrapper, EurobertWrapper):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, PooledTokenTextWrapper))

    def test_non_xlm_roberta_models_are_not_in_registry(self):
        """mmBERT/EuroBERT repo ids do not leak into the XLM-RoBERTa registry."""
        ids = set(_BACKBONES.values())
        self.assertNotIn("jhu-clsp/mmBERT-base", ids)
        self.assertNotIn("EuroBERT/EuroBERT-610m", ids)


if __name__ == "__main__":
    unittest.main()
