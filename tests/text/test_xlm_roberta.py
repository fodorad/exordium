"""Shape-contract tests for XlmRobertaWrapper's pooled sentence embeddings.

These tests exercise the pooling + ``as_sequence`` promotion logic on a fake
``last_hidden`` tensor and a stub model, so no checkpoint download or network
access is needed. They lock in the contract that the default output is a pooled
``(B, 768)`` vector with no time axis, and that ``as_sequence=True`` returns the
same vector as a length-1 sequence ``(B, 1, 768)``.
"""

import inspect
import unittest

import torch

from exordium.text.xlm_roberta import XlmRobertaWrapper


class _StubOutput:
    """Mimics a HuggingFace model output exposing ``last_hidden_state``."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


def _wrapper_with_fake_model(last_hidden: torch.Tensor) -> XlmRobertaWrapper:
    """Build an XlmRobertaWrapper whose forward pass returns *last_hidden*.

    ``__new__`` skips ``__init__`` (which would download a checkpoint); the model
    and device are stubbed so :meth:`inference` runs offline. The stub model
    ignores its inputs and always returns *last_hidden*, which is all the pooling
    and ``as_sequence`` shape logic under test needs.
    """
    wrapper = XlmRobertaWrapper.__new__(XlmRobertaWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.model = lambda **_: _StubOutput(last_hidden)
    return wrapper


class TestXlmRobertaSequenceShape(unittest.TestCase):
    """XlmRobertaWrapper.inference — pooled vs. length-1-sequence output shape."""

    def _inputs(self, batch: int, seq_len: int, hidden: int = 768):
        last_hidden = torch.randn(batch, seq_len, hidden)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        return last_hidden, {"attention_mask": attention_mask}

    def test_default_is_pooled_rank2(self):
        """Default output is a pooled ``(B, 768)`` vector — rank 2, no time axis."""
        last_hidden, inputs = self._inputs(batch=4, seq_len=7)
        wrapper = _wrapper_with_fake_model(last_hidden)

        out = wrapper.inference(inputs)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (4, 768))

    def test_as_sequence_is_rank3_with_unit_time_axis(self):
        """``as_sequence=True`` promotes to ``(B, 1, 768)`` — rank 3, T=1."""
        last_hidden, inputs = self._inputs(batch=4, seq_len=7)
        wrapper = _wrapper_with_fake_model(last_hidden)

        out = wrapper.inference(inputs, as_sequence=True)

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (4, 1, 768))

    def test_as_sequence_content_matches_pooled_squeezed(self):
        """The ``(B, 1, 768)`` view squeezed on T equals the pooled ``(B, 768)``."""
        last_hidden, inputs = self._inputs(batch=3, seq_len=5)
        wrapper = _wrapper_with_fake_model(last_hidden)

        pooled = wrapper.inference(inputs)
        sequence = wrapper.inference(inputs, as_sequence=True)

        self.assertTrue(torch.allclose(sequence.squeeze(1), pooled))

    def test_batch_of_one_default_stays_rank2(self):
        """A batch of one keeps rank 2 ``(1, 768)`` — not collapsed to ``(768,)``."""
        last_hidden, inputs = self._inputs(batch=1, seq_len=6)
        wrapper = _wrapper_with_fake_model(last_hidden)

        out = wrapper.inference(inputs)

        self.assertEqual(out.shape, (1, 768))

    def test_signature_defaults_to_pooled(self):
        """``as_sequence`` defaults to False, so existing callers are unaffected."""
        sig = inspect.signature(XlmRobertaWrapper.inference)
        self.assertIn("as_sequence", sig.parameters)
        self.assertEqual(sig.parameters["as_sequence"].default, False)


if __name__ == "__main__":
    unittest.main()
