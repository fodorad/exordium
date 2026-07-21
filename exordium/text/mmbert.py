"""mmBERT multilingual text encoder wrapper (token-level or pooled).

mmBERT is a **ModernBERT-architecture** massively multilingual encoder — not
XLM-RoBERTa. It is kept in its own module so the naming stays architecture-honest;
it shares the ``pooling`` API of :class:`~exordium.text.base.PooledTokenTextWrapper`
with :class:`~exordium.text.xlm_roberta.XlmRobertaWrapper` and
:class:`~exordium.text.eurobert.EurobertWrapper`.
"""

from exordium.text.base import PooledTokenTextWrapper


class MmbertWrapper(PooledTokenTextWrapper):
    r"""mmBERT multilingual encoder — token-level ``(B, T, 768)`` or pooled ``(B, 768)``.

    Loads ``jhu-clsp/mmBERT-base``, a ModernBERT-architecture massively
    multilingual encoder (1800+ languages, hidden 768, up to 8192 context, 2025),
    and returns either the token-level sequence (``pooling="none"``, for a
    cross-modal sequence model such as MulT/LinMulT) or a pooled sentence
    embedding (``pooling="mean"``, the default). See
    :class:`~exordium.text.base.PooledTokenTextWrapper` for the full shape contract
    and the ``pooling`` semantics.

    mmBERT is a *pretrained* encoder, **not fine-tuned for semantic similarity**,
    so its pooled embeddings are not cross-lingually aligned the way
    :class:`~exordium.text.xlm_roberta.XlmRobertaWrapper`'s ``mpnet`` backbone is —
    prefer that one for the pooled similarity path. mmBERT's strength here is the
    modern, high-quality **token-level** representation.

    `mmBERT: A Modern Multilingual Encoder with Annealed Language Learning, Marone
    et al. 2025 <https://arxiv.org/abs/2509.06888>`_ ·
    `card <https://huggingface.co/jhu-clsp/mmBERT-base>`_

    Args:
        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU/MPS.
        pretrained: ``True`` (default) loads the real weights. ``False`` builds
            the architecture with random weights — outputs are meaningless but the
            shapes and call contract hold (used by the test suite to avoid downloads).

    Attributes:
        hidden_size: The backbone's hidden size ``H`` (768), read from the config.

    Example::

        m = MmbertWrapper()
        tokens = m("Ich bin sehr glücklich.", pooling="none")   # (1, T, 768)
        pooled = m("Ich bin sehr glücklich.")                   # (1, 768)

    """

    def __init__(self, device_id: int = -1, pretrained: bool = True) -> None:
        super().__init__("jhu-clsp/mmBERT-base", device_id, pretrained=pretrained)
