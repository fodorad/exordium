"""EuroBERT multilingual text encoder wrapper (token-level or pooled).

EuroBERT is a **EuroBERT-architecture** encoder (Llama-style: rotary embeddings,
grouped-query attention) — not XLM-RoBERTa. It is kept in its own module so the
naming stays architecture-honest; it shares the ``pooling`` API of
:class:`~exordium.text.base.PooledTokenTextWrapper` with
:class:`~exordium.text.xlm_roberta.XlmRobertaWrapper` and
:class:`~exordium.text.mmbert.MmbertWrapper`.
"""

from exordium.text.base import PooledTokenTextWrapper


class EurobertWrapper(PooledTokenTextWrapper):
    r"""EuroBERT multilingual encoder — token-level ``(B, T, 1152)`` or pooled ``(B, 1152)``.

    Loads ``EuroBERT/EuroBERT-610m``, a EuroBERT-architecture encoder for 15
    European languages (**hidden 1152**, up to 8192 context, 2025) that is strong
    on European-language, code, and math tasks, and returns either the token-level
    sequence (``pooling="none"``, for a cross-modal sequence model such as
    MulT/LinMulT) or a pooled sentence embedding (``pooling="mean"``, the default).
    See :class:`~exordium.text.base.PooledTokenTextWrapper` for the full shape
    contract and the ``pooling`` semantics.

    Note the hidden size is **1152**, not 768 like the other multilingual wrappers;
    read it from :attr:`hidden_size` rather than assuming a value. Use this wrapper
    for European-language-heavy data; for broad multilingual coverage prefer
    :class:`~exordium.text.mmbert.MmbertWrapper`, and for cross-lingually aligned
    pooled embeddings prefer :class:`~exordium.text.xlm_roberta.XlmRobertaWrapper`.

    `EuroBERT: Scaling Multilingual Encoders for European Languages, Boizard et al.
    2025 <https://arxiv.org/abs/2503.05500>`_ ·
    `card <https://huggingface.co/EuroBERT/EuroBERT-610m>`_

    Args:
        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU/MPS.
        pretrained: ``True`` (default) loads the real weights. ``False`` builds
            the architecture with random weights — outputs are meaningless but the
            shapes and call contract hold (used by the test suite to avoid downloads).

    Attributes:
        hidden_size: The backbone's hidden size ``H`` (1152), read from the config.

    Example::

        m = EurobertWrapper()
        tokens = m("Ich bin sehr glücklich.", pooling="none")   # (1, T, 1152)
        pooled = m("Ich bin sehr glücklich.")                   # (1, 1152)

    """

    def __init__(self, device_id: int = -1, pretrained: bool = True) -> None:
        super().__init__("EuroBERT/EuroBERT-610m", device_id, pretrained=pretrained)
