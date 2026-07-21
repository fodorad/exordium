"""XLM-RoBERTa multilingual text encoder wrapper (token-level or pooled).

Both backbones here are genuinely **XLM-RoBERTa architecture** encoders. For the
newer non-XLM-RoBERTa multilingual encoders see
:class:`~exordium.text.mmbert.MmbertWrapper` (ModernBERT) and
:class:`~exordium.text.eurobert.EurobertWrapper` (EuroBERT); they share the same
``pooling`` API but are separate classes so the naming stays architecture-honest.
"""

from exordium.text.base import PooledTokenTextWrapper

_BACKBONES: dict[str, str] = {
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "e5": "intfloat/multilingual-e5-base",
}
"""Named XLM-RoBERTa-architecture backbones. Keys are the short aliases accepted
by :class:`XlmRobertaWrapper`; values are their HuggingFace Hub ids. A raw Hub id
passed as ``backbone`` is used verbatim, so other XLM-RoBERTa checkpoints work too."""


class XlmRobertaWrapper(PooledTokenTextWrapper):
    r"""XLM-RoBERTa multilingual encoder — token-level ``(B, T, H)`` or pooled ``(B, H)``.

    Loads an **XLM-RoBERTa-architecture** multilingual encoder and returns either
    the token-level sequence (``pooling="none"``, for a cross-modal sequence model
    such as MulT/LinMulT) or a pooled sentence embedding (``pooling="mean"``, the
    default, for sentence similarity). See
    :class:`~exordium.text.base.PooledTokenTextWrapper` for the full shape contract
    and the ``pooling`` semantics, and
    :class:`~exordium.text.roberta.RobertaWrapper` for the English-only counterpart.

    Both backbones have hidden size 768; :attr:`hidden_size` reports it from the
    model config regardless.

    Args:
        backbone: Which XLM-RoBERTa encoder to load — a key of :data:`_BACKBONES`
            or a raw HuggingFace Hub id. Options:

            * ``"mpnet"`` (default) — XLM-RoBERTa sentence-transformer fine-tuned
              for semantic similarity (50+ languages, hidden 768); its embeddings
              are **cross-lingually aligned**, so it is the right choice for the
              pooled similarity path.
              `Sentence-BERT, Reimers & Gurevych 2019
              <https://arxiv.org/abs/1908.10084>`_ · `card <https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2>`_
            * ``"e5"`` — multilingual E5, initialized from XLM-RoBERTa and
              contrastively retrained (100+ languages, hidden 768). Its *pooled*
              mode expects ``"query: "`` / ``"passage: "`` prefixes on the input;
              the token-level output is unaffected.
              `Multilingual E5 Text Embeddings, Wang et al. 2024
              <https://arxiv.org/abs/2402.05672>`_ · `card <https://huggingface.co/intfloat/multilingual-e5-base>`_

        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU/MPS.
        pretrained: ``True`` (default) loads the real weights. ``False`` builds
            the architecture with random weights — outputs are meaningless but the
            shapes and call contract hold (used by the test suite to avoid downloads).

    Attributes:
        backbone: The ``backbone`` argument as given (alias or raw id).
        hidden_size: The backbone's hidden size ``H`` (768), read from the config.

    Example::

        m = XlmRobertaWrapper()                                 # default: mpnet
        tokens = m("Ich bin sehr glücklich.", pooling="none")   # (1, T, 768)
        pooled = m("Ich bin sehr glücklich.")                   # (1, 768)
        e5 = XlmRobertaWrapper(backbone="e5")                   # one-keyword switch

    """

    def __init__(
        self,
        backbone: str = "mpnet",
        device_id: int = -1,
        pretrained: bool = True,
    ) -> None:
        super().__init__(
            _BACKBONES.get(backbone, backbone),
            device_id,
            pretrained=pretrained,
        )
        self.backbone = backbone
