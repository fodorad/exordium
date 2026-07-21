"""XLM-RoBERTa multilingual text encoding wrapper."""

import numpy as np
import torch

from exordium.text.base import TextModelWrapper


class XlmRobertaWrapper(TextModelWrapper):
    """XLM-RoBERTa sentence-transformer wrapper. Hidden size: 768.

    This model is a sentence-transformer fine-tuned for semantic similarity, so
    :meth:`inference` mean-pools the token embeddings internally and returns a
    **pooled sentence embedding**: one ``(768,)`` vector per input, shape
    ``(B, 768)``. There is *no* time axis — the pooled vector is not a length-1
    sequence, and consumers should not treat ``(B, 768)`` as ``(B, T=1, 768)``.

    A time-series consumer that genuinely wants a length-1 sequence (e.g. a
    LinMulT/LinT model with ``time_dim_reducer='gap'`` that expects ``(B, T, D)``)
    can request the promoted rank explicitly via ``as_sequence=True``, which
    returns ``(B, 1, 768)``. That unsqueeze is a consumer *choice* named here in
    exordium rather than re-derived in every downstream repo; it adds no new
    information over the pooled vector.

    """

    def __init__(self, device_id: int = -1, pretrained: bool = True) -> None:
        super().__init__(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            device_id,
            pretrained=pretrained,
        )

    def inference(
        self,
        inputs: dict[str, torch.Tensor],
        as_sequence: bool = False,
    ) -> torch.Tensor:
        """Run XLM-RoBERTa forward pass with mean pooling.

        Args:
            inputs: Tokenizer output dict on ``self.device``.
            as_sequence: When ``False`` (default), return the pooled sentence
                embedding of shape ``(B, 768)`` — one ``(768,)`` vector per input,
                with no time axis. When ``True``, unsqueeze a length-1 time axis
                and return ``(B, 1, 768)``; this is a length-1 *sequence* view of
                the same pooled vector, not additional information.

        Returns:
            Pooled sentence embeddings of shape ``(B, 768)``, or ``(B, 1, 768)``
            when ``as_sequence`` is ``True``.

        """
        last_hidden = self.model(**inputs).last_hidden_state
        pooled = self._mean_pool(last_hidden, inputs["attention_mask"])
        if as_sequence:
            return pooled.unsqueeze(1)
        return pooled

    def __call__(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool | str = True,
        as_sequence: bool = False,
    ) -> torch.Tensor:
        """Tokenize and encode text, returning a pooled sentence-embedding tensor.

        Args:
            text: Single string or list of strings.
            max_length: Maximum sequence length. ``None`` uses tokenizer default.
            padding: Padding strategy. Default: ``True`` (pad to longest).
            as_sequence: Forwarded to :meth:`inference`. ``False`` returns
                ``(B, 768)`` (pooled, no time axis); ``True`` returns
                ``(B, 1, 768)`` (length-1 sequence view).

        Returns:
            Feature tensor on ``self.device``: ``(B, 768)`` by default, or
            ``(B, 1, 768)`` when ``as_sequence`` is ``True``.

        """
        with torch.inference_mode():
            return self.inference(
                self.preprocess(text, max_length, padding), as_sequence=as_sequence
            )

    def predict(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool | str = True,
        as_sequence: bool = False,
    ) -> np.ndarray:
        """Tokenize and encode text, returning a pooled sentence-embedding array.

        Args:
            text: Single string or list of strings.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            as_sequence: Forwarded to :meth:`inference`. ``False`` returns
                ``(B, 768)`` (pooled, no time axis); ``True`` returns
                ``(B, 1, 768)`` (length-1 sequence view).

        Returns:
            Feature array: ``(B, 768)`` by default, or ``(B, 1, 768)`` when
            ``as_sequence`` is ``True``.

        """
        return self(text, max_length, padding, as_sequence=as_sequence).cpu().numpy()
