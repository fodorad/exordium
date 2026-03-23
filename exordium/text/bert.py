"""BERT text encoding model wrapper."""

import torch

from exordium.text.base import TextModelWrapper


class BertWrapper(TextModelWrapper):
    """BERT (bert-base-uncased) token-level encoder. Hidden size: 768.

    Returns all token hidden states of shape ``(B, T, 768)``.  BERT is not a
    sentence-transformer — it was not fine-tuned for semantic similarity, so
    mean-pooling its outputs does not produce meaningful sentence embeddings.
    Use the ``[CLS]`` token (index 0) for sequence-level tasks, or use a
    dedicated sentence-transformer such as
    :class:`~exordium.text.xml_roberta.XmlRobertaWrapper` instead.

    """

    def __init__(self, device_id: int = -1) -> None:
        super().__init__("bert-base-uncased", device_id)

    def inference(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run BERT forward pass.

        Args:
            inputs: Tokenizer output dict on ``self.device``.

        Returns:
            Token embeddings of shape ``(B, T, 768)``.

        """
        return self.model(**inputs).last_hidden_state
