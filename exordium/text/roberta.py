"""RoBERTa text encoding model wrapper."""

import torch

from exordium.text.base import TextModelWrapper


class RobertaWrapper(TextModelWrapper):
    """RoBERTa-large token-level encoder. Hidden size: 1024.

    Returns all token hidden states of shape ``(B, T, 1024)``.  RoBERTa is not
    a sentence-transformer — it was not fine-tuned for semantic similarity, so
    mean-pooling its outputs does not produce meaningful sentence embeddings.
    Use the ``[CLS]`` token (index 0) for sequence-level tasks, or use a
    dedicated sentence-transformer such as
    :class:`~exordium.text.xml_roberta.XmlRobertaWrapper` instead.

    """

    def __init__(self, device_id: int = -1) -> None:
        super().__init__("roberta-large", device_id)

    def inference(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run RoBERTa forward pass.

        Args:
            inputs: Tokenizer output dict on ``self.device``.

        Returns:
            Token embeddings of shape ``(B, T, 1024)``.

        """
        return self.model(**inputs).last_hidden_state
