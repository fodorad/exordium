"""XLM-RoBERTa multilingual text encoding wrapper."""

import torch

from exordium.text.base import TextModelWrapper


class XmlRobertaWrapper(TextModelWrapper):
    """XLM-RoBERTa sentence-transformer wrapper. Hidden size: 768.

    Returns mean-pooled sentence embeddings of shape ``(B, 768)``.
    This model is a sentence-transformer fine-tuned for semantic similarity,
    so mean pooling is applied internally by :meth:`inference`.

    """

    def __init__(self, device_id: int = -1) -> None:
        super().__init__(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            device_id,
        )

    def inference(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run XLM-RoBERTa forward pass with mean pooling.

        Args:
            inputs: Tokenizer output dict on ``self.device``.

        Returns:
            Sentence embeddings of shape ``(B, 768)``.

        """
        last_hidden = self.model(**inputs).last_hidden_state
        return self._mean_pool(last_hidden, inputs["attention_mask"])
