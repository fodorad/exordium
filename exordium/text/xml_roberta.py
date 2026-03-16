"""XLM-RoBERTa multilingual text encoding wrapper."""

from exordium.text.base import TextModelWrapper


class XmlRobertaWrapper(TextModelWrapper):
    """XLM-RoBERTa sentence-transformer wrapper. Hidden size: 768."""

    def __init__(self, device_id: int = -1) -> None:
        super().__init__(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            device_id,
        )
