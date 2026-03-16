"""RoBERTa text encoding model wrapper."""

from exordium.text.base import TextModelWrapper


class RobertaWrapper(TextModelWrapper):
    """RoBERTa-large text encoder. Hidden size: 1024."""

    def __init__(self, device_id: int = -1) -> None:
        super().__init__("roberta-large", device_id)
