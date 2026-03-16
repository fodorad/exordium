from exordium.text.base import TextModelWrapper


class BertWrapper(TextModelWrapper):
    """BERT (bert-base-uncased) text encoder. Hidden size: 768."""

    def __init__(self, device_id: int = -1) -> None:
        super().__init__("bert-base-uncased", device_id)
