import torch
import transformers as tfm

from exordium.utils.device import get_torch_device


class TextModelWrapper:
    """Base wrapper for HuggingFace encoder-only transformer models.

    Handles device placement, tokenization, and inference for models that
    return sequence hidden states (BERT, RoBERTa, XLM-RoBERTa, etc.).

    Args:
        model_name: HuggingFace model identifier.
        device_id: Device index. Use -1 or None for CPU, 0+ for GPU.

    """

    def __init__(self, model_name: str, device_id: int = -1) -> None:
        self.model_name = model_name
        self.device = get_torch_device(device_id)
        self.tokenizer = tfm.AutoTokenizer.from_pretrained(model_name)
        self.model = tfm.AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = True,
        max_length: int | None = None,
        pool: bool = False,
    ) -> torch.Tensor:
        """Encode text to hidden-state tensors.

        Args:
            text: Single string or list of strings.
            padding: Padding strategy passed to the tokenizer.
            max_length: Maximum sequence length (truncates if exceeded).
            pool: If True, apply mean pooling and return (B, H).
                  If False (default), return full sequence (B, T, H).

        Returns:
            torch.Tensor: Shape (B, T, H) when pool=False, (B, H) when pool=True.

        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state  # (B, T, H)

        if pool:
            return self._mean_pool(last_hidden, inputs["attention_mask"])  # (B, H)

        return last_hidden

    @torch.inference_mode()
    def inference(self, text: str, pool: bool = False) -> torch.Tensor:
        """Single-string inference optimized with inference mode.

        Args:
            text: A single input string.
            pool: If True, return mean-pooled (H,). If False, return (T, H).

        Returns:
            torch.Tensor: Shape (T, H) when pool=False, (H,) when pool=True.

        """
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        last_hidden = self.model(**inputs).last_hidden_state.squeeze(0)  # (T, H)

        if pool:
            return last_hidden.mean(dim=0)  # (H,)

        return last_hidden

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention-mask-weighted mean pooling over the sequence dimension.

        Args:
            last_hidden_state: Token embeddings of shape (B, T, H).
            attention_mask: Binary mask of shape (B, T).

        Returns:
            torch.Tensor: Sentence embeddings of shape (B, H).

        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )
