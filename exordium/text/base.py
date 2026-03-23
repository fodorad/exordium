"""Abstract base classes for text and speech-to-text models."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import transformers as tfm

from exordium.utils.device import get_torch_device


class TextModelWrapper(ABC):
    """Abstract base class for HuggingFace encoder-only transformer models.

    Handles device placement and provides the standard
    ``preprocess → inference → __call__`` pipeline used by all text wrappers
    in this library (mirrors :class:`~exordium.video.deep.base.VisualModelWrapper`
    and :class:`~exordium.audio.base.AudioModelWrapper`).

    Subclasses must implement :meth:`inference`, which receives the tokenizer
    output dict (already on ``self.device``) and returns a tensor.

    Typical patterns:

    * Token-level features — return ``outputs.last_hidden_state`` ``(B, T, H)``
    * Sentence-level features — return :meth:`_mean_pool` applied to
      ``last_hidden_state`` ``(B, H)``
    * CLS-token only — return ``outputs.last_hidden_state[:, 0]`` ``(B, H)``

    Args:
        model_name: HuggingFace model identifier (e.g. ``"bert-base-uncased"``).
        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU.

    Example::

        class BertWrapper(TextModelWrapper):
            def __init__(self, device_id=-1):
                super().__init__("bert-base-uncased", device_id)

            def inference(self, inputs):
                return self.model(**inputs).last_hidden_state  # (B, T, 768)

        model = BertWrapper()
        hidden = model("Hello world")          # torch.Tensor (1, T, 768)
        feats  = model.predict(["a", "b"])     # np.ndarray  (2, T, 768)

    """

    def __init__(self, model_name: str, device_id: int = -1) -> None:
        self.model_name = model_name
        self.device = get_torch_device(device_id)
        self.tokenizer = tfm.AutoTokenizer.from_pretrained(model_name)
        self.model = tfm.AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def preprocess(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool | str = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize text and move tensors to ``self.device``.

        Args:
            text: Single string or list of strings.
            max_length: Maximum sequence length; truncates if exceeded.
                ``None`` uses the tokenizer's default.
            padding: Padding strategy forwarded to the tokenizer.
                ``True`` / ``"longest"`` pads to the longest sequence in the
                batch; ``"max_length"`` pads to ``max_length``.

        Returns:
            Dict of tensors (``input_ids``, ``attention_mask``, …) on
            ``self.device``.

        """
        assert self.tokenizer is not None
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @abstractmethod
    def inference(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Model forward pass.

        Receives the tokenizer output dict (already on ``self.device``) and
        returns the feature tensor.  All extraction logic (which output to use,
        whether to pool, etc.) belongs here.

        Args:
            inputs: Dict returned by :meth:`preprocess` — ``input_ids``,
                ``attention_mask``, and optionally ``token_type_ids``.

        Returns:
            Feature tensor, shape depends on the subclass:

            * ``(B, T, H)`` for token-level encoders
            * ``(B, H)`` for sentence-level encoders (pooled)

        """

    def __call__(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool | str = True,
    ) -> torch.Tensor:
        """Tokenize and encode text, returning a feature tensor.

        Args:
            text: Single string or list of strings.
            max_length: Maximum sequence length. ``None`` uses tokenizer default.
            padding: Padding strategy. Default: ``True`` (pad to longest).

        Returns:
            Feature tensor on ``self.device``.  Shape is determined by
            :meth:`inference` (token-level or sentence-level).

        """
        with torch.inference_mode():
            return self.inference(self.preprocess(text, max_length, padding))

    def predict(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool | str = True,
    ) -> np.ndarray:
        """Tokenize and encode text, returning a numpy feature array.

        Args:
            text: Single string or list of strings.
            max_length: Maximum sequence length.
            padding: Padding strategy.

        Returns:
            Feature array. Shape is determined by :meth:`inference`.

        """
        return self(text, max_length, padding).cpu().numpy()

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-mask-weighted mean pooling over the sequence dimension.

        Excludes padding tokens from the mean by weighting with the binary
        attention mask before summing and normalising.

        Args:
            last_hidden_state: Token embeddings of shape ``(B, T, H)``.
            attention_mask: Binary mask of shape ``(B, T)``.

        Returns:
            Sentence embeddings of shape ``(B, H)``.

        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class SpeechToText(ABC):
    """Abstract base class for speech-to-text models.

    Defines the standard ``preprocess → inference → __call__`` pipeline.
    Subclasses must implement :meth:`preprocess` and :meth:`inference`.

    """

    @abstractmethod
    def preprocess(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
    ) -> object:
        """Convert any audio input to model-ready features.

        Args:
            audio: Audio file path, numpy array, or torch tensor.

        Returns:
            Model-ready input (type depends on the backend).

        """

    @abstractmethod
    def inference(self, inputs: object, **kwargs: object) -> str:
        """Run the model and return the full transcript.

        Args:
            inputs: Output of :meth:`preprocess`.
            **kwargs: Optional backend-specific keyword arguments.

        Returns:
            Transcribed text as a plain string.

        """

    def __call__(self, audio: Path | str | np.ndarray | torch.Tensor) -> str:
        """Preprocess and transcribe, returning the full transcript.

        Args:
            audio: Audio file path, numpy array, or torch tensor.

        Returns:
            Transcribed text as a plain string.

        """
        return self.inference(self.preprocess(audio))


class StreamingMixin(ABC):
    """Mixin for STT backends that support word-by-word streaming output.

    Add this mixin alongside :class:`SpeechToText` for backends where the
    model decodes incrementally and can yield tokens as they are produced::

        class WhisperWrapper(StreamingMixin, SpeechToText):
            ...

    """

    @abstractmethod
    def stream(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
    ) -> Iterator[str]:
        """Transcribe audio and yield decoded text chunks incrementally.

        Args:
            audio: Audio file path, numpy array, or torch tensor.

        Yields:
            Decoded text chunks (words or sub-word tokens) as they are produced.

        """
