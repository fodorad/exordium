"""Abstract base classes for text and speech-to-text models."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transformers as tfm

from exordium.utils.device import get_torch_device

WHISPER_WINDOW_SECONDS = 30.0
"""Length of Whisper's fixed receptive field; audio longer than this needs long-form decoding."""


@dataclass
class Segment:
    """An utterance-level chunk of a transcript with its time span.

    Produced by :meth:`~exordium.text.whisper.WhisperWrapper.transcribe_segments`
    and consumed by :meth:`ForcedAligner.align_segments`, which aligns each
    segment independently. Chunking this way keeps forced alignment bounded for
    long recordings (a 16-minute waveform never enters a single forward pass).

    Attributes:
        text: The segment's transcribed text.
        start: Segment start in seconds from the beginning of the audio.
        end: Segment end in seconds from the beginning of the audio.

    """

    text: str
    start: float
    end: float


def to_mono_16k(audio: "Path | str | np.ndarray | torch.Tensor") -> tuple[torch.Tensor, int]:
    """Return *audio* as a 1-D float32 waveform at 16 kHz plus its sample rate.

    Args:
        audio: File path, numpy array, or torch tensor (arrays/tensors are
            assumed to already be 16 kHz mono).

    Returns:
        Tuple of ``(waveform_1d, sample_rate)``.

    """
    from exordium.audio.io import load_audio

    if isinstance(audio, (str, Path)):
        waveform, sample_rate = load_audio(audio, target_sample_rate=16000, mono=True, squeeze=True)
        return waveform.float().reshape(-1), sample_rate
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().float().reshape(-1), 16000
    return torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(-1), 16000


@dataclass
class Word:
    """A single transcribed word with its time span on the audio timeline.

    This is the common currency exchanged between word-timestamp backends
    (Whisper + forced alignment, whisperX) and downstream consumers such as
    :func:`~exordium.text.transcript_align.find_segment`.  Keeping every backend
    behind this one type makes the re-alignment logic backend-agnostic.

    Attributes:
        text: The word as decoded (may include surrounding punctuation).
        start: Word start time in seconds from the beginning of the audio.
        end: Word end time in seconds from the beginning of the audio.
        score: Alignment confidence in ``[0, 1]``; ``1.0`` when unknown.

    """

    text: str
    start: float
    end: float
    score: float = 1.0


@dataclass
class SegmentMatch:
    """Result of fuzzy-searching a known transcript inside a word stream.

    Returned by :func:`~exordium.text.transcript_align.find_segment`.  The
    ``start``/``end`` recover the segment's true position on the *current*
    audio timeline, while ``score`` reports how well the query text matched
    (useful to accept, re-cut, or drop a dataset segment).

    Attributes:
        text: The concatenated matched span from the word stream.
        start: Match start time in seconds (from the first matched word).
        end: Match end time in seconds (from the last matched word).
        score: Fuzzy match coverage score in ``[0, 100]`` (rapidfuzz scale).
        word_start_idx: Index of the first matched word in the stream.
        word_end_idx: Index of the last matched word (inclusive).

    """

    text: str
    start: float
    end: float
    score: float
    word_start_idx: int
    word_end_idx: int


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


class ForcedAligner(ABC):
    """Abstract base class for transcript-to-audio forced aligners.

    A forced aligner takes audio **and its known transcript** and returns each
    word's time span.  Use it when the transcript is already available (e.g. a
    dataset annotation) and only the timing needs to be recovered.

    Concrete backends:

    * :class:`~exordium.text.alignment.TorchaudioForcedAligner`
      (``torchaudio`` ``MMS_FA``, always available).
    * :class:`~exordium.text.whisperx_align.WhisperxForcedAligner`
      (whisperX wav2vec2 alignment; ships with the ``text`` extra).

    """

    @abstractmethod
    def align(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        transcript: str,
        language: str | None = None,
    ) -> list[Word]:
        """Align a known transcript to audio and return timed words.

        Args:
            audio: Audio file path, numpy array, or torch tensor (16 kHz mono).
            transcript: The known transcript for this audio.
            language: Language code (e.g. ``"en"``) or ``None`` for the default.

        Returns:
            Words in chronological order with ``start``/``end``/``score``.

        """

    def align_segments(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        segments: list[Segment],
        language: str | None = None,
    ) -> list[Word]:
        """Align each segment separately and return one merged word stream.

        Aligning per segment keeps every forward pass short, so recordings of any
        length work without exhausting memory. Word times are shifted back onto
        the full-audio timeline by each segment's ``start``.

        Backends with a native batched segment API (e.g. whisperX) override this.

        Args:
            audio: Full audio — file path, numpy array, or torch tensor.
            segments: Timestamped transcript segments covering the audio.
            language: Language code or ``None`` for the default.

        Returns:
            Words in chronological order on the full-audio timeline.

        """
        waveform, sample_rate = to_mono_16k(audio)
        words: list[Word] = []
        for segment in segments:
            if not segment.text.strip():
                continue
            start = max(0, int(segment.start * sample_rate))
            end = min(waveform.numel(), int(segment.end * sample_rate))
            if end - start < sample_rate // 100:  # skip <10 ms slivers
                continue
            for word in self.align(waveform[start:end], segment.text, language=language):
                words.append(
                    Word(
                        text=word.text,
                        start=word.start + segment.start,
                        end=word.end + segment.start,
                        score=word.score,
                    )
                )
        return words


class WordTimestamper(ABC):
    """Abstract base class for open-vocabulary word-timestamp backends.

    Unlike :class:`ForcedAligner`, a word-timestamper needs **only audio**: it
    transcribes and time-stamps in one call (ASR followed by alignment).  This
    is the entry point for raw video where no transcript is known yet.

    Concrete backend:
    :class:`~exordium.text.alignment.WhisperWordTimestamper` (Whisper ASR plus a
    pluggable :class:`ForcedAligner`).

    """

    @abstractmethod
    def transcribe_words(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        language: str | None = None,
    ) -> list[Word]:
        """Transcribe audio and return timed words.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            language: Language code or ``None`` for auto/default.

        Returns:
            Words in chronological order with ``start``/``end``/``score``.

        """
