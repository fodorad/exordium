"""Abstract base classes for text and speech-to-text models."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transformers as tfm

from exordium.utils.device import get_torch_device

logger = logging.getLogger(__name__)
"""Module-level logger."""

WHISPER_WINDOW_SECONDS = 30.0
"""Length of Whisper's fixed receptive field; audio longer than this needs long-form decoding."""

WAV2VEC2_RECEPTIVE_FIELD_SAMPLES = 400
"""Samples consumed by one wav2vec2 emission frame (the conv frontend's receptive field)."""

WAV2VEC2_STRIDE_SAMPLES = 320
"""Samples between consecutive wav2vec2 emission frames (the conv frontend's total stride)."""

MIN_ALIGN_SAMPLES = WAV2VEC2_RECEPTIVE_FIELD_SAMPLES + WAV2VEC2_STRIDE_SAMPLES
"""Shortest slice (720 samples, 45 ms @ 16 kHz) that yields **two** emission frames.

One frame is never enough: whisperX rescales its timestamps by
``duration / (trellis.size(0) - 1)``, so a one-row trellis divides by zero
(``ZeroDivisionError``) and aborts the whole recording, and ``MMS_FA`` cannot even run
its conv stack below 400 samples (``RuntimeError``).  This is the floor for *any* text;
:func:`wav2vec2_min_samples` gives the real, text-dependent requirement.
"""


WIDEN_WINDOW_SECONDS = 0.5
"""Window a degenerate segment is widened to, over and above the geometric minimum.

:func:`wav2vec2_min_samples` says what the *model* needs; it says nothing about what the
*speech* needs.  A 4-token word like ``"that"`` needs only 85 ms of frames, but takes
~250 ms to say, so widening to the geometric floor alone hands CTC a window that clips
the word — it aligns, but the timestamp comes back truncated (measured: ``"hey"`` scored
0.02 and returned a 65 ms span against a true 320 ms, starting 137 ms late).

Half a second gives the word room to sit inside the window; measured start error against
ground truth drops to ~20 ms (about one emission frame).  Wider windows start to drift,
as the aligner has more audio in which to mistake a similar-sounding neighbour for the
word it is looking for.
"""


def wav2vec2_min_samples(num_tokens: int) -> int:
    """Samples needed to force-align *num_tokens* tokens with a wav2vec2 CTC model.

    The conv frontend turns ``N`` samples into ``floor((N - 400) / 320) + 1`` emission
    frames, and CTC cannot emit more tokens than it has frames.  Aligning ``T`` tokens
    therefore needs ``T`` frames, i.e. ``N >= 400 + 320 * (T - 1)`` — never fewer than
    :data:`MIN_ALIGN_SAMPLES`, since a lone frame breaks both backends outright.

    This is why a *constant* minimum is the wrong shape: ``"I"`` needs 720 samples but
    ``"that"`` needs 1360.  Starving a segment of frames does not merely lose precision —
    ``MMS_FA`` raises, and whisperX's backtrack fails and returns the words untimed,
    which drops them silently.

    Args:
        num_tokens: Number of CTC tokens (characters, for these models) to align.

    Returns:
        Minimum slice length in samples.

    """
    tokens = max(1, num_tokens)
    span = WAV2VEC2_RECEPTIVE_FIELD_SAMPLES + WAV2VEC2_STRIDE_SAMPLES * (tokens - 1)
    return max(MIN_ALIGN_SAMPLES, span)


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


def _widen(start: int, end: int, target: int, limit: int | None) -> tuple[int, int]:
    """Grow a ``[start, end)`` sample window to *target* samples around its midpoint.

    Slides the window back inside ``[0, limit)`` if centring would overshoot either edge,
    so the result is always ``target`` samples long (the caller guarantees the audio is
    long enough).

    Args:
        start: Window start in samples.
        end: Window end in samples.
        target: Desired window length in samples.
        limit: Length of the audio in samples, or ``None`` if unbounded.

    Returns:
        The widened ``(start, end)`` in samples.

    """
    deficit = target - (end - start)
    new_start = start - deficit // 2
    new_end = new_start + target
    if new_start < 0:
        new_start, new_end = 0, target
    if limit is not None and new_end > limit:
        new_end, new_start = limit, limit - target
    return max(0, new_start), new_end


def prepare_segments(
    segments: list[Segment],
    min_samples_for: "Callable[[str], int]",
    sample_rate: int = 16000,
    num_samples: int | None = None,
) -> list[Segment]:
    """Widen segments that are too short for a wav2vec2 aligner to time-stamp.

    Long-form Whisper occasionally emits **degenerate micro-segments**: real text with a
    near-zero span (e.g. ``"I"`` over 0.02 s).  The text is fine and the surrounding audio
    exists — only the *window* is too narrow to give CTC one emission frame per token.
    Such a window makes ``MMS_FA`` raise and makes whisperX divide by zero, so the naive
    remedy is to drop the segment; that throws the word away for no reason.

    Instead, each short segment is grown around its midpoint until it satisfies both
    floors — :func:`wav2vec2_min_samples` (what the model needs) and
    :data:`WIDEN_WINDOW_SECONDS` (what the *speech* needs) — clamped to the audio.  The
    aligner then sees enough audio to actually locate the word, and returns a real
    timestamp for it instead of nothing.

    Healthy segments are passed through with their span clamped to the audio, and are
    otherwise untouched: the speech floor applies only to segments already known to be too
    short for the model, so normal short utterances keep the span Whisper gave them.

    A widened window may overlap its neighbours, so the words two adjacent segments emit
    can interleave in time; :meth:`ForcedAligner.align_segments` sorts the merged stream.

    Blank-text segments are skipped.  A segment is dropped when the *whole recording* is
    too short to carry its text, or when it lies entirely past the end of the audio —
    the two cases widening cannot fix.

    Args:
        segments: Transcript segments, typically straight from Whisper.
        min_samples_for: Returns the samples a given text needs. Backends pass their own
            tokenizer-exact version; see :meth:`ForcedAligner.min_align_samples`.
        sample_rate: Sample rate of the audio the segments refer to.
        num_samples: Length of the audio in samples. Windows are clamped to it, so a
            segment running past the end of the audio is judged on the slice that exists.

    Returns:
        Alignable segments in chronological order, some with widened spans. Widened and
        dropped segments are logged so neither is silent.

    """
    limit = num_samples
    prepared: list[Segment] = []
    widened: list[Segment] = []
    dropped: list[Segment] = []

    for segment in segments:
        text = segment.text
        if not text.strip():
            continue

        start = max(0, int(float(segment.start) * sample_rate))
        end = max(start, int(float(segment.end) * sample_rate))
        needed = min_samples_for(text)

        clamped = False
        if limit is not None:
            # Widening cannot conjure audio: neither for text longer than the whole
            # recording, nor for a segment that starts past the end of it.
            if needed > limit or start >= limit:
                dropped.append(segment)
                continue
            clamped = end > limit
            end = min(end, limit)

        if end - start >= needed:
            # A segment overrunning the audio is clamped, so whisperX rescales its
            # timestamps against the audio it really gets — a nominal end past the file
            # end would otherwise inflate its frame ratio and stretch the word times.
            prepared.append(_to_segment(text, start, end, sample_rate) if clamped else segment)
            continue

        # The span is unusable, so give the word a window it can actually be heard in,
        # not merely the model's bare minimum — but never more audio than we have.
        target = max(needed, int(WIDEN_WINDOW_SECONDS * sample_rate))
        if limit is not None:
            target = max(needed, min(target, limit))

        new_start, new_end = _widen(start, end, target, limit)
        prepared.append(_to_segment(text, new_start, new_end, sample_rate))
        widened.append(segment)

    _log_prepared(segments, widened, dropped)
    return prepared


def _to_segment(text: str, start: int, end: int, sample_rate: int) -> Segment:
    """Build a :class:`Segment` from a sample-space window."""
    return Segment(text=text, start=start / sample_rate, end=end / sample_rate)


def _log_prepared(segments: list[Segment], widened: list[Segment], dropped: list[Segment]) -> None:
    """Report widened and dropped segments so neither loss is silent."""

    def preview(items: list[Segment]) -> str:
        shown = ", ".join(f"{s.text.strip()!r} @ {s.start:.2f}s" for s in items[:5])
        return shown + (f" (+{len(items) - 5} more)" if len(items) > 5 else "")

    if widened:
        logger.info(
            f"Widened {len(widened)}/{len(segments)} segment(s) too short to force-align; "
            f"their word timings are approximate: {preview(widened)}"
        )
    if dropped:
        logger.warning(
            f"Dropping {len(dropped)}/{len(segments)} segment(s) the audio cannot carry "
            f"(too short for their text, or past the end of the recording): "
            f"{preview(dropped)}"
        )


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

    def __init__(self, model_name: str, device_id: int = -1, pretrained: bool = True) -> None:
        from exordium.utils.ckpt import build_hf_model

        self.model_name = model_name
        self.device = get_torch_device(device_id)
        self.tokenizer = tfm.AutoTokenizer.from_pretrained(model_name)
        self.model = build_hf_model(tfm.AutoModel, model_name, pretrained=pretrained)
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


class PooledTokenTextWrapper(TextModelWrapper):
    r"""Encoder wrapper exposing both token-level and pooled outputs via ``pooling``.

    A :class:`TextModelWrapper` whose ``preprocess → inference → __call__``
    pipeline is parameterised by a ``pooling`` mode, so one wrapper serves both
    an affective-computing fusion layer (which wants the token sequence) and a
    sentence-similarity consumer (which wants one vector per input):

    * ``pooling="none"`` — the raw token sequence ``last_hidden_state`` of shape
      ``(B, T, H)`` (token-level; feed to a cross-modal sequence model such as
      MulT/LinMulT alongside the ``attention_mask``).
    * ``pooling="mean"`` (default) — the attention-masked mean-pooled sentence
      embedding of shape ``(B, H)``.

    A single ``str`` is encoded as a batch of one, so one sentence yields
    ``(1, T, H)`` (token) or ``(1, H)`` (pooled); index ``[0]`` for the per-sample
    ``(T, H)`` / ``(H,)`` views. ``H`` is read from the model config into
    :attr:`hidden_size`, so it is never assumed.

    Subclasses only need to call ``super().__init__(model_name, ...)`` with their
    HuggingFace model id; the pooling API is inherited unchanged.

    Attributes:
        hidden_size: The backbone's hidden size ``H``, read from the model config.

    """

    def __init__(self, model_name: str, device_id: int = -1, pretrained: bool = True) -> None:
        super().__init__(model_name, device_id, pretrained=pretrained)
        self.hidden_size: int = self.model.config.hidden_size

    def inference(
        self,
        inputs: dict[str, torch.Tensor],
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Run the encoder forward pass and return token-level or pooled features.

        Args:
            inputs: Tokenizer output dict on ``self.device``.
            pooling: ``"none"`` returns the raw token sequence
                ``last_hidden_state`` of shape ``(B, T, H)`` (token-level, for a
                cross-modal sequence model). ``"mean"`` (default) returns the
                attention-masked mean-pooled sentence embedding of shape
                ``(B, H)``.

        Returns:
            Token embeddings ``(B, T, H)`` when ``pooling="none"``, or pooled
            sentence embeddings ``(B, H)`` when ``pooling="mean"``.

        Raises:
            ValueError: If ``pooling`` is not ``"none"`` or ``"mean"``.

        """
        last_hidden = self.model(**inputs).last_hidden_state
        if pooling == "none":
            return last_hidden
        if pooling == "mean":
            return self._mean_pool(last_hidden, inputs["attention_mask"])
        raise ValueError(f"Unknown pooling {pooling!r}; use 'none' or 'mean'.")

    def __call__(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool | str = True,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Tokenize and encode text, returning a feature tensor.

        Args:
            text: Single string or list of strings.
            max_length: Maximum sequence length. ``None`` uses tokenizer default.
            padding: Padding strategy. Default: ``True`` (pad to longest).
            pooling: Forwarded to :meth:`inference`. ``"mean"`` (default) returns
                pooled ``(B, H)``; ``"none"`` returns token-level ``(B, T, H)``.

        Returns:
            Feature tensor on ``self.device``: ``(B, H)`` pooled by default, or
            ``(B, T, H)`` token-level when ``pooling="none"``.

        """
        with torch.inference_mode():
            return self.inference(self.preprocess(text, max_length, padding), pooling=pooling)

    def predict(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool | str = True,
        pooling: str = "mean",
    ) -> np.ndarray:
        """Tokenize and encode text, returning a numpy feature array.

        Args:
            text: Single string or list of strings.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            pooling: Forwarded to :meth:`inference`. ``"mean"`` (default) returns
                pooled ``(B, H)``; ``"none"`` returns token-level ``(B, T, H)``.

        Returns:
            Feature array: ``(B, H)`` pooled by default, or ``(B, T, H)``
            token-level when ``pooling="none"``.

        """
        return self(text, max_length, padding, pooling=pooling).cpu().numpy()


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

    def min_align_samples(self, text: str) -> int:
        """Samples this backend needs to align *text*.

        The default counts non-whitespace characters as CTC tokens, which holds for the
        character-level wav2vec2 models both backends use.  Backends with a tokenizer to
        hand should override this and count exactly.

        Args:
            text: The text to be aligned.

        Returns:
            Minimum slice length in samples.

        """
        return wav2vec2_min_samples(sum(1 for c in text if not c.isspace()))

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

        Segments too short to align are widened by :func:`prepare_segments` rather than
        crashing the recording or being thrown away.  Because a widened window may
        overlap its neighbours, the merged stream is sorted by time at the end.

        Backends with a native batched segment API (e.g. whisperX) override this.

        Args:
            audio: Full audio — file path, numpy array, or torch tensor.
            segments: Timestamped transcript segments covering the audio.
            language: Language code or ``None`` for the default.

        Returns:
            Words in chronological order on the full-audio timeline.

        """
        waveform, sample_rate = to_mono_16k(audio)
        num_samples = waveform.numel()
        words: list[Word] = []
        for segment in prepare_segments(segments, self.min_align_samples, sample_rate, num_samples):
            start = max(0, int(segment.start * sample_rate))
            end = min(num_samples, int(segment.end * sample_rate))
            for word in self.align(waveform[start:end], segment.text, language=language):
                words.append(
                    Word(
                        text=word.text,
                        start=word.start + segment.start,
                        end=word.end + segment.start,
                        score=word.score,
                    )
                )
        words.sort(key=lambda w: (w.start, w.end))
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
