"""High-level speech transcription & alignment facade.

:class:`SpeechAlignmentPipeline` hides the choice of forced-alignment backend
(``torchaudio`` ``MMS_FA`` or whisperX) and the fuzzy-matching mechanics behind a
small set of methods that return **structured** results.  Downstream projects
consume those results (transcripts, timed words, metrics, per-segment decisions
with recovered start/end) and drive their own actions such as audio cutting.

Covers four common tasks:

1. :meth:`transcribe` — audio → transcript text.
2. :meth:`transcribe_words` — audio → word-level timestamps.
3. :meth:`evaluate_annotation` — audio + annotated text → closeness metrics.
4. :meth:`validate_segments` — audio + annotated ``(text, start, end)`` segments
   → per-segment accept/recut/drop decisions with recovered times.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from exordium.text.alignment import TorchaudioForcedAligner, WhisperWordTimestamper
from exordium.text.base import ForcedAligner, Word

if TYPE_CHECKING:
    from exordium.text.whisper import WhisperWrapper
from exordium.text.evaluation import (
    Embedder,
    SegmentValidation,
    TranscriptMetrics,
    build_embedder,
    evaluate_transcript,
    validate_segments,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""

_BACKENDS: frozenset[str] = frozenset({"torchaudio", "whisperx"})
"""Supported forced-alignment backends."""

AudioInput = Path | str | np.ndarray | torch.Tensor
"""Any audio form accepted by the wrappers: path, numpy array, or tensor."""


class SpeechAlignmentPipeline:
    """One entry point for transcription, word alignment, and re-alignment.

    The forced-alignment backend is selected once via *backend* and then hidden:
    every method returns plain data structures, never backend objects.

    Args:
        backend: ``"whisperx"`` (default; whisperX wav2vec2 alignment, ships with
            the ``text`` extra) or ``"torchaudio"`` (``MMS_FA``, no extra deps).
            Both yield the same :class:`~exordium.text.base.Word` stream.
        language: Language code used for ASR and alignment (e.g. ``"en"``).
        whisper_model: HuggingFace Whisper model id for transcription.
        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU/MPS.
        whisper: An existing :class:`WhisperWrapper` to reuse (e.g. to share one
            ASR model across two backends and avoid loading Whisper twice —
            Whisper is by far the largest model here). Created if omitted.
        semantic_model: Sentence-embedding model for the optional semantic
            similarity in :meth:`evaluate_annotation` (see
            :func:`~exordium.text.evaluation.build_embedder`); default
            ``"xlm-roberta"``. Loaded lazily only when ``semantic=True``.

    Example::

        pipe = SpeechAlignmentPipeline(backend="whisperx", device_id=0)
        text = pipe.transcribe("clip.wav")                       # case 1
        words = pipe.transcribe_words("clip.wav")                # case 2
        metrics = pipe.evaluate_annotation("clip.wav", annotated)  # case 3
        results = pipe.validate_segments("video.wav", segments)  # case 4

        # compare backends without loading Whisper twice:
        other = SpeechAlignmentPipeline(backend="torchaudio", whisper=pipe.whisper)

    """

    def __init__(
        self,
        backend: str = "whisperx",
        language: str = "en",
        whisper_model: str = "distil-whisper/distil-large-v3",
        device_id: int | None = 0,
        whisper: "WhisperWrapper | None" = None,
        semantic_model: str = "xlm-roberta",
        pretrained: bool = True,
    ) -> None:
        if backend not in _BACKENDS:
            # Validate before building anything: constructing Whisper first would
            # download gigabytes only to raise on a typo'd backend name.
            raise ValueError(f"Unknown backend {backend!r}; use one of {sorted(_BACKENDS)}.")

        self.backend = backend
        self.language = language
        self.device_id = device_id
        self.semantic_model = semantic_model
        self.pretrained = pretrained

        if whisper is None:
            from exordium.text.whisper import WhisperWrapper

            whisper = WhisperWrapper(
                model_name=whisper_model,
                device_id=device_id if device_id is not None else -1,
                pretrained=pretrained,
            )
        self.whisper = whisper
        aligner = self._build_aligner(backend, language, device_id, pretrained=pretrained)
        self._timestamper = WhisperWordTimestamper(whisper=self.whisper, aligner=aligner)
        self._embedder: Embedder | None = None

    @staticmethod
    def _build_aligner(
        backend: str, language: str, device_id: int | None, pretrained: bool = True
    ) -> ForcedAligner:
        """Instantiate the requested forced-alignment backend.

        ``pretrained=False`` reaches the torchaudio backend, which can build its
        architecture without a checkpoint. whisperX has no weight-free path, so it always
        loads its alignment model.
        """
        if backend == "torchaudio":
            return TorchaudioForcedAligner(device_id=device_id, pretrained=pretrained)
        if backend == "whisperx":
            from exordium.text.whisperx_align import WhisperxForcedAligner

            return WhisperxForcedAligner(language=language, device_id=device_id)
        raise ValueError(f"Unknown backend {backend!r}; use one of {sorted(_BACKENDS)}.")

    def transcribe(self, audio: AudioInput, language: str | None = None) -> str:
        """Case 1 — return the plain transcript for *audio*.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            language: Override the pipeline language for this call.

        Returns:
            The transcribed text.

        """
        return self.whisper(audio, language=language or self.language)

    def transcribe_words(self, audio: AudioInput, language: str | None = None) -> list[Word]:
        """Case 2 — return word-level timestamps for *audio*.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            language: Override the pipeline language for this call.

        Returns:
            Words with ``start``/``end``/``score`` in chronological order.

        """
        return self._timestamper.transcribe_words(audio, language=language or self.language)

    def evaluate_annotation(
        self,
        audio: AudioInput,
        annotated_text: str,
        language: str | None = None,
        *,
        semantic: bool = False,
    ) -> TranscriptMetrics:
        """Case 3 — score an annotated transcript against a fresh prediction.

        Transcribes *audio* and compares the result to *annotated_text*. Prefer
        the ``normalized_*`` fields of the result, which ignore number-format and
        contraction differences and reflect true recognition quality.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            annotated_text: The existing annotation to judge.
            language: Override the pipeline language for this call.
            semantic: If ``True``, also compute meaning-level cosine similarity
                using a BERT sentence embedder (loaded lazily on first use).

        Returns:
            :class:`TranscriptMetrics` (raw + normalized WER/CER/similarity, and
            ``semantic_similarity`` when *semantic* is set).

        """
        prediction = self.transcribe(audio, language=language)
        embedder = self._get_embedder() if semantic else None
        return evaluate_transcript(annotated_text, prediction, embedder=embedder)

    def _get_embedder(self) -> Embedder:
        """Lazily build the sentence embedder (``semantic_model``) for scoring."""
        if self._embedder is None:
            self._embedder = build_embedder(self.semantic_model, self.device_id)
        return self._embedder

    def validate_segments(
        self,
        audio: AudioInput,
        segments: list[tuple[str, float, float]],
        *,
        language: str | None = None,
        score_cutoff: float = 80.0,
        tolerance: float = 0.3,
    ) -> list[SegmentValidation]:
        """Case 4 — check annotated segments against a single ASR pass.

        Runs word-level ASR once over the full *audio*, then fuzzy-locates each
        annotated segment to decide accept / recut / drop and recover true times.

        Args:
            audio: Full-audio file path, numpy array, or torch tensor.
            segments: Annotated ``(text, start_sec, end_sec)`` tuples.
            language: Override the pipeline language for this call.
            score_cutoff: Minimum fuzzy score to consider a segment present.
            tolerance: Max absolute start/end offset (seconds) to accept as-is.

        Returns:
            One :class:`SegmentValidation` per segment, each carrying the
            decision and recovered start/end for your cutting step.

        """
        words = self.transcribe_words(audio, language=language)
        return validate_segments(segments, words, score_cutoff=score_cutoff, tolerance=tolerance)
