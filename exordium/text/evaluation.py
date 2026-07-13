"""Structured transcript-quality metrics and segment-annotation validation.

Pure logic (no models): turns a word stream and known annotations into
structured, serialisable results that downstream code (e.g. an audio cutter in
another project) can act on directly.

* :func:`evaluate_transcript` → :class:`TranscriptMetrics` — how close an
  annotated transcript is to an ASR prediction (WER / CER / similarity).
* :func:`validate_segment` / :func:`validate_segments` → :class:`SegmentValidation`
  — whether an annotated ``(text, start, end)`` segment is where the annotation
  claims, and, when not, the recovered start/end to re-cut with.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from exordium.text.base import SegmentMatch, Word
from exordium.text.transcript_align import find_segment, find_segments, normalize_token

logger = logging.getLogger(__name__)
"""Module-level logger."""

Embedder = Callable[[list[str]], np.ndarray]
"""Maps a list of texts to a ``(N, H)`` embedding array (for semantic similarity)."""

DEFAULT_MAX_NORMALIZED_WER = 0.20
"""Default accept threshold: normalized WER at/under this is considered a match.

Heuristic — there is no universal ASR-quality cutoff. Rough guidance: ``<0.10``
excellent, ``0.10–0.20`` good, ``0.20–0.35`` fair, ``>0.5`` poor. The default
``0.20`` accepts up to the good/fair boundary; raise to ``~0.35`` to also accept
"fair" annotations. Tune per dataset.
"""

DEFAULT_MIN_SEMANTIC_SIMILARITY = 0.70
"""Default rescue threshold: semantic cosine at/over this counts as same-meaning.

Heuristic for the ``xml-roberta`` sentence-transformer: ``>0.85`` near-identical
meaning, ``0.70–0.85`` clearly related/paraphrase, ``<0.5`` different. The default
``0.70`` accepts clearly-related/paraphrase pairs. Tune per dataset.
"""

_NORMALIZERS: dict[str, object] = {}
"""Lazily-built Whisper text-normalizer singletons, keyed by kind."""


def normalize_asr(text: str, kind: str = "english") -> str:
    """Apply Whisper's standard ASR text normalizer.

    This is **not** a hand-rolled normalizer: it is the normalizer published with
    OpenAI Whisper (Whisper paper, Appendix C "Text Standardization") and shipped
    in ``transformers`` — the same routine the HuggingFace Open ASR Leaderboard
    uses, so WER/CER are comparable to the literature. It canonicalizes numbers
    (``"ten thousand"`` and ``"10,000"`` → ``"10000"``), expands contractions,
    and strips casing/punctuation.

    Args:
        text: Raw text to normalize.
        kind: ``"english"`` (``EnglishTextNormalizer`` — number/contraction/
            spelling aware, English only) or ``"basic"`` (``BasicTextNormalizer``
            — language-agnostic: lowercasing, punctuation/symbol removal).

    Returns:
        The normalized text.

    """
    if kind not in _NORMALIZERS:
        from transformers.models.whisper.english_normalizer import (
            BasicTextNormalizer,
            EnglishTextNormalizer,
        )

        if kind == "english":
            _NORMALIZERS[kind] = EnglishTextNormalizer({})
        elif kind == "basic":
            _NORMALIZERS[kind] = BasicTextNormalizer()
        else:
            raise ValueError(f"Unknown normalizer {kind!r}; use 'english' or 'basic'.")
    normalizer = cast("Callable[[str], str]", _NORMALIZERS[kind])
    return normalizer(text)


def _score(ref_norm: str, hyp_norm: str) -> tuple[float, float, float]:
    """Return ``(wer, cer, similarity)`` for two already-normalized strings."""
    ref_words, hyp_words = ref_norm.split(), hyp_norm.split()
    wer = Levenshtein.distance(ref_words, hyp_words) / max(len(ref_words), 1)
    cer = Levenshtein.distance(ref_norm, hyp_norm) / max(len(ref_norm), 1)
    similarity = float(fuzz.ratio(ref_norm, hyp_norm))
    return wer, cer, similarity


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, in ``[-1, 1]``."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


@dataclass
class TranscriptMetrics:
    """Closeness between a reference (annotation) and a hypothesis (ASR).

    Two families of metrics are reported:

    * **Raw** (``wer`` / ``cer`` / ``similarity``) — on lightly normalized text
      (lowercased, de-accented, punctuation stripped). Apostrophes and digits are
      kept, so ``"it's"`` ≠ ``"its"`` and ``"ten thousand"`` ≠ ``"10,000"`` count
      as errors. Sensitive to formatting.
    * **Normalized** (``normalized_*``) — on Whisper's ``EnglishTextNormalizer``,
      which canonicalizes numbers and contractions. This reflects *true*
      recognition quality by ignoring cosmetic differences. **Prefer this** to
      judge whether an annotation is fine.

    WER/CER: ``0`` = identical, lower is better (can exceed ``1`` if the
    hypothesis is much longer). Similarity: ``0–100``, higher is better.

    Attributes:
        wer: Raw word error rate.
        cer: Raw character error rate.
        similarity: Raw rapidfuzz ratio in ``[0, 100]``.
        normalized_wer: WER after Whisper normalization (fair recognition error).
        normalized_cer: CER after Whisper normalization.
        normalized_similarity: rapidfuzz ratio after Whisper normalization.
        semantic_similarity: Meaning-level cosine in ``[-1, 1]`` (higher = closer
            in meaning), or ``None`` when no embedder was supplied.
        reference: The original reference text.
        hypothesis: The original hypothesis text.

    """

    wer: float
    cer: float
    similarity: float
    normalized_wer: float
    normalized_cer: float
    normalized_similarity: float
    semantic_similarity: float | None
    reference: str
    hypothesis: str


def evaluate_transcript(
    reference: str,
    hypothesis: str,
    *,
    embedder: Embedder | None = None,
    normalizer: str = "english",
) -> TranscriptMetrics:
    """Score how closely *hypothesis* matches *reference*.

    Judges an annotated transcript against an ASR prediction of the same audio.
    Reports both raw and Whisper-normalized WER/CER/similarity; prefer the
    ``normalized_*`` values, which ignore number-format and contraction
    differences and thus reflect true recognition quality.

    Args:
        reference: Known/annotated transcript.
        hypothesis: Predicted transcript (e.g. from Whisper).
        embedder: Optional callable mapping ``[reference, hypothesis]`` to a
            ``(2, H)`` embedding array; when given, ``semantic_similarity`` is
            the cosine between the two rows.
        normalizer: Whisper normalizer for the ``normalized_*`` metrics —
            ``"english"`` or ``"basic"`` (see :func:`normalize_asr`).

    Returns:
        A :class:`TranscriptMetrics` with raw, normalized, and (optional)
        semantic scores.

    """
    wer, cer, similarity = _score(normalize_token(reference), normalize_token(hypothesis))
    nwer, ncer, nsim = _score(
        normalize_asr(reference, normalizer), normalize_asr(hypothesis, normalizer)
    )

    semantic: float | None = None
    if embedder is not None:
        emb = np.asarray(embedder([reference, hypothesis]), dtype=float)
        semantic = _cosine(emb[0], emb[1])

    return TranscriptMetrics(
        wer=wer,
        cer=cer,
        similarity=similarity,
        normalized_wer=nwer,
        normalized_cer=ncer,
        normalized_similarity=nsim,
        semantic_similarity=semantic,
        reference=reference,
        hypothesis=hypothesis,
    )


def is_acceptable(
    metrics: TranscriptMetrics,
    *,
    max_normalized_wer: float = DEFAULT_MAX_NORMALIZED_WER,
    min_semantic_similarity: float = DEFAULT_MIN_SEMANTIC_SIMILARITY,
) -> bool:
    """Heuristic accept/reject for an annotation given its :class:`TranscriptMetrics`.

    An annotation is accepted when it is lexically close **or** semantically
    equivalent to the prediction:

    * ``normalized_wer <= max_normalized_wer`` (primary, standard ASR criterion), or
    * ``semantic_similarity >= min_semantic_similarity`` (rescues paraphrases that
      score a high WER but mean the same thing — only when a semantic score exists).

    The thresholds are **heuristic defaults** (see
    :data:`DEFAULT_MAX_NORMALIZED_WER` / :data:`DEFAULT_MIN_SEMANTIC_SIMILARITY`);
    pass your own to match your dataset's tolerance. Judge on the ``normalized_*``
    metrics, not the raw ones.

    Args:
        metrics: Result of :func:`evaluate_transcript` / :meth:`TranscriptEvaluator.evaluate`.
        max_normalized_wer: Highest normalized WER still accepted.
        min_semantic_similarity: Semantic cosine at/above which a high-WER pair is
            still accepted.

    Returns:
        ``True`` if the annotation should be trusted, else ``False``.

    """
    if metrics.normalized_wer <= max_normalized_wer:
        return True
    return (
        metrics.semantic_similarity is not None
        and metrics.semantic_similarity >= min_semantic_similarity
    )


def build_embedder(
    model: str = "xml-roberta", device_id: int | None = -1, pretrained: bool = True
) -> Embedder:
    """Build a sentence embedder for :attr:`TranscriptMetrics.semantic_similarity`.

    Args:
        model: One of ``"xml-roberta"`` (default;
            :class:`~exordium.text.xml_roberta.XmlRobertaWrapper`, a multilingual
            sentence-transformer fine-tuned for semantic similarity — recommended),
            ``"roberta"`` (:class:`~exordium.text.roberta.RobertaWrapper`, CLS
            token), or ``"bert"`` (:class:`~exordium.text.bert.BertWrapper`, CLS
            token).
        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU/MPS.
        pretrained: ``True`` (default) loads the real weights. ``False`` builds the
            architecture with random weights — embeddings are then meaningless, but the
            shapes and call contract hold (used by the test suite to avoid downloads).

    Returns:
        A callable mapping a list of texts to an ``(N, H)`` embedding array.

    """
    name = model.lower()
    dev = device_id if device_id is not None else -1
    if name in ("xml-roberta", "xlm-roberta"):
        from exordium.text.xml_roberta import XmlRobertaWrapper

        wrapper = XmlRobertaWrapper(device_id=dev, pretrained=pretrained)
        return lambda texts: wrapper(texts).detach().cpu().numpy()  # already (N, 768)
    if name == "roberta":
        from exordium.text.roberta import RobertaWrapper

        rob = RobertaWrapper(device_id=dev, pretrained=pretrained)
        return lambda texts: rob(texts)[:, 0].detach().cpu().numpy()  # <s> token
    if name == "bert":
        from exordium.text.bert import BertWrapper

        bert = BertWrapper(device_id=dev, pretrained=pretrained)
        return lambda texts: bert(texts)[:, 0].detach().cpu().numpy()  # [CLS] token
    raise ValueError(f"Unknown semantic model {model!r}; use 'xml-roberta', 'roberta', or 'bert'.")


class TranscriptEvaluator:
    """Standalone transcript comparator — two strings in, metrics out.

    Independent of audio, ASR, and the alignment pipeline: call it whenever you
    have a reference and a hypothesis string. It computes raw and Whisper
    -normalized WER/CER/similarity, plus (unless disabled) a semantic similarity
    from a sentence-embedding model loaded lazily on first use.

    Args:
        semantic_model: Embedding model for semantic similarity
            (see :func:`build_embedder`), or ``None`` to skip it entirely (then
            no model is ever loaded — pure-text metrics only).
        device_id: Device index for the embedding model.

    Example::

        evaluator = TranscriptEvaluator()  # xml-roberta semantic
        metrics = evaluator.evaluate(annotation_text, predicted_text)
        print(metrics.normalized_wer, metrics.semantic_similarity)

    """

    def __init__(
        self,
        semantic_model: str | None = "xml-roberta",
        device_id: int | None = -1,
        normalizer: str = "english",
        pretrained: bool = True,
    ) -> None:
        self.semantic_model = semantic_model
        self.device_id = device_id
        self.normalizer = normalizer
        self.pretrained = pretrained
        self._embedder: Embedder | None = None

    def evaluate(self, reference: str, hypothesis: str) -> TranscriptMetrics:
        """Compare *reference* and *hypothesis*; see :func:`evaluate_transcript`."""
        return evaluate_transcript(
            reference, hypothesis, embedder=self._get_embedder(), normalizer=self.normalizer
        )

    def _get_embedder(self) -> Embedder | None:
        """Lazily build the semantic embedder (``None`` if disabled)."""
        if self.semantic_model is None:
            return None
        if self._embedder is None:
            self._embedder = build_embedder(
                self.semantic_model, self.device_id, pretrained=self.pretrained
            )
        return self._embedder


@dataclass
class SegmentValidation:
    """Result of checking one annotated segment against a timed word stream.

    ``recovered_start`` / ``recovered_end`` are the precise times where the
    annotated text actually occurs on the current audio timeline — use them as
    the corrected cut points whenever ``decision`` is not ``"drop"``.

    Attributes:
        text: The annotated segment text.
        annotated_start: Segment start as claimed by the annotation (seconds).
        annotated_end: Segment end as claimed by the annotation (seconds).
        recovered_start: True start found in the word stream, or ``None`` if the
            text was not locatable.
        recovered_end: True end found in the word stream, or ``None``.
        start_offset: ``recovered_start - annotated_start`` (``None`` if dropped).
        end_offset: ``recovered_end - annotated_end`` (``None`` if dropped).
        score: Fuzzy match coverage in ``[0, 100]`` (``0`` if dropped).
        decision: ``"accept"`` (annotation within tolerance),
            ``"recut"`` (found but shifted — re-cut with recovered times), or
            ``"drop"`` (text not found — likely a wrong edit).
        match: The underlying :class:`SegmentMatch`, or ``None`` if dropped.

    """

    text: str
    annotated_start: float
    annotated_end: float
    recovered_start: float | None
    recovered_end: float | None
    start_offset: float | None
    end_offset: float | None
    score: float
    decision: str
    match: SegmentMatch | None


def validate_segment(
    text: str,
    words: list[Word],
    annotated_start: float,
    annotated_end: float,
    *,
    score_cutoff: float = 80.0,
    tolerance: float = 0.3,
) -> SegmentValidation:
    """Check whether an annotated segment sits where the annotation claims.

    Fuzzy-searches *text* in the *words* stream and compares the recovered span
    to ``annotated_start``/``annotated_end``:

    * text not found (below *score_cutoff*) → ``decision="drop"``.
    * found and both offsets within *tolerance* → ``decision="accept"``.
    * found but shifted → ``decision="recut"`` (use ``recovered_start``/
      ``recovered_end`` as the corrected cut points).

    Args:
        text: The annotated segment transcript.
        words: Timed word stream for the full audio (one ASR pass).
        annotated_start: Annotated segment start in seconds.
        annotated_end: Annotated segment end in seconds.
        score_cutoff: Minimum fuzzy score to consider the text present.
        tolerance: Max absolute start/end offset (seconds) to accept as-is.

    Returns:
        A :class:`SegmentValidation` describing the decision and recovered times.

    """
    match = find_segment(text, words, score_cutoff=score_cutoff)
    return _decide(text, annotated_start, annotated_end, match, tolerance)


def _decide(
    text: str,
    annotated_start: float,
    annotated_end: float,
    match: SegmentMatch | None,
    tolerance: float,
) -> SegmentValidation:
    """Turn a (possibly missing) fuzzy match into an accept/recut/drop decision."""
    if match is None:
        return SegmentValidation(
            text=text,
            annotated_start=annotated_start,
            annotated_end=annotated_end,
            recovered_start=None,
            recovered_end=None,
            start_offset=None,
            end_offset=None,
            score=0.0,
            decision="drop",
            match=None,
        )

    start_offset = match.start - annotated_start
    end_offset = match.end - annotated_end
    within = abs(start_offset) <= tolerance and abs(end_offset) <= tolerance
    return SegmentValidation(
        text=text,
        annotated_start=annotated_start,
        annotated_end=annotated_end,
        recovered_start=match.start,
        recovered_end=match.end,
        start_offset=start_offset,
        end_offset=end_offset,
        score=match.score,
        decision="accept" if within else "recut",
        match=match,
    )


def validate_segments(
    segments: list[tuple[str, float, float]],
    words: list[Word],
    *,
    score_cutoff: float = 80.0,
    tolerance: float = 0.3,
) -> list[SegmentValidation]:
    """Batch-validate ``(text, start, end)`` segments against one word stream.

    The word stream is normalized once and reused for every segment, so long
    recordings with many annotations stay cheap.

    Args:
        segments: Annotated segments as ``(text, start_sec, end_sec)`` tuples.
        words: Shared timed word stream for the full audio.
        score_cutoff: Minimum fuzzy score to consider a segment present.
        tolerance: Max absolute start/end offset (seconds) to accept as-is.

    Returns:
        One :class:`SegmentValidation` per input segment, in order.

    """
    matches = find_segments([text for text, _, _ in segments], words, score_cutoff=score_cutoff)
    return [
        _decide(text, start, end, match, tolerance)
        for (text, start, end), match in zip(segments, matches, strict=True)
    ]
