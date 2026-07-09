"""Dataset-agnostic fuzzy re-alignment of a known transcript to a word stream.

Given a word stream with timestamps (``list[Word]`` from any backend in
:mod:`exordium.text.alignment` / :mod:`exordium.text.whisperx_align`) and a
**known** segment transcript from any dataset, :func:`find_segment` locates where
that text occurs on the audio timeline and how well it matches.

This is the core of transcript-guided re-alignment: run open-vocabulary ASR once
over a full video, then fuzzy-search each dataset segment's annotated text inside
the word stream to recover its true start/end plus a coverage score.  Decide per
segment: high score → accept / re-cut with recovered times; low score → flag as
unrecoverable (wrong edit) and drop.

The module knows nothing about any specific dataset — callers supply their own
segment transcripts, so the same code adapts to CMU-MOSEI, MOSI, or any future
corpus.
"""

import logging
import re
import unicodedata

from rapidfuzz import fuzz

from exordium.text.base import SegmentMatch, Word

logger = logging.getLogger(__name__)
"""Module-level logger."""

_NON_ALNUM = re.compile(r"[^a-z0-9']+")
"""Matches runs of characters outside the normalized alphabet."""


def normalize_token(text: str) -> str:
    """Normalize text for matching: lowercase, de-accent, strip punctuation.

    Applies NFKD decomposition to drop diacritics, lowercases, and removes every
    character outside ``[a-z0-9']``.  Applied identically to both the query and
    each stream word so the fuzzy comparison is punctuation/casing insensitive.

    Args:
        text: Raw token or phrase.

    Returns:
        Normalized string (may be empty, e.g. for pure punctuation).

    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", stripped).strip()


def find_segment(
    query_text: str,
    words: list[Word],
    *,
    score_cutoff: float = 80.0,
) -> SegmentMatch | None:
    """Fuzzy-search *query_text* inside a timed *words* stream.

    Normalizes both sides, then finds the span of ``words`` whose concatenated
    text best matches the query (via :func:`rapidfuzz.fuzz.partial_ratio_alignment`,
    which is robust to insertions/deletions from ASR errors and differing
    segmentation).  Returns the recovered start/end and coverage score, or
    ``None`` when the best match falls below *score_cutoff*.

    Args:
        query_text: The known transcript to locate (from any dataset).
        words: Timed word stream to search within.
        score_cutoff: Minimum rapidfuzz score in ``[0, 100]`` to accept a match.
            Below this the segment is considered unrecoverable (``None``).

    Returns:
        A :class:`SegmentMatch` on the audio timeline, or ``None``.

    """
    query = normalize_token(query_text)
    if not query or not words:
        return None

    # Build the searchable haystack, tracking each char range back to its word.
    pieces: list[str] = []
    char_to_word: list[tuple[int, int, int]] = []  # (char_start, char_end, word_idx)
    cursor = 0
    for idx, word in enumerate(words):
        norm = normalize_token(word.text)
        if not norm:
            continue
        if pieces:
            cursor += 1  # the joining space
        pieces.append(norm)
        char_to_word.append((cursor, cursor + len(norm), idx))
        cursor += len(norm)

    if not pieces:
        return None
    haystack = " ".join(pieces)

    alignment = fuzz.partial_ratio_alignment(query, haystack, score_cutoff=score_cutoff)
    if alignment is None:
        return None

    first_idx, last_idx = _span_to_word_indices(
        alignment.dest_start, alignment.dest_end, char_to_word
    )
    if first_idx is None or last_idx is None:
        return None

    matched = words[first_idx : last_idx + 1]
    return SegmentMatch(
        text=" ".join(w.text for w in matched),
        start=matched[0].start,
        end=matched[-1].end,
        score=float(alignment.score),
        word_start_idx=first_idx,
        word_end_idx=last_idx,
    )


def find_segments(
    query_texts: list[str],
    words: list[Word],
    *,
    score_cutoff: float = 80.0,
) -> list[SegmentMatch | None]:
    """Batch-apply :func:`find_segment` to many segment transcripts.

    Args:
        query_texts: Known transcripts, e.g. one per dataset segment.
        words: Shared timed word stream (one full-video ASR pass).
        score_cutoff: Minimum score to accept each match.

    Returns:
        One result per query, in order; ``None`` where a segment is
        unrecoverable.

    """
    return [find_segment(q, words, score_cutoff=score_cutoff) for q in query_texts]


def _span_to_word_indices(
    char_start: int,
    char_end: int,
    char_to_word: list[tuple[int, int, int]],
) -> tuple[int | None, int | None]:
    """Map a matched ``[char_start, char_end)`` haystack range to word indices."""
    first_idx: int | None = None
    last_idx: int | None = None
    for c_start, c_end, word_idx in char_to_word:
        # Overlap between the word's char span and the matched range.
        if c_end > char_start and c_start < char_end:
            if first_idx is None:
                first_idx = word_idx
            last_idx = word_idx
    return first_idx, last_idx
