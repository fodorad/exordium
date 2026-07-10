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

import bisect
import logging
import re
import unicodedata
from dataclasses import dataclass

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
    return _search(normalize_token(query_text), build_word_index(words), words, score_cutoff)


def find_segments(
    query_texts: list[str],
    words: list[Word],
    *,
    score_cutoff: float = 80.0,
) -> list[SegmentMatch | None]:
    """Locate many segment transcripts in one shared *words* stream.

    The normalized haystack is built **once** and reused for every query, so a
    long recording searched for many segments costs ``O(N + Q·search)`` rather
    than re-normalizing all ``N`` words for each of the ``Q`` queries.

    Args:
        query_texts: Known transcripts, e.g. one per dataset segment.
        words: Shared timed word stream (one full-video ASR pass).
        score_cutoff: Minimum score to accept each match.

    Returns:
        One result per query, in order; ``None`` where a segment is
        unrecoverable.

    """
    index = build_word_index(words)
    return [_search(normalize_token(q), index, words, score_cutoff) for q in query_texts]


@dataclass(frozen=True)
class WordIndex:
    """Precomputed search structure over a timed word stream.

    Built once by :func:`build_word_index` and reused across queries.

    Attributes:
        haystack: The normalized words joined by single spaces.
        starts: Character offset where each indexed word begins in ``haystack``.
        ends: Character offset just past each indexed word.
        word_ids: Index into the original ``words`` list for each entry.

    """

    haystack: str
    starts: list[int]
    ends: list[int]
    word_ids: list[int]


def build_word_index(words: list[Word]) -> WordIndex:
    """Normalize *words* once into a reusable :class:`WordIndex`.

    Words that normalize to nothing (pure punctuation) are skipped, while
    ``word_ids`` keeps the mapping back to their original positions.

    Args:
        words: Timed word stream.

    Returns:
        The searchable index (possibly empty).

    """
    pieces: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    word_ids: list[int] = []
    cursor = 0
    for idx, word in enumerate(words):
        norm = normalize_token(word.text)
        if not norm:
            continue
        if pieces:
            cursor += 1  # the joining space
        pieces.append(norm)
        starts.append(cursor)
        ends.append(cursor + len(norm))
        word_ids.append(idx)
        cursor += len(norm)
    return WordIndex(haystack=" ".join(pieces), starts=starts, ends=ends, word_ids=word_ids)


def _search(
    query: str,
    index: WordIndex,
    words: list[Word],
    score_cutoff: float,
) -> SegmentMatch | None:
    """Fuzzy-match a normalized *query* against a prebuilt *index*."""
    if not query or not index.haystack:
        return None

    alignment = fuzz.partial_ratio_alignment(query, index.haystack, score_cutoff=score_cutoff)
    if alignment is None:
        return None

    first_idx, last_idx = _span_to_word_indices(alignment.dest_start, alignment.dest_end, index)
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


def _span_to_word_indices(
    char_start: int,
    char_end: int,
    index: WordIndex,
) -> tuple[int | None, int | None]:
    """Map a matched ``[char_start, char_end)`` haystack range to word indices.

    Word char spans are sorted and non-overlapping, so the first/last overlapping
    entries are found by binary search instead of scanning every word.
    """
    # First entry whose end is past char_start; last entry whose start precedes char_end.
    first = bisect.bisect_right(index.ends, char_start)
    last = bisect.bisect_left(index.starts, char_end) - 1
    if first > last or first >= len(index.word_ids) or last < 0:
        return None, None
    return index.word_ids[first], index.word_ids[last]
