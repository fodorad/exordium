"""Word-level forced alignment and open-vocabulary word timestamps.

Two public classes:

* :class:`TorchaudioForcedAligner` — aligns a **known** transcript to audio
  using ``torchaudio``'s multilingual ``MMS_FA`` bundle.  No extra dependencies
  beyond the pinned ``torchaudio``.
* :class:`WhisperWordTimestamper` — transcribes **raw** audio with
  :class:`~exordium.text.whisper.WhisperWrapper` and then forced-aligns the
  result, yielding open-vocabulary word timestamps from audio alone.

Both return ``list[Word]`` (see :class:`~exordium.text.base.Word`), the common
currency consumed by :mod:`exordium.text.transcript_align`.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torchaudio

from exordium.audio.io import load_audio
from exordium.text.base import (
    MIN_ALIGN_SAMPLES,
    ForcedAligner,
    Word,
    WordTimestamper,
    wav2vec2_min_samples,
)
from exordium.utils.device import get_torch_device

if TYPE_CHECKING:
    from exordium.text.whisper import WhisperWrapper

logger = logging.getLogger(__name__)
"""Module-level logger."""

_WORD_RE = re.compile(r"[^a-z']+")
"""Matches runs of characters not in the ``MMS_FA`` latin/apostrophe alphabet."""


def normalize_words(transcript: str) -> list[str]:
    """Normalize a transcript into aligner-friendly lowercase word tokens.

    Lowercases, strips diacritics (NFKD), and removes every character outside
    ``[a-z']`` so the tokens match the ``MMS_FA`` dictionary.  Empty tokens are
    dropped.  Used by :class:`TorchaudioForcedAligner`.

    Args:
        transcript: Raw transcript text.

    Returns:
        List of normalized word tokens (possibly empty).

    """
    decomposed = unicodedata.normalize("NFKD", transcript.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    tokens = (_WORD_RE.sub("", tok) for tok in stripped.split())
    return [tok for tok in tokens if tok]


class TorchaudioForcedAligner(ForcedAligner):
    """Forced aligner backed by ``torchaudio.pipelines.MMS_FA``.

    ``MMS_FA`` is a Massively Multilingual Speech CTC alignment model covering
    1000+ languages; it produces accurate word timings by aligning the known
    transcript against the audio (the same idea whisperX uses, but with a single
    multilingual model and no extra dependencies).

    Args:
        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU/MPS.

    Example::

        aligner = TorchaudioForcedAligner(device_id=None)
        words = aligner.align("audio.wav", "hey guys what do you want to eat")
        for w in words:
            print(w.text, w.start, w.end, w.score)

    """

    def __init__(self, device_id: int | None = 0, pretrained: bool = True) -> None:
        self.device = get_torch_device(device_id)
        bundle = torchaudio.pipelines.MMS_FA
        self.sample_rate: int = int(bundle.sample_rate)
        if pretrained:
            logger.info(f"Loading MMS_FA forced aligner on {self.device}...")
            model = bundle.get_model()
        else:
            # The bundle has no public weight-free path, but it builds the architecture
            # from these params before loading the checkpoint — so we can too.
            logger.info("Building MMS_FA architecture with random weights (no checkpoint).")
            from torchaudio.models import wav2vec2_model

            model = wav2vec2_model(**bundle._params)  # noqa: SLF001 - no public equivalent
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def min_align_samples(self, text: str) -> int:
        """Samples ``MMS_FA`` needs to align *text*, counted with its own tokenizer.

        Overrides the base class's character estimate: ``normalize_words`` strips
        everything outside the ``MMS_FA`` alphabet, so counting raw characters would
        demand more audio than the model actually needs.

        Args:
            text: The text to be aligned.

        Returns:
            Minimum slice length in samples.

        """
        words = normalize_words(text)
        if not words:
            return MIN_ALIGN_SAMPLES
        tokens = cast("list[list[int]]", self.tokenizer(words))
        return wav2vec2_min_samples(sum(len(word_tokens) for word_tokens in tokens))

    def align(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        transcript: str,
        language: str | None = None,
    ) -> list[Word]:
        """Align *transcript* to *audio* and return timed :class:`Word` objects.

        Args:
            audio: Audio file path, numpy array, or torch tensor (16 kHz mono).
            transcript: The known transcript for this audio.
            language: Unused — ``MMS_FA`` is multilingual. Accepted for
                interface compatibility with :class:`ForcedAligner`.

        Returns:
            Words in chronological order. Empty if *transcript* has no alignable
            tokens, or if the audio is too short to carry them — see
            :func:`~exordium.text.base.wav2vec2_min_samples`.  Callers aligning
            whole recordings should go through :meth:`align_segments`, which
            widens short windows instead of giving up on them.

        """
        del language  # MMS_FA is a single multilingual model.
        waveform = self._to_waveform(audio)
        words = normalize_words(transcript)
        if not words:
            logger.warning("Transcript has no alignable tokens; returning [].")
            return []

        num_samples = waveform.size(1)
        needed = self.min_align_samples(transcript)
        if num_samples < needed:
            # Too few frames for the tokens: MMS_FA's conv stack or forced_align raises.
            logger.warning(
                f"Audio is {num_samples} samples but {' '.join(words)!r} needs {needed} "
                "to force-align; returning []."
            )
            return []

        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))
        tokens = cast("list[list[int]]", self.tokenizer(words))
        token_spans = self.aligner(emission[0], tokens)

        num_frames = emission.size(1)
        # Seconds per emission frame: (samples / frame) / (samples / second).
        seconds_per_frame = waveform.size(1) / num_frames / self.sample_rate

        return [
            Word(
                text=word,
                start=spans[0].start * seconds_per_frame,
                end=spans[-1].end * seconds_per_frame,
                score=_span_score(spans),
            )
            for word, spans in zip(words, token_spans, strict=True)
        ]

    def _to_waveform(self, audio: Path | str | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return a ``(1, T)`` float32 waveform at the bundle sample rate."""
        if isinstance(audio, (str, Path)):
            waveform, _ = load_audio(
                audio, target_sample_rate=self.sample_rate, mono=True, squeeze=True
            )
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).squeeze()
        else:
            waveform = audio.detach().cpu().float().squeeze()
        return waveform.reshape(1, -1)


def _span_score(spans: list) -> float:
    """Length-weighted mean confidence over a word's token spans."""
    total = sum(s.end - s.start for s in spans)
    if total == 0:
        return 0.0
    return float(sum(s.score * (s.end - s.start) for s in spans) / total)


class WhisperWordTimestamper(WordTimestamper):
    """Open-vocabulary word timestamps: Whisper ASR + forced alignment.

    Transcribes raw audio with :class:`~exordium.text.whisper.WhisperWrapper`,
    then forced-aligns the transcript with a pluggable
    :class:`~exordium.text.base.ForcedAligner` (``MMS_FA`` by default, or
    :class:`~exordium.text.whisperx_align.WhisperxForcedAligner`).  This mirrors
    the whisperX architecture (ASR → wav2vec2 alignment) while running on the
    repo's pinned ``torch`` stack.

    Args:
        whisper: A ready :class:`WhisperWrapper`; created with ``device_id`` if
            omitted.
        aligner: A ready :class:`ForcedAligner`; a
            :class:`TorchaudioForcedAligner` is created with ``device_id`` if
            omitted.
        device_id: Device used only when *whisper* / *aligner* are not supplied.

    Example::

        ts = WhisperWordTimestamper(device_id=None)
        words = ts.transcribe_words("audio.wav")

    """

    def __init__(
        self,
        whisper: "WhisperWrapper | None" = None,
        aligner: ForcedAligner | None = None,
        device_id: int | None = 0,
        pretrained: bool = True,
    ) -> None:
        if whisper is None:
            from exordium.text.whisper import WhisperWrapper

            whisper = WhisperWrapper(
                device_id=device_id if device_id is not None else -1, pretrained=pretrained
            )
        self.whisper = whisper
        self.aligner = (
            aligner
            if aligner is not None
            else TorchaudioForcedAligner(device_id, pretrained=pretrained)
        )

    def transcribe_words(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        language: str | None = None,
    ) -> list[Word]:
        """Transcribe *audio* and return forced-aligned :class:`Word` objects.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            language: Language code forwarded to both ASR and alignment, or
                ``None`` for auto/default.

        Returns:
            Words in chronological order with ``start``/``end``/``score``.

        """
        # Segment-wise ASR + alignment: correct for clips of any length. Whisper
        # decodes long-form (>30 s) instead of silently cropping, and each
        # segment is aligned on its own slice so long files stay memory-bounded.
        segments = self.whisper.transcribe_segments(audio, language=language)
        if not segments:
            return []
        return self.aligner.align_segments(audio, segments, language=language)
