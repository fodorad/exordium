"""whisperX wav2vec2 forced alignment (ships with the ``text`` extra).

Wraps whisperX's per-language wav2vec2 CTC alignment so a **known** transcript
can be aligned to audio, returning the same ``list[Word]`` as the built-in
:class:`~exordium.text.alignment.TorchaudioForcedAligner`.  Having both lets you
cross-check word timings from two independent aligners (useful when validating
dataset timestamps such as CMU-MOSEI).

.. note::

   Only whisperX's **alignment** stage is used here.  Its faster-whisper ASR /
   pyannote VAD path relies on ``torchaudio`` APIs removed in the version this
   repo pins, so transcription is done by
   :class:`~exordium.text.whisper.WhisperWrapper` instead (see
   :class:`~exordium.text.alignment.WhisperWordTimestamper`).

whisperX ships with the ``text`` extra::

    uv sync --extra text
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch

from exordium.audio.io import load_audio
from exordium.text.base import ForcedAligner, Segment, Word
from exordium.utils.device import get_torch_device

logger = logging.getLogger(__name__)
"""Module-level logger."""

try:
    import whisperx

    _WHISPERX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    whisperx = None  # ty: ignore[invalid-assignment]
    _WHISPERX_AVAILABLE = False


class WhisperxForcedAligner(ForcedAligner):
    """Forced aligner backed by whisperX's wav2vec2 alignment models.

    Loads (and caches) a language-specific alignment model on first use, then
    aligns the known transcript against the audio.

    Args:
        language: Language code for the alignment model (e.g. ``"en"``).
        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU/MPS.

    Raises:
        ImportError: If whisperX (the ``text`` extra) is not installed.

    Example::

        aligner = WhisperxForcedAligner(language="en", device_id=None)
        words = aligner.align("audio.wav", "hey guys what do you want to eat")

    """

    def __init__(self, language: str = "en", device_id: int | None = 0) -> None:
        if not _WHISPERX_AVAILABLE:
            raise ImportError(
                "whisperX is not installed. Install the text extra with "
                "`uv sync --extra text` to use WhisperxForcedAligner."
            )
        self.language = language
        self.device = get_torch_device(device_id)
        self._device_str = "cuda" if self.device.type == "cuda" else "cpu"
        logger.info(f"Loading whisperX align model ({language}) on {self._device_str}...")
        self.model, self.metadata = whisperx.load_align_model(
            language_code=language, device=self._device_str
        )

    def align(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        transcript: str,
        language: str | None = None,
    ) -> list[Word]:
        """Align *transcript* to *audio* using whisperX and return timed words.

        Args:
            audio: Audio file path, numpy array, or torch tensor (16 kHz mono).
            transcript: The known transcript for this audio.
            language: Optional language code; must match the model's language if
                given (a mismatch is logged, not enforced).

        Returns:
            Words in chronological order. Empty if *transcript* is blank.

        """
        self._warn_language(language)
        if not transcript.strip():
            return []

        waveform, sr = self._to_array(audio)
        duration = len(waveform) / sr
        return self._run([{"text": transcript, "start": 0.0, "end": duration}], waveform)

    def align_segments(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        segments: list[Segment],
        language: str | None = None,
    ) -> list[Word]:
        """Align all *segments* in one whisperX call (native batched path).

        whisperX slices the audio per segment internally and returns word times
        already on the full-audio timeline, so long recordings stay bounded
        without the per-segment Python loop of the base implementation.

        Args:
            audio: Full audio — file path, numpy array, or torch tensor.
            segments: Timestamped transcript segments covering the audio.
            language: Optional language code (must match the loaded model).

        Returns:
            Words in chronological order on the full-audio timeline.

        """
        self._warn_language(language)
        payload = [
            {"text": s.text, "start": float(s.start), "end": float(s.end)}
            for s in segments
            if s.text.strip()
        ]
        if not payload:
            return []
        waveform, _ = self._to_array(audio)
        return self._run(payload, waveform)

    def _warn_language(self, language: str | None) -> None:
        """Log when a caller asks for a language the loaded model was not built for."""
        if language is not None and language != self.language:
            logger.warning(
                f"Requested language {language!r} != model language {self.language!r}; "
                "reload WhisperxForcedAligner with the desired language."
            )

    def _run(self, segments: list[dict], waveform: np.ndarray) -> list[Word]:
        """Run whisperX alignment over *segments* and convert to :class:`Word`."""
        result = whisperx.align(
            segments,
            self.model,
            self.metadata,
            waveform,
            self._device_str,
            return_char_alignments=False,
        )
        words = []
        for w in result["word_segments"]:
            start, end = w.get("start"), w.get("end")
            # whisperX leaves unalignable words without (or with NaN) times.
            if start is None or end is None:
                continue
            start, end = float(start), float(end)
            if math.isnan(start) or math.isnan(end):
                continue
            words.append(
                Word(text=str(w["word"]), start=start, end=end, score=float(w.get("score", 1.0)))
            )
        return words

    @staticmethod
    def _to_array(audio: Path | str | np.ndarray | torch.Tensor) -> tuple[np.ndarray, int]:
        """Return a float32 mono waveform array at 16 kHz and its sample rate."""
        if isinstance(audio, (str, Path)):
            waveform, sr = load_audio(audio, target_sample_rate=16000, mono=True, squeeze=True)
            return waveform.numpy().astype(np.float32), sr
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().float().squeeze().numpy(), 16000
        return np.asarray(audio, dtype=np.float32).squeeze(), 16000
