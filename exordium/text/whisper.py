"""Whisper speech-to-text model wrapper."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import cast

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, TextIteratorStreamer

from exordium.audio.io import load_audio
from exordium.text.base import (
    WHISPER_WINDOW_SECONDS,
    Segment,
    SpeechToText,
    StreamingMixin,
    to_mono_16k,
)
from exordium.utils.device import get_torch_device

logger = logging.getLogger(__name__)
"""Module-level logger."""

_TIMESTAMP_TOKEN_RE = re.compile(r"<\|[\d.]+\|>")
"""Matches Whisper timestamp tokens (e.g. ``<|12.34|>``) emitted during long-form decoding."""


class WhisperWrapper(StreamingMixin, SpeechToText):
    """Wrapper for HuggingFace Whisper speech-to-text models.

    Uses ``transformers`` for native MPS/CUDA/CPU support with optional
    word-by-word streaming via :class:`~transformers.TextIteratorStreamer`.

    The pipeline:

    * :meth:`preprocess` — converts any audio input to a feature dict ready
      for the model.
    * :meth:`inference` — runs ``model.generate`` and returns the decoded
      string.  For streaming, use :meth:`stream` instead.
    * :meth:`__call__` — chains ``preprocess`` + ``inference``; returns the
      full transcript as a string.
    * :meth:`stream` — chains ``preprocess`` + streaming generation; yields
      text chunks word-by-word as they are decoded.

    Args:
        model_name: HuggingFace model identifier. Recommended:
            ``"distil-whisper/distil-large-v3"`` for quality/speed on GPU,
            ``"openai/whisper-large-v3-turbo"`` for highest quality.
        device_id: Device index. ``-1`` or ``None`` → CPU, ``0+`` → GPU/MPS.
        dtype: Torch dtype for the model weights. Defaults to
            ``torch.float16`` on GPU/MPS, ``torch.float32`` on CPU.

    Example — full transcript::

        wrapper = WhisperWrapper()
        text = wrapper(waveform)
        text = wrapper("audio.wav")

    Example — streaming word-by-word::

        for chunk in wrapper.stream(waveform):
            print(chunk, end="", flush=True)

    """

    def __init__(
        self,
        model_name: str = "distil-whisper/distil-large-v3",
        device_id: int = 0,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.device = get_torch_device(device_id)
        self.model_name = model_name

        if dtype is None:
            dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.dtype = dtype

        logger.info(f"Loading {model_name} on {self.device} ({dtype})...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def preprocess(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
    ) -> dict[str, torch.Tensor]:
        """Convert any audio input to model-ready input features on ``self.device``.

        Args:
            audio: One of:

                * ``str | Path`` — loaded via torchaudio, resampled to 16 kHz.
                * ``np.ndarray`` — float32, 1-D or ``(1, T)``, 16 kHz mono.
                * ``torch.Tensor`` — same shape/dtype as ndarray.

            sample_rate: Sample rate of the input waveform when passing an
                array or tensor directly. Ignored for file paths (detected
                automatically). Defaults to 16000.

        Returns:
            Dict with ``input_features`` on ``self.device``, plus an
            ``attention_mask`` when the audio exceeds Whisper's 30 s window.

        """
        if isinstance(audio, (str, Path)):
            waveform, _ = load_audio(audio, target_sample_rate=16000, mono=True, squeeze=True)
            audio = waveform.numpy()
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().squeeze().numpy()
        else:
            audio = np.asarray(audio).squeeze()

        # Whisper's encoder sees a fixed 30 s window. The feature extractor's
        # defaults (truncation + pad to 30 s) would silently discard everything
        # past 30 s, so longer audio must be kept whole and decoded long-form.
        if audio.shape[-1] > int(WHISPER_WINDOW_SECONDS * sample_rate):
            inputs = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
            )
        else:
            inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")

        # Only the float mel features take the model dtype; the attention mask
        # must stay integral.
        return {
            key: value.to(self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }

    def inference(self, inputs: object, **kwargs: object) -> str:
        """Run greedy/beam-search decoding and return the full transcript.

        Args:
            inputs: Feature dict from :meth:`preprocess` on ``self.device``.
            **kwargs: Optional ``language`` (str) and ``beam_size`` (int).

        Returns:
            Transcribed text as a plain string.

        """
        inputs_dict = cast("dict[str, torch.Tensor]", inputs)
        generate_kwargs = self._generate_kwargs(inputs_dict, **kwargs)

        with torch.inference_mode():
            token_ids = self.model.generate(**inputs_dict, **generate_kwargs)

        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()

    def _generate_kwargs(self, inputs: dict[str, torch.Tensor], **kwargs: object) -> dict:
        """Build ``generate`` kwargs, enabling long-form decoding when needed."""
        language: str | None = cast("str | None", kwargs.get("language"))
        beam_size_obj = kwargs.get("beam_size", 5)
        beam_size: int = beam_size_obj if isinstance(beam_size_obj, int) else 5
        generate_kwargs: dict = {"num_beams": beam_size}
        if language is not None:
            generate_kwargs["language"] = language
        # Long-form (>30 s) inputs carry an attention mask and require timestamps
        # for Whisper's sequential chunk-by-chunk decoding.
        if "attention_mask" in inputs:
            generate_kwargs["return_timestamps"] = True
        return generate_kwargs

    def transcribe_segments(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        language: str | None = None,
        beam_size: int = 5,
        sample_rate: int = 16000,
    ) -> list[Segment]:
        """Transcribe audio into timestamped :class:`Segment` chunks.

        Works for audio of any length: short clips decode in one pass, longer
        recordings use Whisper's sequential long-form decoding. The segments are
        what :meth:`~exordium.text.base.ForcedAligner.align_segments` consumes to
        keep forced alignment bounded on long files.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            language: Language code or ``None`` for auto-detection.
            beam_size: Decoder beam width.
            sample_rate: Sample rate of the input waveform (ignored for paths).

        Returns:
            Segments in chronological order; empty if nothing was transcribed.

        """
        inputs = self.preprocess(audio, sample_rate=sample_rate)
        generate_kwargs = self._generate_kwargs(inputs, language=language, beam_size=beam_size)
        generate_kwargs["return_timestamps"] = True
        generate_kwargs["return_segments"] = True

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        segments: list[Segment] = []
        for raw in outputs["segments"][0]:
            text = self.processor.batch_decode(
                raw["tokens"].unsqueeze(0), skip_special_tokens=True
            )[0].strip()
            if not text:
                continue
            segments.append(Segment(text=text, start=float(raw["start"]), end=float(raw["end"])))
        return segments

    def __call__(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        language: str | None = None,
        beam_size: int = 5,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe audio and return the full text as a plain string.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            language: Language code or ``None`` for auto-detection.
            beam_size: Decoder beam width.
            sample_rate: Sample rate of the input waveform (ignored for paths).

        Returns:
            Transcribed text.

        """
        return self.inference(
            self.preprocess(audio, sample_rate=sample_rate),
            language=language,
            beam_size=beam_size,
        )

    def stream(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        language: str | None = None,
        beam_size: int = 1,
        sample_rate: int = 16000,
    ) -> Iterator[str]:
        """Transcribe audio with word-by-word streaming output.

        Runs ``model.generate`` in a background thread and yields decoded text
        chunks to the caller as they are produced — no need to wait for the
        full transcript.  Beam size defaults to ``1`` (greedy) for lowest
        latency; increase for better quality at the cost of first-token delay.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            language: Language code or ``None`` for auto-detection.
            beam_size: Decoder beam width. Defaults to ``1`` for streaming.
            sample_rate: Sample rate of the input waveform (ignored for paths).

        Audio longer than Whisper's 30 s window is streamed window by window:
        ``TextIteratorStreamer`` only observes the first chunk of Whisper's
        sequential long-form decoding, so streaming the whole file in one
        ``generate`` call would silently stop after 30 s. Each window is decoded
        short-form instead, which keeps the full transcript flowing. A word
        spanning a window boundary may be split.

        Yields:
            Decoded text chunks (words or sub-word tokens) as strings.

        Raises:
            Exception: Whatever ``model.generate`` raised in the worker thread,
                re-raised on the caller's thread once the stream drains.

        Example::

            for chunk in wrapper.stream("audio.wav"):
                print(chunk, end="", flush=True)
            print()  # newline after full transcript

        """
        if isinstance(audio, (str, Path)):
            waveform, rate = to_mono_16k(audio)
        else:
            waveform, _ = to_mono_16k(audio)
            rate = sample_rate

        window = int(WHISPER_WINDOW_SECONDS * rate)
        total = waveform.numel()
        for start in range(0, total, window):
            chunk = waveform[start : start + window]
            if chunk.numel() < rate // 10:  # ignore <100 ms tail
                continue
            yield from self._stream_window(chunk, language, beam_size, rate)

    def _stream_window(
        self,
        waveform: torch.Tensor,
        language: str | None,
        beam_size: int,
        sample_rate: int,
    ) -> Iterator[str]:
        """Stream one ``<=30 s`` window, re-raising worker-thread failures."""
        inputs = self.preprocess(waveform, sample_rate=sample_rate)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )
        generate_kwargs = self._generate_kwargs(inputs, language=language, beam_size=beam_size)

        failure: list[BaseException] = []

        def _generate() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**inputs, streamer=streamer, **generate_kwargs)
            except BaseException as exc:  # noqa: BLE001 - surfaced to the caller below
                failure.append(exc)
                # Without this the consumer would block on the streamer forever.
                streamer.end()

        thread = Thread(target=_generate, daemon=True)
        thread.start()

        for chunk in streamer:
            text = _TIMESTAMP_TOKEN_RE.sub("", chunk)
            if text:
                yield text

        thread.join()
        if failure:
            raise failure[0]
