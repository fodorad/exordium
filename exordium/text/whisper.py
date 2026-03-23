"""Whisper speech-to-text model wrapper."""

import logging
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import cast

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, TextIteratorStreamer

from exordium.audio.io import load_audio
from exordium.text.base import SpeechToText, StreamingMixin
from exordium.utils.device import get_torch_device

logger = logging.getLogger(__name__)
"""Module-level logger."""


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
            Dict with ``input_features`` on ``self.device``.

        """
        if isinstance(audio, (str, Path)):
            waveform, _ = load_audio(audio, target_sample_rate=16000, mono=True, squeeze=True)
            audio = waveform.numpy()
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().squeeze().numpy()
        else:
            audio = np.asarray(audio).squeeze()

        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        return {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

    def inference(self, inputs: object, **kwargs: object) -> str:
        """Run greedy/beam-search decoding and return the full transcript.

        Args:
            inputs: Feature dict from :meth:`preprocess` on ``self.device``.
            **kwargs: Optional ``language`` (str) and ``beam_size`` (int).

        Returns:
            Transcribed text as a plain string.

        """
        inputs_dict = cast("dict[str, torch.Tensor]", inputs)
        language: str | None = cast("str | None", kwargs.get("language"))
        beam_size_obj = kwargs.get("beam_size", 5)
        beam_size: int = beam_size_obj if isinstance(beam_size_obj, int) else 5
        generate_kwargs: dict = {"num_beams": beam_size}
        if language is not None:
            generate_kwargs["language"] = language

        with torch.inference_mode():
            token_ids = self.model.generate(**inputs_dict, **generate_kwargs)

        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()

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

        Yields:
            Decoded text chunks (words or sub-word tokens) as strings.

        Example::

            for chunk in wrapper.stream("audio.wav"):
                print(chunk, end="", flush=True)
            print()  # newline after full transcript

        """
        inputs = self.preprocess(audio, sample_rate=sample_rate)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        generate_kwargs: dict = {
            **inputs,
            "streamer": streamer,
            "num_beams": beam_size,
        }
        if language is not None:
            generate_kwargs["language"] = language

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs, daemon=True)
        thread.start()

        yield from streamer

        thread.join()
