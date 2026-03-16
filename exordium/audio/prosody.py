from collections import deque

import numpy as np
import torch


class ProsodyExtractor:
    """Prosody feature extractor with VAD-gated analysis.

    Uses Silero VAD to detect speech regions in each audio chunk,
    then computes pitch, energy, and energy variance only on the voiced
    segments.  A voice ratio (fraction of speech) is derived from the
    VAD timestamps and feeds into the final engagement score.

    Args:
        sr: Sample rate in Hz.  Must be 8000 or 16000 (Silero VAD requirement).
        buffer_size: Number of past chunks to smooth over.

    """

    def __init__(self, sr: int = 16000, buffer_size: int = 50):
        if sr not in (8000, 16000):
            raise ValueError(f"Silero VAD requires sr=8000 or sr=16000, got {sr}")
        self.sr = sr
        self.pitch_buffer: deque[float] = deque(maxlen=buffer_size)
        self.energy_buffer: deque[float] = deque(maxlen=buffer_size)
        self.energy_variance_buffer: deque[float] = deque(maxlen=buffer_size)
        self.voice_ratio_buffer: deque[float] = deque(maxlen=buffer_size)
        self._vad_model, self._get_speech_timestamps = self._load_vad()

    def _load_vad(self) -> tuple:
        """Load Silero VAD model and helper from torch.hub."""
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        get_speech_timestamps = utils[0]
        return model, get_speech_timestamps

    def reset(self) -> None:
        """Clear all smoothing buffers and reset VAD state."""
        self.pitch_buffer.clear()
        self.energy_buffer.clear()
        self.energy_variance_buffer.clear()
        self.voice_ratio_buffer.clear()
        self._vad_model.reset_states()

    def _to_tensor(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input to 1D float tensor."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.copy())
        return audio.float().squeeze()

    def _get_speech_segments(
        self,
        audio: torch.Tensor,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        vad_threshold: float = 0.4,
    ) -> tuple[torch.Tensor | None, float, bool]:
        """Run VAD and return concatenated speech audio, voice ratio, and binary speech flag.

        Args:
            audio: Audio tensor.
            min_speech_duration_ms: Minimum speech segment duration in ms (lower = more sensitive).
            min_silence_duration_ms: Minimum silence duration to separate segments in ms.
            vad_threshold: Minimum voice_ratio (0-1) to classify as "speech active".

        Returns:
            Tuple of (speech_audio, voice_ratio, is_speech_active).
            speech_audio is None when no speech is detected.
            is_speech_active is True if voice_ratio >= vad_threshold.

        """
        timestamps = self._get_speech_timestamps(
            audio,
            self._vad_model,
            sampling_rate=self.sr,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        if not timestamps:
            return None, 0.0, False

        speech_samples = sum(ts["end"] - ts["start"] for ts in timestamps)
        voice_ratio = speech_samples / len(audio)

        parts = [audio[ts["start"] : ts["end"]] for ts in timestamps]
        speech_audio = torch.cat(parts)

        # Binary classification: active speech if voice_ratio exceeds threshold
        is_speech_active = voice_ratio >= vad_threshold

        return speech_audio, float(voice_ratio), is_speech_active

    def _compute_energy(self, audio: torch.Tensor) -> float:
        """RMS energy of the audio chunk."""
        return torch.sqrt(torch.mean(audio**2)).item()

    def _compute_energy_variance(self, audio: torch.Tensor) -> float:
        """Variance of short-term energy within the audio chunk.

        Computes RMS energy in 25ms frames with 10ms hop, then returns
        the variance of those values.  High variance indicates dynamic,
        expressive speech; low variance indicates flat, monotone speech.

        """
        frame_length = int(0.025 * self.sr)  # 25 ms
        hop_length = int(0.010 * self.sr)  # 10 ms

        energies = []
        for start in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[start : start + frame_length]
            rms = torch.sqrt(torch.mean(frame**2)).item()
            energies.append(rms)

        if len(energies) < 2:
            return 0.0
        return float(np.var(energies))

    def _compute_pitch(self, audio: torch.Tensor) -> float:
        """Estimate fundamental frequency via autocorrelation.

        Searches for pitch in the range ~50 Hz to ~500 Hz.
        Returns 0.0 if no clear pitch is detected.

        """
        min_lag = max(1, self.sr // 500)  # ~500 Hz upper bound
        max_lag = min(len(audio) // 2, self.sr // 50)  # ~50 Hz lower bound

        if max_lag <= min_lag:
            return 0.0

        # Autocorrelation via conv1d
        x = audio.unsqueeze(0).unsqueeze(0)
        autocorr = torch.nn.functional.conv1d(x, x, padding=len(audio) - 1).squeeze()

        mid = len(autocorr) // 2
        zero_lag = autocorr[mid].item()

        # No energy → no pitch
        if zero_lag < 1e-10:
            return 0.0

        lags = autocorr[mid + min_lag : mid + max_lag]

        if len(lags) == 0:
            return 0.0

        peak_val = lags.max().item()

        # Peak must be at least 30% of zero-lag energy to count as periodic
        if peak_val < 0.3 * zero_lag:
            return 0.0

        peak_idx = torch.argmax(lags).item() + min_lag
        return self.sr / peak_idx if peak_idx > 0 else 0.0

    def extract(
        self,
        audio_chunk: np.ndarray | torch.Tensor,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        vad_threshold: float = 0.4,
    ) -> dict[str, float | bool]:
        """Extract prosody features from an audio chunk.

        VAD is applied first to isolate speech regions.  Pitch, energy,
        and energy variance are computed only on the voiced segments.
        When no speech is detected, prosody features are 0.0.

        Args:
            audio_chunk: 1D audio signal (mono).
            min_speech_duration_ms: Minimum speech segment duration in ms (lower = more sensitive).
            min_silence_duration_ms: Minimum silence between segments in ms.
            vad_threshold: Minimum voice_ratio (0-1) to classify as "speech active" (default 0.4 = 40%).

        Returns:
            Dict with smoothed pitch, energy, energy_variance, voice_ratio,
            pitch_variance (over buffer), engagement score, and is_speech_active (binary).

        """
        audio = self._to_tensor(audio_chunk)
        speech_audio, voice_ratio, is_speech_active = self._get_speech_segments(
            audio, min_speech_duration_ms, min_silence_duration_ms, vad_threshold
        )

        if speech_audio is not None and len(speech_audio) > 0:
            pitch = self._compute_pitch(speech_audio)
            energy = self._compute_energy(speech_audio)
            energy_var = self._compute_energy_variance(speech_audio)
        else:
            pitch = 0.0
            energy = 0.0
            energy_var = 0.0

        self.pitch_buffer.append(pitch)
        self.energy_buffer.append(energy)
        self.energy_variance_buffer.append(energy_var)
        self.voice_ratio_buffer.append(voice_ratio)

        smooth_energy = float(np.mean(self.energy_buffer))
        smooth_energy_var = float(np.mean(self.energy_variance_buffer))
        smooth_voice_ratio = float(np.mean(self.voice_ratio_buffer))

        # Pitch variance over the buffer window (expressiveness over time)
        voiced_pitches = [p for p in self.pitch_buffer if p > 0]
        pitch_variance = float(np.var(voiced_pitches)) if len(voiced_pitches) >= 2 else 0.0

        engagement = self._compute_engagement(
            pitch_variance, smooth_energy, smooth_energy_var, smooth_voice_ratio
        )

        return {
            "pitch": float(np.mean(self.pitch_buffer)),
            "energy": smooth_energy,
            "energy_variance": smooth_energy_var,
            "voice_ratio": smooth_voice_ratio,
            "engagement": engagement,
        }

    def _compute_engagement(
        self,
        pitch_variance: float,
        energy_mean: float,
        energy_variance: float,
        voice_ratio: float,
    ) -> float:
        """Compute engagement score from prosody features.

        Components:
            - pitch_variance: Monotone (low) vs expressive (high).
            - energy_mean: Quiet (low) vs loud (high).
            - energy_variance: Flat (low) vs dynamic (high).
            - voice_ratio: Silence (low) vs active (high).

        Returns:
            Score between 0.0 (low) and 1.0 (high).

        """
        pitch_var_norm = np.clip(pitch_variance / 2000, 0, 1)  # Hz^2 variance
        energy_norm = np.clip(energy_mean / 0.15, 0, 1)  # RMS 0-0.15
        energy_var_norm = np.clip(energy_variance / 0.005, 0, 1)  # RMS variance

        engagement = (
            0.25 * pitch_var_norm + 0.30 * energy_norm + 0.20 * energy_var_norm + 0.25 * voice_ratio
        )
        return float(np.clip(engagement, 0.0, 1.0))
