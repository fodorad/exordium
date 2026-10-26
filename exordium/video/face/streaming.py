"""Live face tracking: frames in, per-frame tracks and a provisional protagonist out."""

import logging
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from exordium.video.core.streaming import StreamingTracker, StreamUpdate
from exordium.video.face.detector.base import FaceDetector

logger = logging.getLogger(__name__)
"""Module-level logger."""


class StreamingFacePipeline:
    """Detect and track faces on frames as they arrive.

    Chains a :class:`~exordium.video.face.detector.base.FaceDetector` with a
    :class:`~exordium.video.core.streaming.StreamingTracker`; nothing is read
    ahead, so it works on a live camera, a call, or any frame iterable.

    Args:
        detector: Face detector run on every frame.
        tracker: Streaming tracker fed with the detections.

    Example::

        pipeline = StreamingFacePipeline(
            YoloFace11Detector(), StreamingTracker(IouTracker())
        )
        for update in pipeline.process(frames):       # frames: (3, H, W) uint8 RGB
            if update.stable:
                print("protagonist:", update.protagonist_id)

    """

    def __init__(self, detector: FaceDetector, tracker: StreamingTracker) -> None:
        self.detector = detector
        self.tracker = tracker
        self._next_frame_id = 0

    def push(self, frame: torch.Tensor | np.ndarray, frame_id: int | None = None) -> StreamUpdate:
        """Detect and track faces in the next frame.

        Args:
            frame: ``(3, H, W)`` tensor or ``(H, W, 3)`` array, uint8 RGB.
            frame_id: Frame index; counts up from 0 when omitted.

        Returns:
            The :class:`~exordium.video.core.streaming.StreamUpdate` for this frame.

        """
        if frame_id is None:
            frame_id = self._next_frame_id
        self._next_frame_id = frame_id + 1
        detections = self.detector.detect_frame(frame, frame_id)
        return self.tracker.push(detections, frame_id=frame_id)

    def process(self, frames: Iterable[torch.Tensor | np.ndarray]) -> Iterator[StreamUpdate]:
        """Lazily yield one update per incoming frame.

        Args:
            frames: Any iterable of frames, such as a camera generator.

        Yields:
            One :class:`~exordium.video.core.streaming.StreamUpdate` per frame.

        """
        for frame in frames:
            yield self.push(frame)

    def reset(self) -> None:
        """Forget all tracks and restart frame numbering for a new session."""
        self.tracker.reset()
        self._next_frame_id = 0
