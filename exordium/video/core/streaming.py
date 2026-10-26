"""Streaming face tracking: one frame at a time, with a provisional protagonist.

Batch tracking (:meth:`~exordium.video.core.detection.Tracker.label`) needs every
frame up front.  :class:`StreamingTracker` wraps an *online* tracker
(:class:`~exordium.video.core.detection.IouTracker` or
:class:`~exordium.video.core.detection.FaceIdTracker`) so a live feed can be
tracked as frames arrive, and keeps a running guess of the protagonist that is
revised as evidence accumulates.

The protagonist is scored like the offline
:meth:`~exordium.video.core.detection.FaceClusterTracker.select_protagonist`:
frames on screen times the largest face size.  Each update also says whether the
current choice looks settled (see :class:`StreamUpdate`), so a consumer knows
when to commit.

Memory stays bounded on endless streams: a track keeps only its most recent
detections, scores are kept as running aggregates, and tracks that stay absent
are forgotten.  The globally clustering
:class:`~exordium.video.core.detection.FaceClusterTracker` is inherently offline
and cannot be wrapped.
"""

import logging
import math
from dataclasses import dataclass, field

from exordium.video.core.detection import Detection, FrameDetections, Tracker

logger = logging.getLogger(__name__)
"""Module-level logger."""


@dataclass
class TrackStats:
    """Running aggregates of one track, independent of the detections kept.

    Attributes:
        track_id: Id of the track in the wrapped tracker.
        first_frame: Frame id where the track started.
        last_frame: Frame id of its latest detection.
        frames: On-screen frames, discounted by ``decay`` when forgetting is on.
        max_size: Largest box side seen, in pixels.

    """

    track_id: int
    first_frame: int
    last_frame: int
    frames: float = 0.0
    max_size: float = 0.0

    @property
    def score(self) -> float:
        """Protagonist score: on-screen frames times largest face size."""
        return self.frames * self.max_size


@dataclass(frozen=True)
class StreamUpdate:
    """Result of pushing one frame into a :class:`StreamingTracker`.

    Attributes:
        frame_id: Id of the frame just processed.
        assignments: ``(track_id, detection)`` for every kept detection.
        protagonist_id: Current protagonist track, or ``None`` before any track.
        scores: Current score of every live track.
        margin: Protagonist score over the runner-up's (``inf`` with one track).
        stable: ``True`` once the choice looks settled: the protagonist has been on
            screen for ``min_track_frames``, has led for ``min_stable_frames`` and
            its margin is at least ``min_margin``.
        stable_for: Frames the current protagonist has led without interruption.
        protagonist_detection: The protagonist's detection in this frame, or
            ``None`` if it is not visible in it.

    """

    frame_id: int
    assignments: list[tuple[int, Detection]]
    protagonist_id: int | None
    scores: dict[int, float]
    margin: float
    stable: bool
    stable_for: int
    protagonist_detection: Detection | None = field(default=None)


class StreamingTracker:
    """Track faces frame by frame and follow the likely protagonist.

    Args:
        tracker: Online tracker doing the frame-to-track assignment; it is driven
            through :meth:`~exordium.video.core.detection.Tracker.update`.
        min_score: Minimum detection confidence to track.
        max_lost: Maximum frame gap to keep a track open for matching.
        iou_threshold: Minimum IoU to continue a track.
        decay: Per-frame factor in ``(0, 1]`` applied to on-screen frames, so old
            evidence fades on long streams.  ``1.0`` keeps everything, which makes
            the final choice equal the offline one.
        switch_ratio: A challenger must out-score the protagonist by this factor
            to replace it, which stops flapping between near equals.  ``1.0``
            always follows the highest score.
        min_track_frames: Frames the protagonist needs on screen to be ``stable``.
        min_stable_frames: Frames the protagonist needs to have led to be ``stable``.
        min_margin: Required protagonist-over-runner-up score ratio for ``stable``.
        window: Recent detections kept per track.  Scores are unaffected.
        forget_after: Drop a track after this many frames without a detection.

    Raises:
        ValueError: If ``decay`` is outside ``(0, 1]``.

    """

    def __init__(
        self,
        tracker: Tracker,
        min_score: float = 0.7,
        max_lost: int = 30,
        iou_threshold: float = 0.2,
        decay: float = 1.0,
        switch_ratio: float = 1.1,
        min_track_frames: int = 15,
        min_stable_frames: int = 30,
        min_margin: float = 1.2,
        window: int = 300,
        forget_after: int = 900,
    ) -> None:
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.tracker = tracker
        self.min_score = min_score
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.decay = decay
        self.switch_ratio = switch_ratio
        self.min_track_frames = min_track_frames
        self.min_stable_frames = min_stable_frames
        self.min_margin = min_margin
        self.window = window
        self.forget_after = forget_after
        self.stats: dict[int, TrackStats] = {}
        """Running aggregates of the live tracks."""
        self._frame_id: int | None = None
        self._protagonist_id: int | None = None
        self._stable_for = 0

    def reset(self) -> None:
        """Forget all tracks and state, so a new sequence can start (ids restart)."""
        self.tracker.reset()
        self.stats.clear()
        self._frame_id = None
        self._protagonist_id = None
        self._stable_for = 0

    def push(self, frame_detections: FrameDetections, frame_id: int | None = None) -> StreamUpdate:
        """Process the next frame.

        Args:
            frame_detections: Detections of this frame (may be empty).
            frame_id: Frame id; taken from the detections when omitted.  Required
                for an empty frame.

        Returns:
            The :class:`StreamUpdate` after this frame.

        Raises:
            ValueError: If the frame id cannot be determined, disagrees with the
                detections, or does not exceed the previous frame's.

        """
        frame_id = self._resolve_frame_id(frame_detections, frame_id)
        assignments = self.tracker.update(
            frame_detections, self.min_score, self.max_lost, self.iou_threshold
        )
        self._record(frame_id, assignments)
        self._forget_absent(frame_id)
        self._frame_id = frame_id

        previous = self._protagonist_id
        self._protagonist_id = self._pick_protagonist()
        self._stable_for = (
            self._stable_for + 1
            if self._protagonist_id is not None and self._protagonist_id == previous
            else 0
        )
        return self._update(frame_id, assignments)

    def _resolve_frame_id(self, frame_detections: FrameDetections, frame_id: int | None) -> int:
        if len(frame_detections) > 0:
            own = frame_detections.frame_id
            if frame_id is not None and frame_id != own:
                raise ValueError(f"frame_id {frame_id} disagrees with detections of frame {own}")
            frame_id = own
        if frame_id is None:
            raise ValueError("frame_id is required for a frame without detections")
        if self._frame_id is not None and frame_id <= self._frame_id:
            raise ValueError(f"frame ids must increase: got {frame_id} after {self._frame_id}")
        return frame_id

    def _record(self, frame_id: int, assignments: list[tuple[int, Detection]]) -> None:
        """Fold this frame into the running stats and trim stored detections."""
        if self.decay < 1.0:
            for stats in self.stats.values():
                stats.frames *= self.decay
        for track_id, detection in assignments:
            stats = self.stats.setdefault(track_id, TrackStats(track_id, frame_id, frame_id))
            stats.frames += 1
            stats.last_frame = frame_id
            stats.max_size = max(
                stats.max_size, float(max(detection.bb_xywh[2], detection.bb_xywh[3]))
            )
            detections = self.tracker.tracks[track_id].detections
            if len(detections) > self.window:
                del detections[: len(detections) - self.window]

    def _forget_absent(self, frame_id: int) -> None:
        gone = [t for t, s in self.stats.items() if frame_id - s.last_frame > self.forget_after]
        for track_id in gone:
            self.tracker.drop_track(track_id)
            del self.stats[track_id]

    def _pick_protagonist(self) -> int | None:
        """Best track, but the current protagonist is only replaced by a clear winner."""
        if not self.stats:
            return None
        best = max(self.stats.values(), key=lambda s: s.score)
        current = self.stats.get(self._protagonist_id) if self._protagonist_id is not None else None
        if current is None or best.score > current.score * self.switch_ratio:
            return best.track_id
        return current.track_id

    def _update(self, frame_id: int, assignments: list[tuple[int, Detection]]) -> StreamUpdate:
        scores = {track_id: stats.score for track_id, stats in self.stats.items()}
        leader = self._protagonist_id
        margin = math.inf
        if leader is not None:
            runner_up = max((s for t, s in scores.items() if t != leader), default=0.0)
            margin = scores[leader] / runner_up if runner_up > 0 else math.inf
        stable = (
            leader is not None
            and self.stats[leader].frames >= self.min_track_frames
            and self._stable_for >= self.min_stable_frames
            and margin >= self.min_margin
        )
        visible = next((d for t, d in assignments if t == leader), None)
        return StreamUpdate(
            frame_id=frame_id,
            assignments=assignments,
            protagonist_id=leader,
            scores=scores,
            margin=margin,
            stable=stable,
            stable_for=self._stable_for,
            protagonist_detection=visible,
        )
