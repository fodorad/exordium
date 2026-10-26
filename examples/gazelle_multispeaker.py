"""Gaze-LLE on the 3-speaker fixture: who looks at whom?

Detects and tracks the faces, predicts gaze targets at 10 fps, then reports per
person how often the target is inside the frame and how much of the heatmap
lands on another person's face.  A low in/out score means the person looks out
of the scene, typically at the camera; the heatmap peak is then not meaningful.

Usage::

    python examples/gazelle_multispeaker.py [--video PATH] [--device-id 0]
"""

import argparse
import logging
from pathlib import Path

import torch

from exordium.video.core.detection import IouTracker
from exordium.video.core.io import Video
from exordium.video.face.detector.yolo11 import YoloFace11Detector
from exordium.video.face.gaze.base import mass_in_box
from exordium.video.face.gaze.gazelle import GazelleWrapper, boxes_from_tracks

logger = logging.getLogger("gazelle_multispeaker")

DEFAULT_VIDEO = Path(__file__).parents[1] / "tests" / "fixtures" / "video" / "multispeaker.mp4"
HEAD_SCALE = 1.5
"""Expansion from tight face boxes to the head boxes Gaze-LLE expects."""


def main() -> None:
    """Run the pipeline and log per-person gaze statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with Video(args.video) as video:
        width, height = video.width, video.height
    detections = YoloFace11Detector(device_id=args.device_id).detect_video(args.video)
    tracker = IouTracker(max_lost=30).label(detections, min_score=0.5).merge()
    tracker.select_long_tracks(min_length=100)
    tracks = tracker.selected_tracks
    head_boxes = boxes_from_tracks(tracks, width, height, scale=HEAD_SCALE)
    face_boxes = boxes_from_tracks(tracks, width, height)
    people = sorted(tracks)

    def mass_on_others(heatmaps, frame_ids, person_ids):
        """Per row, the heatmap share on each person's face (zero on one's own)."""
        out = torch.zeros(len(frame_ids), len(people))
        for row, (frame, person) in enumerate(zip(frame_ids.tolist(), person_ids.tolist())):
            for col, other in enumerate(people):
                if other != person and other in face_boxes.get(frame, {}):
                    out[row, col] = mass_in_box(heatmaps[row], face_boxes[frame][other])
        return out

    model = GazelleWrapper(device_id=args.device_id)
    result = model.video_to_gaze(
        args.video, head_boxes, fps=args.fps, reducers={"mass_on_others": mass_on_others}
    )
    for person in people:
        rows = result["person_ids"] == person
        inside = result["inout"][rows] > 0.5
        on_other = result["mass_on_others"][rows].max(dim=1).values > 0.3
        logger.info(
            f"person {person}: {int(rows.sum())} samples, "
            f"target in frame {inside.float().mean():.0%}, "
            f"looking at another face {(inside & on_other).float().mean():.0%}"
        )


if __name__ == "__main__":
    main()
