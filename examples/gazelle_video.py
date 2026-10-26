"""Run Gaze-LLE on any video: gaze-target features plus an overlay video.

Pipeline: detect faces, track them, expand the face boxes to head boxes, predict
each person's gaze target at ``--fps``, then write

* ``<out-dir>/gaze.st``: safetensors with ``frame_ids``, ``person_ids``,
  ``gaze_points`` (normalized xy), ``inout`` and ``mass_on_others`` columns;
* ``<out-dir>/gaze.xlsx`` and ``gaze.csv``: the same as a table, one row per
  (frame, person), with a per-person ``summary`` sheet in the workbook;
* ``<out-dir>/gaze_overlay.mp4``: the sampled frames with each head box, an arrow
  to the gaze target, and the in/out score.  When the score is below 0.5 the
  person looks out of the scene (often at the camera), so no arrow is drawn.

and logs per-person statistics.

Usage::

    python examples/gazelle_video.py --video my.mp4 --out-dir out --device-id 0

Without ``--video`` it runs on the repo's 3-speaker test clip.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from exordium.video.core.detection import IouTracker
from exordium.video.core.io import Video, save_video
from exordium.video.face.detector.yolo11 import YoloFace11Detector
from exordium.video.face.gaze.base import box_to_pixels, mass_in_box
from exordium.video.face.gaze.gazelle import GazelleWrapper, boxes_from_tracks, gaze_table

logger = logging.getLogger("gazelle_video")

DEFAULT_VIDEO = Path(__file__).parents[1] / "tests" / "fixtures" / "video" / "multispeaker.mp4"
HEAD_SCALE = 1.5
"""Expansion from tight face boxes to the head boxes Gaze-LLE expects."""
IN_FRAME = 0.5
"""In/out score above which the gaze target counts as inside the frame."""
PALETTE = [(60, 220, 60), (255, 140, 30), (40, 140, 255), (220, 60, 220), (240, 220, 40)]
"""BGR colour per person, cycled."""


def draw_overlay(
    frame: np.ndarray, rows: list[tuple[int, torch.Tensor, float]], boxes: dict, label: str
) -> np.ndarray:
    """Draw head boxes, gaze arrows and in/out scores on one BGR frame.

    Args:
        frame: BGR image, modified in place.
        rows: ``(person_id, gaze_point, inout)`` for every person in the frame.
        boxes: ``{person_id: normalized head box}`` of this frame.
        label: Text for the top-left corner.

    Returns:
        The annotated frame.

    """
    height, width = frame.shape[:2]
    for person, point, inout in rows:
        colour = PALETTE[person % len(PALETTE)]
        x1, y1, x2, y2 = (int(v) for v in box_to_pixels(boxes[person], width, height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        if inout > IN_FRAME:
            target = (int(point[0] * width), int(point[1] * height))
            cv2.arrowedLine(frame, ((x1 + x2) // 2, (y1 + y2) // 2), target, colour, 4)
            cv2.circle(frame, target, 10, colour, -1)
        cv2.putText(frame, f"{person}: {inout:.2f}", (x1, y1 - 8), 0, 0.9, colour, 2)
    cv2.putText(frame, label, (12, 34), 0, 1.0, (255, 255, 255), 2)
    return frame


def main() -> None:
    """Run the pipeline, write features and overlay, log statistics."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out-dir", type=Path, default=Path("gazelle_out"))
    parser.add_argument("--device-id", type=int, default=None, help="0 for MPS/CUDA, omit for CPU")
    parser.add_argument("--fps", type=float, default=10.0, help="gaze sampling rate")
    parser.add_argument("--min-track", type=int, default=100, help="min frames for a face track")
    parser.add_argument("--variant", default="vitl14_inout_childplay")
    parser.add_argument("--autocast", action="store_true", help="bf16/fp16; faster on MPS/CUDA")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with Video(args.video) as video:
        width, height, video_fps = video.width, video.height, video.fps
    detections = YoloFace11Detector(device_id=args.device_id).detect_video(args.video)
    tracker = IouTracker(max_lost=30).label(detections, min_score=0.5).merge()
    tracker.select_long_tracks(min_length=args.min_track)
    tracks = tracker.selected_tracks
    if not tracks:
        raise SystemExit(f"No face track of at least {args.min_track} frames in {args.video}")
    head_boxes = boxes_from_tracks(tracks, width, height, scale=HEAD_SCALE)
    face_boxes = boxes_from_tracks(tracks, width, height)
    people = sorted(tracks)
    logger.info(f"tracked people: {people}")

    def mass_on_others(heatmaps, frame_ids, person_ids):
        """Per row, the heatmap share on each person's face (zero on one's own)."""
        out = torch.zeros(len(frame_ids), len(people))
        for row, (frame, person) in enumerate(zip(frame_ids.tolist(), person_ids.tolist())):
            for col, other in enumerate(people):
                if other != person and other in face_boxes.get(frame, {}):
                    out[row, col] = mass_in_box(heatmaps[row], face_boxes[frame][other])
        return out

    model = GazelleWrapper(args.variant, device_id=args.device_id, autocast=args.autocast)
    result = model.video_to_gaze(
        args.video, head_boxes, fps=args.fps, reducers={"mass_on_others": mass_on_others}
    )
    save_file(result, str(args.out_dir / "gaze.st"))
    table = gaze_table(
        result, fps=video_fps, labels={"mass_on_others": [f"person_{p}" for p in people]}
    )
    table.to_csv(args.out_dir / "gaze.csv", index=False)

    frame_ids = sorted(set(result["frame_ids"].tolist()))
    overlay: list[np.ndarray] = []
    with Video(args.video) as video:
        for start in range(0, len(frame_ids), 32):
            chunk = frame_ids[start : start + 32]
            for frame_id, rgb in zip(chunk, video.get_frames_at(chunk)):
                picked = (result["frame_ids"] == frame_id).nonzero().flatten().tolist()
                rows = [
                    (
                        int(result["person_ids"][i]),
                        result["gaze_points"][i],
                        float(result["inout"][i]),
                    )
                    for i in picked
                ]
                bgr = cv2.cvtColor(rgb.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
                label = f"frame {frame_id}  t={frame_id / video_fps:.1f}s"
                overlay.append(draw_overlay(bgr, rows, head_boxes[frame_id], label))
    save_video(
        [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in overlay],
        args.out_dir / "gaze_overlay.mp4",
        fps=args.fps,
    )

    summary = []
    for person in people:
        rows = result["person_ids"] == person
        inside = result["inout"][rows] > IN_FRAME
        on_other = result["mass_on_others"][rows].max(dim=1).values > 0.3
        summary.append(
            {
                "person_id": person,
                "samples": int(rows.sum()),
                "target_in_frame": float(inside.float().mean()),
                "looking_at_another_face": float((inside & on_other).float().mean()),
            }
        )
        logger.info(
            f"person {person}: {summary[-1]['samples']} samples, "
            f"target in frame {summary[-1]['target_in_frame']:.0%}, "
            f"looking at another face {summary[-1]['looking_at_another_face']:.0%}"
        )
    with pd.ExcelWriter(args.out_dir / "gaze.xlsx") as workbook:
        table.to_excel(workbook, sheet_name="gaze", index=False)
        pd.DataFrame(summary).to_excel(workbook, sheet_name="summary", index=False)
    logger.info(f"wrote gaze.st, gaze.xlsx, gaze.csv and gaze_overlay.mp4 to {args.out_dir}")


if __name__ == "__main__":
    main()
