"""Follow the protagonist of a video as if it were a live stream.

Frames are fed one at a time to a detector + tracker; the script logs when the
provisional protagonist changes and when the choice first becomes stable, then
compares the final choice with the offline whole-clip answer.

Usage::

    python examples/streaming_protagonist.py [--video PATH] [--device-id 0]
"""

import argparse
import logging
from pathlib import Path

from exordium.video.core.detection import FaceClusterTracker, IouTracker
from exordium.video.core.io import Video
from exordium.video.core.streaming import StreamingTracker
from exordium.video.deep.adaface import AdaFaceWrapper
from exordium.video.face.detector.yolo11 import YoloFace11Detector
from exordium.video.face.streaming import StreamingFacePipeline

logger = logging.getLogger("streaming_protagonist")

DEFAULT_VIDEO = Path(__file__).parents[1] / "tests" / "fixtures" / "video" / "fi_protagonist.mp4"


def main() -> None:
    """Stream a video through the pipeline and compare with the offline protagonist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--device-id", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    detector = YoloFace11Detector(device_id=args.device_id)
    pipeline = StreamingFacePipeline(detector, StreamingTracker(IouTracker(max_lost=30)))

    leader, first_stable, last = None, None, None
    with Video(args.video) as video:
        frames = (frame for batch in video.iter_batches(batch_size=16) for frame in batch)
        for update in pipeline.process(frames):
            if update.protagonist_id != leader:
                leader = update.protagonist_id
                logger.info(f"frame {update.frame_id}: protagonist -> track {leader}")
            if update.stable and first_stable is None:
                first_stable = update.frame_id
                logger.info(f"frame {update.frame_id}: choice is stable (track {leader})")
            last = update
    if last is None:
        raise SystemExit(f"No frames in {args.video}")
    logger.info(f"streaming: protagonist track {last.protagonist_id}, margin {last.margin:.2f}")

    stats = pipeline.tracker.stats
    spans = ", ".join(
        f"track {t}: frames {st.first_frame}..{st.last_frame}" for t, st in sorted(stats.items())
    )
    logger.info(f"streaming tracks: {spans}")

    offline = FaceClusterTracker(AdaFaceWrapper(device_id=args.device_id))
    protagonist = offline.run(detector.detect_video(args.video))
    if protagonist is not None:
        logger.info(
            f"offline protagonist: frames "
            f"{protagonist.first_detection().frame_id}..{protagonist.last_detection().frame_id}"
        )
    logger.info(
        "A person split into several online tracks (e.g. overlapping boxes) can change the "
        "streaming protagonist id; the offline path merges them by global face clustering."
    )


if __name__ == "__main__":
    main()
