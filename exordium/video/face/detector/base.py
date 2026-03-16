"""Base class for face detectors."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tqdm import tqdm

from exordium.utils.decorator import load_or_create
from exordium.video.core.bb import xyxy2xywh
from exordium.video.core.detection import FrameDetections, VideoDetections
from exordium.video.core.io import Video, image_to_np


class FaceDetector(ABC):
    """Face detector abstract class definition.

    notation:
        y is row (height), x is column (width)

    image:
        (0,0)---(0,w)
          |       |
          |       |
        (h,0)---(h,w)

    bounding box:
        (x_min, y_min, x_max, y_max)

    detection format
        (y_min, x_min) -- (y_min, x_max)
              |                 |
        (y_max, x_min) -- (y_max, x_max)
    """

    def __init__(self, batch_size: int = 32, verbose: bool = False):
        self.batch_size = batch_size
        self.verbose = verbose

    @abstractmethod
    def run_detector(
        self, images_rgb: list[np.ndarray]
    ) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
        """Run detector on the images.

        If no face is detected, an empty list will be at that frame index.

        Args:
            images_rgb: List of input images of shape (H, W, 3) in RGB channel order.

        Returns:
            List of detections per image. Each detection is a tuple of
            (bb_xyxy, landmarks, score) where bb_xyxy is [x_min, y_min, x_max, y_max],
            landmarks is (5, 2) with (x, y) coordinates, and score is in [0, 1].

        """

    @load_or_create("fdet")
    def detect_image_path(self, image_path: str | Path, **kwargs) -> FrameDetections:
        """Run detector on a single image given by its path.

        Args:
            image_path: Path to the image file.
            **kwargs: Additional arguments forwarded to detect_image.

        Returns:
            FrameDetections: detections within the image.

        """
        image_rgb = image_to_np(image_path, channel_order="RGB")
        return self.detect_image(image_rgb, image_path=str(image_path), **kwargs)

    @load_or_create("fdet")
    def detect_image(self, image_rgb: np.ndarray, **kwargs) -> FrameDetections:
        """Run detector on a single image.

        Args:
            image_rgb: Input image of shape (H, W, 3) in RGB channel order.
            **kwargs: Additional arguments including image_path for caching.

        Returns:
            FrameDetections containing detected faces within the image.

        """
        frame_dets: list[list[tuple[np.ndarray, np.ndarray, float]]] = self.run_detector(
            [image_rgb]
        )

        frame_detections = FrameDetections()
        for bb_xyxy, landmarks, score in frame_dets[0]:
            frame_detections.add_dict(
                {
                    "frame_id": -1,
                    "source": kwargs.get("image_path", image_rgb),
                    "score": score,
                    "bb_xywh": xyxy2xywh(bb_xyxy),
                    "landmarks": np.rint(np.array(landmarks)).astype(int),
                }
            )

        return frame_detections

    @load_or_create("vdet")
    def detect_frame_dir(self, frame_dir: str | Path, **_kwargs) -> VideoDetections:
        """Run detector on ordered frames of a video in a single folder.

        Args:
            frame_dir (str | Path): path to the frame directory.

        Returns:
            VideoDetections: detections within the images.

        """
        frame_paths = sorted(Path(frame_dir).iterdir())
        video_detections = VideoDetections()

        for batch_ind in tqdm(
            range(0, len(frame_paths), self.batch_size),
            desc="Face detection",
            disable=not self.verbose,
        ):
            batch_frame_paths = frame_paths[batch_ind : batch_ind + self.batch_size]
            images = [image_to_np(frame_path, "RGB") for frame_path in batch_frame_paths]
            frame_dets: list[list[tuple[np.ndarray, np.ndarray, float]]] = self.run_detector(images)

            for frame_det, frame_path in zip(frame_dets, batch_frame_paths):
                frame_detections = FrameDetections()
                for bb_xyxy, landmarks, score in frame_det:
                    frame_detections.add_dict(
                        {
                            "frame_id": int(Path(frame_path).stem),
                            "source": str(frame_path),
                            "score": score,
                            "bb_xywh": xyxy2xywh(bb_xyxy),
                            "landmarks": np.rint(np.array(landmarks)).astype(int),
                        }
                    )

                video_detections.add(frame_detections)

        return video_detections

    @load_or_create("vdet")
    def detect_video(self, video_path: str | Path, **_kwargs) -> VideoDetections:
        """Run detector on a video given by its path.

        Args:
            video_path (str | Path): path to the video file.

        Returns:
            VideoDetections: detections within the images.

        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist at {video_path}")

        video_detections = VideoDetections()
        frame_id = 0

        with Video(video_path) as video:
            for batch in tqdm(
                video.iter_batches(batch_size=self.batch_size),
                total=(video.num_frames + self.batch_size - 1) // self.batch_size,
                desc="Face detection",
                disable=not self.verbose,
            ):
                images = [f.permute(1, 2, 0).cpu().numpy() for f in batch]
                frame_dets = self.run_detector(images)

                for frame_det in frame_dets:
                    frame_detections = FrameDetections()
                    for bb_xyxy, landmarks, score in frame_det:
                        frame_detections.add_dict(
                            {
                                "frame_id": frame_id,
                                "source": str(video_path),
                                "score": score,
                                "bb_xywh": xyxy2xywh(bb_xyxy),
                                "landmarks": np.rint(np.array(landmarks)).astype(int),
                            }
                        )
                    video_detections.add(frame_detections)
                    frame_id += 1

        return video_detections
