"""MARLIN video-clip feature extractor wrapper using HuggingFace weights.

`MARLIN <https://github.com/ControlNet/MARLIN>`_ (Masked Autoencoder for facial
video Representation LearnINg, CVPR 2023) learns self-supervised representations
from facial video clips.  Unlike the frame-wise models in this package, MARLIN
operates on **16-frame clips** of shape ``(B, 3, 16, 224, 224)`` and produces a
single embedding per clip.

This wrapper loads the encoder architecture from the ``marlin-pytorch`` package
and downloads pre-trained weights from HuggingFace Hub in safetensors format.
MARLIN was trained on **unaligned face crops** (bounding-box crop + resize, no
landmark-based affine alignment), so inputs should follow the same convention.

Three ViT variants are available:

.. list-table::
   :header-rows: 1

   * - Variant
     - HuggingFace repo
     - Embedding dim
   * - ``"small"``
     - ``ControlNet/marlin_vit_small_ytf``
     - 384
   * - ``"base"``
     - ``ControlNet/marlin_vit_base_ytf``
     - 768
   * - ``"large"``
     - ``ControlNet/marlin_vit_large_ytf``
     - 1024

Example::

    from exordium.video.deep.marlin import MarlinWrapper

    model = MarlinWrapper(model_name="base", device_id=0)

    # From a pre-loaded (B, 3, 16, 224, 224) uint8 tensor of face crops:
    features = model(clip_tensor)          # (B, 768)

    # From a face track (easiest — crops are already available):
    result = model.track_to_feature(track, num_frames=300)
    # result["features"]: (W, 768), result["mask"]: (W,)

    # From a directory of pre-cropped face images:
    result = model.dir_to_feature(sorted_paths, face_crops=True)

    # From a video file (runs face detection + tracking internally):
    result = model.video_to_feature("clip.mp4")

"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from exordium.utils.decorator import load_or_create
from exordium.utils.padding import fill_gaps_with_repeat, repeat_pad_time_dim
from exordium.video.core.detection import Detection, Track
from exordium.video.core.io import to_uint8_tensor
from exordium.video.deep.base import VisualModelWrapper

logger = logging.getLogger(__name__)

_HF_REPO_IDS: dict[str, str] = {
    "small": "ControlNet/marlin_vit_small_ytf",
    "base": "ControlNet/marlin_vit_base_ytf",
    "large": "ControlNet/marlin_vit_large_ytf",
}
"""Mapping of MARLIN variant names to HuggingFace repo IDs."""

_FEATURE_DIMS: dict[str, int] = {
    "small": 384,
    "base": 768,
    "large": 1024,
}
"""Output embedding dimension per MARLIN variant."""

_DEFAULT_MARLIN_MODEL = "base"
"""Default MARLIN variant (ViT-B, 768-d)."""

_TIME_DIM: int = 16
"""Number of frames per MARLIN input clip (fixed by the architecture)."""


def _frames_to_clips(
    frames: torch.Tensor,
    stride: int = _TIME_DIM,
) -> tuple[torch.Tensor, list[int]]:
    """Split a sequence of frames into 16-frame clips with repeat-padding.

    Args:
        frames: ``(T, 3, H, W)`` uint8 tensor of face crops.
        stride: Step size between consecutive clip windows.

    Returns:
        Tuple of ``(clips, start_ids)`` where clips has shape
        ``(N, 3, 16, H, W)`` and start_ids is a list of starting frame
        indices.

    """
    frames = repeat_pad_time_dim(frames, _TIME_DIM)
    total = frames.shape[0]

    starts = list(range(0, total - _TIME_DIM + 1, stride))
    if not starts:
        starts = [0]

    clips = torch.stack(
        [frames[s : s + _TIME_DIM].permute(1, 0, 2, 3) for s in starts]
    )  # (N, 3, 16, H, W)
    return clips, starts


class MarlinWrapper(VisualModelWrapper):
    """Wrapper for MARLIN clip-wise video feature extraction from face crops.

    MARLIN processes **16-frame video clips** of face crops (not full frames).
    The model was trained on unaligned bounding-box crops resized to 224x224,
    so inputs should follow the same convention — no landmark-based affine
    alignment.

    Inherits from :class:`~exordium.video.deep.base.VisualModelWrapper` and
    overrides the batch extraction methods (``dir_to_feature``,
    ``track_to_feature``, ``video_to_feature``) with clip-aware versions that
    split frame sequences into 16-frame windows with repeat-padding and
    gap-filling.

    * :meth:`preprocess` — convert face-crop clips to model-ready tensors
    * :meth:`inference` — pure forward pass on prepared ``(B, 3, 16, 224, 224)``
    * :meth:`__call__` — ``preprocess`` → ``inference`` (inherited)
    * :meth:`track_to_feature` — extract from a face :class:`Track` (easiest)
    * :meth:`dir_to_feature` — extract from a directory of face crops or full frames
    * :meth:`video_to_feature` — extract from a video (runs face detection internally)

    Args:
        model_name: MARLIN variant — ``"small"`` (384-d), ``"base"``
            (768-d), or ``"large"`` (1024-d).  Defaults to ``"base"``.
        device_id: GPU device index.  ``None`` or negative uses CPU.

    Raises:
        ValueError: If ``model_name`` is not one of the supported variants.

    Example::

        model = MarlinWrapper(model_name="base", device_id=0)
        features = model(clip_tensor)  # (B, 768)

    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MARLIN_MODEL,
        device_id: int | None = None,
    ) -> None:
        if model_name not in _HF_REPO_IDS:
            raise ValueError(
                f"Invalid model_name: {model_name!r}. Choose from {list(_HF_REPO_IDS)}."
            )
        super().__init__(device_id)
        self.feature_dim = _FEATURE_DIMS[model_name]
        self.model = self._load_model(model_name)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _load_model(model_name: str) -> torch.nn.Module:
        """Download weights from HuggingFace Hub and build the MARLIN encoder.

        Args:
            model_name: One of ``"small"``, ``"base"``, or ``"large"``.

        Returns:
            Loaded ``Marlin`` model in feature-extractor mode.

        """
        from huggingface_hub import hf_hub_download
        from marlin_pytorch import Marlin
        from marlin_pytorch.config import resolve_config
        from safetensors.torch import load_file

        repo_id = _HF_REPO_IDS[model_name]
        config_name = f"marlin_vit_{model_name}_ytf"
        config = resolve_config(config_name)

        weights_path = hf_hub_download(repo_id, "model.safetensors")
        state_dict = load_file(weights_path)
        # HuggingFace weights use a "marlin." prefix that the marlin_pytorch
        # module does not expect — strip it.
        state_dict = {k.removeprefix("marlin."): v for k, v in state_dict.items()}

        model = Marlin(
            img_size=config.img_size,
            patch_size=config.patch_size,
            n_frames=config.n_frames,
            encoder_embed_dim=config.encoder_embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_num_heads=config.encoder_num_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=config.norm_layer,
            init_values=config.init_values,
            tubelet_size=config.tubelet_size,
            as_feature_extractor=True,
        )
        model.load_state_dict(state_dict)
        logger.info("MARLIN (%s) loaded — feature_dim=%d", model_name, config.encoder_embed_dim)
        return model

    # ── preprocess / inference / __call__ ──────────────────────────────────

    def preprocess(self, frames: torch.Tensor | Sequence) -> torch.Tensor:
        """Resize and normalise a batch of 16-frame face-crop clips to ``[0, 1]``.

        Args:
            frames: Face-crop clip(s) in one of these formats:

                * ``torch.Tensor (B, 3, 16, H, W)`` — uint8 RGB clip batch
                * ``torch.Tensor (3, 16, H, W)`` — single clip (auto-unsqueezed)
                * ``np.ndarray (B, 16, H, W, 3)`` — uint8 HWC clip batch
                * ``np.ndarray (16, H, W, 3)`` — single HWC clip (auto-unsqueezed)

        Returns:
            Float tensor ``(B, 3, 16, 224, 224)`` on ``self.device``,
            values in ``[0, 1]``.

        Raises:
            TypeError: If input is not a tensor or ndarray.
            ValueError: If shape cannot be interpreted as ``(B, 3, T, H, W)``.

        """
        clips: torch.Tensor
        if isinstance(frames, np.ndarray):
            arr = np.expand_dims(frames, axis=0) if frames.ndim == 4 else frames
            clips = torch.from_numpy(arr).permute(0, 4, 1, 2, 3)
        elif isinstance(frames, torch.Tensor):
            clips = frames
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(frames).__name__}")

        if clips.ndim == 4:
            clips = clips.unsqueeze(0)

        if clips.ndim != 5 or clips.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, T, H, W) tensor, got shape {tuple(clips.shape)}")

        clips = clips.to(self.device)
        b, c, t, h, w = clips.shape
        if h != 224 or w != 224:
            clips = clips.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            clips = TF.resize(
                clips, [224, 224], interpolation=TF.InterpolationMode.BICUBIC, antialias=True
            )
            clips = clips.reshape(b, t, c, 224, 224).permute(0, 2, 1, 3, 4)

        return clips.float().div(255.0)

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """MARLIN encoder forward pass with mean-pooling over sequence tokens.

        The input must already be a prepared float tensor of face crops on
        ``self.device``.

        Args:
            tensor: Preprocessed float tensor ``(B, 3, 16, 224, 224)``
                on ``self.device``, values in ``[0, 1]``.

        Returns:
            Feature tensor ``(B, D)``.

        """
        return self.model.extract_features(tensor, keep_seq=False)  # ty: ignore[call-non-callable]

    # ── batch feature extraction helpers ──────────────────────────────────

    @load_or_create("st")
    def track_to_feature(
        self,
        track: Track,
        num_frames: int | None = None,
        stride: int = _TIME_DIM,
        verbose: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract MARLIN features from a face track over the full video span.

        The entire video timeline (``num_frames`` frames starting at 0) is
        divided into non-overlapping windows of ``stride`` frames.  For each
        window:

        * If the track has **at least one** detection in that range, the
          crops are gathered, gaps are filled by repeating the nearest valid
          crop, and MARLIN inference produces a real feature vector
          (``mask=True``).
        * If the track has **no** detections in that range, the feature is a
          zero vector (``mask=False``).

        This guarantees a fixed temporal grid: output timestep *i* always
        corresponds to source frames ``[i * stride, i * stride + 16)``, so
        downstream models can use the mask to skip empty windows while
        preserving temporal alignment with other modalities.

        .. note::

           If ``num_frames`` is not provided, the span defaults to
           ``max(detection.frame_id) + 1``, which may undercount if the
           video continues after the last detection.  **Always pass
           num_frames explicitly when the video length is known** to get
           the expected output shape.

        Args:
            track: Face track with per-frame detections.
            num_frames: Total number of frames in the source video.  When
                ``None``, defaults to the last detection's frame ID + 1.
            stride: Step between clip windows.  Defaults to 16
                (non-overlapping).
            verbose: Show progress bar.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Dict with:

            * ``"frame_ids"`` — ``(W,)`` long tensor of window start indices.
            * ``"features"`` — ``(W, D)`` float tensor of MARLIN embeddings.
            * ``"mask"`` — ``(W,)`` bool tensor.  ``True`` where at least one
              face detection existed in the window, ``False`` for zero-filled
              windows.

        """
        return self._extract_from_track(
            track,
            num_frames=num_frames,
            stride=stride,
            verbose=verbose,
        )

    @load_or_create("st")
    def dir_to_feature(
        self,
        img_paths: list[str | Path],
        stride: int = _TIME_DIM,
        face_crops: bool = False,
        verbose: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract MARLIN features from a directory of ordered images.

        Args:
            img_paths: Ordered image file paths.  Each file stem must be
                parseable as an integer frame ID.
            stride: Step between clip windows.
            face_crops: If ``True``, images are already face crops (e.g.
                from Aff-Wild2 cropped_aligned directories) and are used
                directly.  If ``False``, face detection is run on each
                image to extract face crops.
            verbose: Show progress bar.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Dict with ``"frame_ids"`` ``(N,)`` and ``"features"`` ``(N, D)``.

        """
        if not img_paths:
            return {
                "frame_ids": torch.tensor([], dtype=torch.long),
                "features": torch.empty(0, self.feature_dim),
            }

        frame_ids_list = [int(Path(p).stem) for p in img_paths]

        if face_crops:
            # Images are already face crops — load directly
            frames = to_uint8_tensor(img_paths)  # (N, 3, H, W)
        else:
            # Run face detection on each image
            frames = self._detect_crops_from_images(img_paths, verbose=verbose)

        if frames.shape[0] == 0:
            return {
                "frame_ids": torch.tensor([], dtype=torch.long),
                "features": torch.empty(0, self.feature_dim),
            }

        # Build dense tensor with gap filling
        first_id = min(frame_ids_list)
        last_id = max(frame_ids_list)
        span = last_id - first_id + 1
        _, _, fh, fw = frames.shape
        all_frames = torch.zeros(span, 3, fh, fw, dtype=torch.uint8)
        valid_mask = torch.zeros(span, dtype=torch.bool)

        for fid, frame in zip(frame_ids_list, frames):
            idx = fid - first_id
            all_frames[idx] = frame
            valid_mask[idx] = True

        all_frames = fill_gaps_with_repeat(all_frames, valid_mask)

        clips, starts = _frames_to_clips(all_frames, stride=stride)
        ids: list[int] = []
        features: list[torch.Tensor] = []

        for i in tqdm(range(clips.shape[0]), desc="MarlinWrapper dir", disable=not verbose):
            features.append(self(clips[i : i + 1]).cpu())
            ids.append(first_id + starts[i])

        return {
            "frame_ids": torch.tensor(ids, dtype=torch.long),
            "features": torch.cat(features, dim=0),
        }

    @load_or_create("st")
    def video_to_feature(
        self,
        video_path: str | Path,
        stride: int = _TIME_DIM,
        conf: float = 0.7,
        min_score: float = 0.7,
        verbose: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract MARLIN features from a video file.

        Runs YOLO11 face detection and IoU tracking on the video, selects
        the longest track, and extracts clip-wise features over the full
        video timeline.  The total frame count is obtained from video
        metadata so the output grid covers the entire video.

        Args:
            video_path: Path to the input video file.
            stride: Step between clip windows.
            conf: YOLO11 detection confidence threshold.
            min_score: Minimum detection score for tracking.
            verbose: Show progress bars.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Dict with ``"frame_ids"`` ``(W,)``, ``"features"`` ``(W, D)``,
            and ``"mask"`` ``(W,)`` — see :meth:`track_to_feature`.

        """
        video_path = Path(video_path)

        from exordium.video.core.io import get_video_metadata

        num_frames = get_video_metadata(video_path)["num_frames"]

        # Detect faces
        from exordium.video.face.detector.yolo11 import YoloFace11Detector

        detector = YoloFace11Detector(
            device_id=self.device.index if self.device.type != "cpu" else -1,
            conf=conf,
            verbose=verbose,
        )
        vdets = detector.detect_video(video_path)

        # Track faces
        from exordium.video.core.detection import IouTracker

        tracker = IouTracker(verbose=verbose)
        tracker.label(vdets, min_score=min_score)

        if not tracker.tracks:
            logger.warning("No face tracks found in %s", video_path)
            n_windows = len(range(0, num_frames, stride))
            return {
                "frame_ids": torch.arange(0, num_frames, stride, dtype=torch.long),
                "features": torch.zeros(n_windows, self.feature_dim),
                "mask": torch.zeros(n_windows, dtype=torch.bool),
            }

        # Select the longest track
        longest_track = max(tracker.tracks.values(), key=lambda t: len(t.detections))
        logger.info(
            "Selected track %d with %d detections",
            longest_track.track_id,
            len(longest_track.detections),
        )

        return self._extract_from_track(
            longest_track,
            num_frames=num_frames,
            stride=stride,
            verbose=verbose,
        )

    # ── private helpers ───────────────────────────────────────────────────

    def _extract_from_track(
        self,
        track: Track,
        num_frames: int | None = None,
        stride: int = _TIME_DIM,
        verbose: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Window-based extraction over the full video timeline.

        Divides ``[0, num_frames)`` into windows of size ``stride`` and runs
        MARLIN on windows that contain at least one detection.  Windows
        without any detection are zero-filled with ``mask=False``.

        Args:
            track: Face track with per-frame detections.
            num_frames: Total frames in the source video.  Falls back to
                ``max(frame_id) + 1`` when ``None``.
            stride: Window step size.
            verbose: Show progress bar.

        Returns:
            Dict with ``"frame_ids"``, ``"features"``, and ``"mask"``.

        """
        detections = track.detections

        # Determine total timeline length
        if num_frames is None and detections:
            num_frames = max(d.frame_id for d in detections) + 1
        elif num_frames is None:
            num_frames = 0

        # Build window grid over the full video
        window_starts = list(range(0, num_frames, stride))
        if not window_starts and num_frames > 0:
            window_starts = [0]
        n_windows = len(window_starts)

        if n_windows == 0 or not detections:
            return {
                "frame_ids": torch.tensor(window_starts, dtype=torch.long),
                "features": torch.zeros(n_windows, self.feature_dim),
                "mask": torch.zeros(n_windows, dtype=torch.bool),
            }

        # Index detections by frame_id for fast lookup (one detection per frame in a track)
        det_by_frame: dict[int, Detection] = {d.frame_id: d for d in detections}

        features = torch.zeros(n_windows, self.feature_dim)
        mask = torch.zeros(n_windows, dtype=torch.bool)

        for wi in tqdm(range(n_windows), desc="MarlinWrapper track", disable=not verbose):
            win_start = window_starts[wi]
            win_end = min(win_start + _TIME_DIM, num_frames)

            # Collect detections that fall within this window
            window_dets: list[tuple[int, torch.Tensor]] = []
            for fid in range(win_start, win_end):
                if fid in det_by_frame:
                    window_dets.append(
                        (fid - win_start, det_by_frame[fid].crop(square=True, extra_space=1.5))
                    )

            if not window_dets:
                # No faces in this window → zero feature, mask=False
                continue

            mask[wi] = True

            # Build a dense (_TIME_DIM, 3, H, W) tensor for this window
            _, ch, cw = window_dets[0][1].shape
            crops = torch.zeros(_TIME_DIM, 3, ch, cw, dtype=torch.uint8)
            valid = torch.zeros(_TIME_DIM, dtype=torch.bool)

            for local_idx, crop in window_dets:
                crops[local_idx] = crop
                valid[local_idx] = True

            # Fill gaps within the window by repeating nearest valid crop
            crops = fill_gaps_with_repeat(crops, valid)

            # Reshape to (1, 3, 16, H, W) for MARLIN
            clip = crops.permute(1, 0, 2, 3).unsqueeze(0)
            features[wi] = self(clip).cpu().squeeze(0)

        return {
            "frame_ids": torch.tensor(window_starts, dtype=torch.long),
            "features": features,
            "mask": mask,
        }

    @staticmethod
    def _detect_crops_from_images(
        img_paths: list[str | Path],
        verbose: bool = False,
    ) -> torch.Tensor:
        """Run face detection on a list of images and return crops.

        For each image, the highest-confidence detection is used.  Images
        with no detection produce a zero tensor to be gap-filled later.

        Args:
            img_paths: Image file paths.
            verbose: Show progress bar.

        Returns:
            ``(N, 3, H, W)`` uint8 tensor.  ``H`` and ``W`` are determined
            by the first successful detection's square crop.

        """
        from exordium.video.face.detector.yolo11 import YoloFace11Detector

        detector = YoloFace11Detector(conf=0.25, verbose=False)

        crops: list[torch.Tensor] = []
        crop_size: tuple[int, int] | None = None

        for p in tqdm(img_paths, desc="Detecting faces", disable=not verbose):
            fdet = detector.detect_image_path(p)
            if len(fdet) > 0:
                # Pick highest-confidence detection
                best = max(fdet.detections, key=lambda d: d.score)
                crop = best.crop(square=True, extra_space=1.5)
                if crop_size is None:
                    crop_size = (crop.shape[1], crop.shape[2])
                # Resize to consistent size if needed
                if crop.shape[1] != crop_size[0] or crop.shape[2] != crop_size[1]:
                    crop = TF.resize(crop, list(crop_size))
                crops.append(crop)
            else:
                if crop_size is None:
                    # Haven't seen a face yet — use 224x224 placeholder
                    crops.append(torch.zeros(3, 224, 224, dtype=torch.uint8))
                else:
                    crops.append(torch.zeros(3, crop_size[0], crop_size[1], dtype=torch.uint8))

        return torch.stack(crops)
