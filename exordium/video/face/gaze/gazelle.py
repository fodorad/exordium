"""Gaze-LLE gaze-target estimation wrapper.

Gaze-LLE predicts *where* a person looks in the image: for every head box it
returns a 64x64 heatmap over the scene and, for the ``inout`` variants, the
probability that the target lies inside the frame.  A frozen DINOv2 encoder
embeds the scene once per frame; a light transformer decoder is then run per
person, conditioned on a learned token added at the head-box location.

The decoder below is a re-implementation of the upstream one so the released
checkpoints load unchanged.  Upstream: https://github.com/fkryan/gazelle
(MIT License, Copyright (c) 2024 Fiona Ryan).  Paper: *Gaze-LLE: Gaze Target
Estimation via Large-Scale Learned Encoders*, Ryan et al., CVPR 2025.

The checkpoints hold decoder weights only; the DINOv2 backbone comes from
Hugging Face (``facebook/dinov2-{base,large}``).  Decoder checkpoints are
mirrored on ``fodorad/exordium-weights`` and downloaded on first use.
"""

import hashlib
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from safetensors.torch import load_file, save_file
from timm.models.vision_transformer import Block
from transformers import Dinov2Model

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import build_hf_model, download_weight
from exordium.utils.decorator import load_or_create
from exordium.utils.device import get_torch_device
from exordium.video.core.detection import Track
from exordium.video.core.io import Video, batch_iterator, to_uint8_tensor
from exordium.video.deep.base import _IMAGENET_MEAN, _IMAGENET_STD
from exordium.video.face.gaze.base import (
    NormalizedBox,
    box_to_normalized,
    expand_box,
    heatmap_to_point,
    mass_in_box,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""

INPUT_SIZE = 448
"""Side of the square image fed to the backbone, in pixels."""

PATCH_SIZE = 14
"""DINOv2 patch size; ``INPUT_SIZE // PATCH_SIZE`` tokens per side."""

FEATURE_GRID = INPUT_SIZE // PATCH_SIZE
"""Side of the backbone token grid (32)."""

HEATMAP_SIZE = FEATURE_GRID * 2
"""Side of the predicted heatmap (64)."""

_UPSTREAM_RELEASE = "https://github.com/fkryan/gazelle/releases/download/v1.0.0"
"""Upstream release hosting the decoder checkpoints; fallback for the mirror."""

_DECODER_DIM = 256
"""Channel width of the decoder."""

_DECODER_LAYERS = 3
"""Number of transformer blocks in the decoder."""


@dataclass(frozen=True)
class GazelleVariant:
    """Static description of a released Gaze-LLE checkpoint.

    Attributes:
        backbone: DINOv2 size understood by :class:`transformers.Dinov2Model`
            (``"base"`` or ``"large"``).
        inout: Whether the checkpoint predicts an in/out-of-frame score.
        weight_file: Path of the decoder checkpoint in the weights mirror.

    """

    backbone: str
    inout: bool
    weight_file: str


def _variant(backbone: str, size: str, suffix: str = "") -> GazelleVariant:
    name = f"gazelle_dinov2_{size}{suffix}.pt"
    return GazelleVariant(backbone, suffix.startswith("_inout"), f"gazelle/{name}")


GAZELLE_VARIANTS: dict[str, GazelleVariant] = {
    "vitb14": _variant("base", "vitb14"),
    "vitl14": _variant("large", "vitl14"),
    "vitb14_inout": _variant("base", "vitb14", "_inout"),
    "vitl14_inout": _variant("large", "vitl14", "_inout"),
    "vitb14_inout_childplay": _variant("base", "vitb14", "_inout_childplay"),
    "vitl14_inout_childplay": _variant("large", "vitl14", "_inout_childplay"),
}
"""Released checkpoints by variant name (``childplay``: GazeFollow then ChildPlay)."""

DEFAULT_VARIANT = "vitl14_inout_childplay"
"""Default variant: ViT-L fine-tuned on ChildPlay, with in/out prediction."""

_BACKBONE_IDS = {"base": "facebook/dinov2-base", "large": "facebook/dinov2-large"}
"""Hugging Face ids of the DINOv2 backbones."""


@dataclass
class GazelleOutput:
    """Gaze-target predictions for a batch of frames.

    People are padded to the largest count in the batch; ``valid`` marks the
    real ones.  Padded entries are zero.

    Attributes:
        heatmap: ``(B, P, 64, 64)`` gaze-target heatmaps in ``[0, 1]``.
        valid: ``(B, P)`` bool mask of real (non-padded) people.
        inout: ``(B, P)`` probability that the target is inside the frame,
            or ``None`` for variants without an in/out head.

    """

    heatmap: torch.Tensor
    valid: torch.Tensor
    inout: torch.Tensor | None = None

    def gaze_point(self, mode: str = "argmax") -> torch.Tensor:
        """Reduce each heatmap to a normalized ``(x, y)`` gaze point.

        Args:
            mode: ``"argmax"`` or ``"soft"``; see
                :func:`~exordium.video.face.gaze.base.heatmap_to_point`.

        Returns:
            Tensor of shape ``(B, P, 2)``.

        """
        return heatmap_to_point(self.heatmap, mode)

    def mass_in_box(self, box: NormalizedBox) -> torch.Tensor:
        """Share of each heatmap that falls inside a normalized box.

        Args:
            box: Box ``(x1, y1, x2, y2)`` in ``[0, 1]`` image coordinates.

        Returns:
            Tensor of shape ``(B, P)``.

        """
        return mass_in_box(self.heatmap, box)


PersonBoxes = Mapping[int, NormalizedBox]
"""Head boxes of one frame, keyed by person id."""

BoxSource = Mapping[int, PersonBoxes] | Callable[[int], PersonBoxes]
"""Head boxes per frame id: a mapping, or a callable evaluated lazily per frame."""

Reducer = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""Maps ``(heatmaps (N, 64, 64), frame_ids (N,), person_ids (N,))`` to per-row values."""


def boxes_from_tracks(
    tracks: Mapping[int, Track], width: int, height: int, scale: float = 1.0
) -> dict[int, dict[int, NormalizedBox]]:
    """Turn face tracks into per-frame head boxes for :meth:`GazelleWrapper.video_to_gaze`.

    Args:
        tracks: Tracks keyed by track id; the id becomes the person id.
        width: Frame width in pixels.
        height: Frame height in pixels.
        scale: Box expansion factor.  Face boxes are tighter than the head boxes
            Gaze-LLE was trained on, so values around ``1.3``-``1.5`` help.

    Returns:
        ``{frame_id: {person_id: normalized xyxy box}}``.

    """
    boxes: dict[int, dict[int, NormalizedBox]] = {}
    for track_id, track in tracks.items():
        for detection in track.detections:
            x1, y1, x2, y2 = (float(v) for v in detection.bb_xyxy)
            box = expand_box(box_to_normalized((x1, y1, x2, y2), width, height), scale)
            boxes.setdefault(detection.frame_id, {})[track_id] = box
    return boxes


def _frame_boxes(boxes: BoxSource, frame_id: int) -> PersonBoxes:
    """Head boxes of one frame from either kind of :data:`BoxSource`."""
    return boxes.get(frame_id, {}) if isinstance(boxes, Mapping) else boxes(frame_id)


def _positional_encoding_2d(d_model: int, height: int, width: int) -> torch.Tensor:
    """Fixed 2D sine/cosine positional encoding of shape ``(d_model, H, W)``."""
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {d_model}")
    half = d_model // 2
    div_term = torch.exp(torch.arange(0.0, half, 2) * -(math.log(10000.0) / half))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe = torch.zeros(d_model, height, width)
    pe[0:half:2] = torch.sin(pos_w * div_term).T.unsqueeze(1).expand(-1, height, -1)
    pe[1:half:2] = torch.cos(pos_w * div_term).T.unsqueeze(1).expand(-1, height, -1)
    pe[half::2] = torch.sin(pos_h * div_term).T.unsqueeze(2).expand(-1, -1, width)
    pe[half + 1 :: 2] = torch.cos(pos_h * div_term).T.unsqueeze(2).expand(-1, -1, width)
    return pe


class GazelleDecoder(nn.Module):
    """Gaze-LLE decoder: head-conditioned transformer over scene tokens.

    Parameter names match the upstream checkpoints (minus the backbone).

    Args:
        backbone_dim: Channel width of the backbone features.
        inout: Add the in/out-of-frame token and head.

    """

    def __init__(self, backbone_dim: int, inout: bool) -> None:
        super().__init__()
        self.inout = inout
        self.linear = nn.Conv2d(backbone_dim, _DECODER_DIM, 1)
        self.head_token = nn.Embedding(1, _DECODER_DIM)
        self.register_buffer(
            "pos_embed", _positional_encoding_2d(_DECODER_DIM, FEATURE_GRID, FEATURE_GRID)
        )
        if inout:
            self.inout_token = nn.Embedding(1, _DECODER_DIM)
        self.transformer = nn.Sequential(
            *[
                Block(dim=_DECODER_DIM, num_heads=8, mlp_ratio=4, drop_path=0.1)
                for _ in range(_DECODER_LAYERS)
            ]
        )
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(_DECODER_DIM, _DECODER_DIM, kernel_size=2, stride=2),
            nn.Conv2d(_DECODER_DIM, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        if inout:
            self.inout_head = nn.Sequential(
                nn.Linear(_DECODER_DIM, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

    def forward(
        self, features: torch.Tensor, head_maps: torch.Tensor, scene_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode one gaze target per person.

        Args:
            features: Scene features ``(B, C, 32, 32)``, one row per frame.
            head_maps: Binary head masks ``(N, 32, 32)``, one row per person.
            scene_index: ``(N,)`` index into ``features`` for each person.

        Returns:
            ``(heatmap, inout)``: ``(N, 64, 64)`` and ``(N,)`` (``None`` without
            an in/out head).

        """
        x = self.linear(features) + self.pos_embed
        x = x[scene_index] + head_maps.unsqueeze(1) * self.head_token.weight.view(1, -1, 1, 1)
        x = x.flatten(2).transpose(1, 2)  # (N, 1024, D)
        if self.inout:
            token = self.inout_token.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([token, x], dim=1)
        x = self.transformer(x)
        inout = None
        if self.inout:
            inout = self.inout_head(x[:, 0]).squeeze(-1)
            x = x[:, 1:]
        x = x.transpose(1, 2).reshape(x.shape[0], _DECODER_DIM, FEATURE_GRID, FEATURE_GRID)
        return self.heatmap_head(x).squeeze(1), inout


def head_maps_from_boxes(
    boxes: Sequence[NormalizedBox | None], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Rasterise normalized head boxes onto the backbone token grid.

    Args:
        boxes: One normalized ``xyxy`` box per person; ``None`` gives an empty mask.
        device: Device of the returned tensor.

    Returns:
        Float tensor ``(N, 32, 32)`` with ones inside each box.

    """
    grid = torch.arange(FEATURE_GRID, device=device)
    if not boxes:
        return torch.zeros(0, FEATURE_GRID, FEATURE_GRID, device=device)
    values = torch.tensor(
        [(0.0, 0.0, 0.0, 0.0) if b is None else b for b in boxes], device=device
    ).clamp(0, 1)
    cells = torch.round(values * FEATURE_GRID)  # round half to even, like upstream
    in_x = (grid >= cells[:, 0:1]) & (grid < cells[:, 2:3])  # (N, 32)
    in_y = (grid >= cells[:, 1:2]) & (grid < cells[:, 3:4])
    return (in_y.unsqueeze(2) & in_x.unsqueeze(1)).float()


class GazelleWrapper:
    """Gaze-LLE gaze-target estimator for multi-person frames.

    Args:
        variant: Key of :data:`GAZELLE_VARIANTS`.
        device_id: Device index; ``None`` or negative for CPU.
        pretrained: Load the released decoder and DINOv2 weights.  ``False``
            builds the architecture with random weights (shape tests, offline).
        autocast: Run the forward pass in reduced precision (fp16 on CUDA, bf16
            elsewhere).  Off by default; about 1.5x faster on MPS, but much slower
            on CPU.

    Raises:
        ValueError: If ``variant`` is unknown.

    Example::

        model = GazelleWrapper(device_id=0)
        out = model(frames, boxes)        # boxes: per frame, normalized xyxy per person
        out.gaze_point()                  # (B, P, 2)

    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        device_id: int | None = None,
        pretrained: bool = True,
        autocast: bool = False,
    ) -> None:
        if variant not in GAZELLE_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {list(GAZELLE_VARIANTS)}.")
        self.device = get_torch_device(device_id)
        spec = GAZELLE_VARIANTS[variant]
        self.variant = variant
        self.autocast = autocast
        backbone = build_hf_model(Dinov2Model, _BACKBONE_IDS[spec.backbone], pretrained=pretrained)
        assert isinstance(backbone, Dinov2Model)
        self.backbone = backbone.eval()
        self.decoder = GazelleDecoder(backbone.config.hidden_size, spec.inout)
        if pretrained:
            path = download_weight(
                spec.weight_file,
                WEIGHT_DIR,
                upstream_url=f"{_UPSTREAM_RELEASE}/{spec.weight_file.rsplit('/', 1)[-1]}",
            )
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.decoder.load_state_dict(state)
        else:
            logger.info("Building Gaze-LLE architecture with random weights (no checkpoint).")
        self.decoder.eval()
        self._to(self.device)
        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)

    def _to(self, device: torch.device) -> None:
        self.device = device
        self.backbone.to(device)
        self.decoder.to(device)

    def preprocess(self, frames) -> torch.Tensor:
        """Resize to 448x448 (aspect ratio not preserved) and normalise.

        Boxes are image-relative, so the distortion does not shift them.

        Args:
            frames: Any input accepted by
                :func:`~exordium.video.core.io.to_uint8_tensor`: uint8 RGB
                tensor or array, or image path(s).

        Returns:
            Float tensor ``(B, 3, 448, 448)`` on ``self.device``.

        """
        x = to_uint8_tensor(frames).to(self.device)
        x = TF.resize(x, [INPUT_SIZE, INPUT_SIZE], antialias=True)
        x = x.float().div(255)
        return (x - self._mean.to(self.device)) / self._std.to(self.device)

    def _forward(
        self, tensor: torch.Tensor, boxes: Sequence[Sequence[NormalizedBox | None]]
    ) -> GazelleOutput:
        counts = [len(b) for b in boxes]
        total = sum(counts)
        max_people = max(counts, default=0)
        dtype = torch.bfloat16 if self.device.type != "cuda" else torch.float16
        with torch.autocast(self.device.type, dtype=dtype, enabled=self.autocast):
            tokens = self.backbone(pixel_values=tensor).last_hidden_state[:, 1:]
            features = tokens.transpose(1, 2).reshape(
                tensor.shape[0], -1, FEATURE_GRID, FEATURE_GRID
            )
            flat_boxes = [box for frame_boxes in boxes for box in frame_boxes]
            scene_index = torch.repeat_interleave(
                torch.arange(len(counts), device=self.device),
                torch.tensor(counts, device=self.device),
            )
            heatmap, inout = self.decoder(
                features, head_maps_from_boxes(flat_boxes, self.device), scene_index
            )
        heatmap, inout = heatmap.float(), None if inout is None else inout.float()

        valid = torch.arange(max_people, device=self.device) < torch.tensor(
            counts, device=self.device
        ).unsqueeze(1)
        padded = heatmap.new_zeros(len(counts), max_people, HEATMAP_SIZE, HEATMAP_SIZE)
        padded[valid] = heatmap.reshape(total, HEATMAP_SIZE, HEATMAP_SIZE)
        padded_inout = None
        if inout is not None:
            padded_inout = inout.new_zeros(len(counts), max_people)
            padded_inout[valid] = inout
        return GazelleOutput(padded, valid, padded_inout)

    def inference(
        self, tensor: torch.Tensor, boxes: Sequence[Sequence[NormalizedBox | None]]
    ) -> GazelleOutput:
        """Run Gaze-LLE on preprocessed frames.

        If the device rejects an op (notably on MPS), the model moves to the CPU
        once, logs a warning, and the call is retried there.

        Args:
            tensor: Output of :meth:`preprocess`, ``(B, 3, 448, 448)``.
            boxes: Per frame, one normalized ``xyxy`` head box per person
                (``None`` for an empty mask).  Out-of-range values are clipped.

        Returns:
            :class:`GazelleOutput` on ``self.device``.

        Raises:
            ValueError: If ``boxes`` does not have one entry per frame.

        """
        if len(boxes) != tensor.shape[0]:
            raise ValueError(f"Expected boxes for {tensor.shape[0]} frames, got {len(boxes)}")
        try:
            return self._forward(tensor, boxes)
        except (RuntimeError, NotImplementedError) as error:
            if self.device.type != "mps":
                raise
            logger.warning(f"MPS failed ({error}); falling back to CPU.")
            self._to(torch.device("cpu"))
            return self._forward(tensor.cpu(), boxes)

    def __call__(self, frames, boxes: Sequence[Sequence[NormalizedBox | None]]) -> GazelleOutput:
        """Preprocess ``frames`` and predict gaze targets for the given head boxes.

        Args:
            frames: Any input accepted by :meth:`preprocess`.
            boxes: Per frame, normalized ``xyxy`` head box per person.

        Returns:
            :class:`GazelleOutput`.

        """
        with torch.inference_mode():
            return self.inference(self.preprocess(frames), boxes)

    def _sample_frame_ids(self, video: Video, fps: float | None) -> list[int]:
        """Frame indices to process: all, or the nearest frame every ``1 / fps`` seconds."""
        if fps is None or fps >= video.fps:
            return list(range(video.num_frames))
        times = torch.arange(0, video.num_frames / video.fps, 1 / fps, dtype=torch.float64)
        indices = torch.round(times * video.fps).long().clamp(max=video.num_frames - 1)
        return torch.unique(indices).tolist()

    def _predict_rows(
        self,
        video: Video,
        frame_ids: list[int],
        boxes: BoxSource,
        batch_size: int,
        store_heatmaps: bool,
        reducers: Mapping[str, Reducer],
    ) -> dict[str, torch.Tensor]:
        """Predict one row per (frame, person) for ``frame_ids``, all boxed."""
        columns: dict[str, list[torch.Tensor]] = {}

        def collect(name: str, value: torch.Tensor) -> None:
            columns.setdefault(name, []).append(value.cpu())

        for batch_ids in batch_iterator(frame_ids, batch_size):
            per_frame = [_frame_boxes(boxes, f) for f in batch_ids]
            frames = video.get_frames_at(batch_ids)
            people = [list(frame_boxes) for frame_boxes in per_frame]
            out = self(frames, [[fb[p] for p in ids] for fb, ids in zip(per_frame, people)])
            valid = out.valid
            rows_frame = torch.tensor(batch_ids).unsqueeze(1).expand_as(valid)[valid.cpu()]
            rows_person = torch.tensor([p for ids in people for p in ids])
            heatmaps = out.heatmap[valid]
            collect("frame_ids", rows_frame)
            collect("person_ids", rows_person)
            collect("gaze_points", out.gaze_point()[valid])
            if out.inout is not None:
                collect("inout", out.inout[valid])
            if store_heatmaps:
                collect("heatmaps", heatmaps)
            for name, reducer in reducers.items():
                collect(name, reducer(heatmaps, rows_frame, rows_person))
        return {name: torch.cat(parts) for name, parts in columns.items()}

    @load_or_create("st")
    def video_to_gaze(
        self,
        video_path: str | Path,
        boxes: BoxSource,
        fps: float | None = None,
        batch_size: int = 8,
        store_heatmaps: bool = False,
        reducers: Mapping[str, Reducer] | None = None,
        chunk_frames: int = 256,
        chunk_dir: str | Path | None = None,
        verbose: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predict gaze targets for every boxed person in a video.

        One row per (frame, person).  Frames without any box are skipped, never
        guessed.  Pass ``output_path`` (and ``overwrite``) to cache the result as
        safetensors; pass ``chunk_dir`` to also persist progress every
        ``chunk_frames`` frames so an interrupted run resumes where it stopped.
        Chunk files carry a digest of their inputs, so changed boxes, variant or
        outputs recompute instead of reusing old chunks.

        Full heatmaps are large, so by default only gaze points and in/out scores
        are kept; use ``reducers`` for per-row values computed from the heatmaps
        (e.g. :func:`~exordium.video.face.gaze.base.mass_in_box` on a partner's
        box) or ``store_heatmaps`` to keep them all.

        Args:
            video_path: Input video.
            boxes: Head boxes per frame id (``{frame_id: {person_id: box}}`` or a
                callable returning that for one frame); see
                :func:`boxes_from_tracks`.
            fps: Subsample to about this rate; ``None`` uses every frame.
                ``frame_ids`` in the result always refer to the original video.
            batch_size: Frames per forward pass.
            store_heatmaps: Also return the ``(N, 64, 64)`` heatmaps.
            reducers: Named callables computing extra per-row columns.
            chunk_frames: Frames per resumable chunk.
            chunk_dir: Directory for chunk files; ``None`` disables resuming.
            verbose: Log progress per chunk.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Dict of CPU tensors: ``frame_ids (N,)``, ``person_ids (N,)``,
            ``gaze_points (N, 2)`` (argmax), ``inout (N,)`` when the variant has
            one, ``heatmaps`` if requested, and one entry per reducer.

        """
        reducers = reducers or {}
        chunk_path = None if chunk_dir is None else Path(chunk_dir)
        parts: list[dict[str, torch.Tensor]] = []
        with Video(video_path) as video:
            wanted = [f for f in self._sample_frame_ids(video, fps) if _frame_boxes(boxes, f)]
            for number, chunk in enumerate(batch_iterator(wanted, chunk_frames)):
                saved = None
                if chunk_path is not None:
                    key = self._chunk_key(chunk, boxes, store_heatmaps, reducers)
                    saved = chunk_path / f"chunk_{number:06d}_{key}.st"
                if saved is not None and saved.exists():
                    parts.append(load_file(str(saved)))
                    continue
                if verbose:
                    logger.info(f"Gaze chunk {number}: frames {chunk[0]}..{chunk[-1]}")
                rows = self._predict_rows(video, chunk, boxes, batch_size, store_heatmaps, reducers)
                if saved is not None:
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    save_file(rows, str(saved))
                parts.append(rows)
        if not parts:
            return self._empty_result(store_heatmaps, reducers)
        return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}

    def _chunk_key(
        self,
        frame_ids: list[int],
        boxes: BoxSource,
        store_heatmaps: bool,
        reducers: Mapping[str, Reducer],
    ) -> str:
        """Short digest of everything that determines a chunk's rows.

        Part of the chunk file name, so a ``chunk_dir`` reused after changing the
        variant, boxes, frames or outputs never serves stale rows.
        """
        per_frame = [(f, sorted(_frame_boxes(boxes, f).items())) for f in frame_ids]
        state = (self.variant, store_heatmaps, sorted(reducers), per_frame)
        return hashlib.sha1(repr(state).encode()).hexdigest()[:10]

    def _empty_result(
        self, store_heatmaps: bool, reducers: Mapping[str, Reducer]
    ) -> dict[str, torch.Tensor]:
        """Result for a video with no boxed frames."""
        result = {
            "frame_ids": torch.zeros(0, dtype=torch.long),
            "person_ids": torch.zeros(0, dtype=torch.long),
            "gaze_points": torch.zeros(0, 2),
        }
        if GAZELLE_VARIANTS[self.variant].inout:
            result["inout"] = torch.zeros(0)
        if store_heatmaps:
            result["heatmaps"] = torch.zeros(0, HEATMAP_SIZE, HEATMAP_SIZE)
        result.update({name: torch.zeros(0) for name in reducers})
        return result
