"""Abstract base class for frame-wise visual feature extractors."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import torch
from tqdm import tqdm

from exordium.utils.decorator import load_or_create
from exordium.utils.device import get_torch_device
from exordium.video.core.densify import densify as densify_features
from exordium.video.core.detection import Track
from exordium.video.core.io import Video, batch_iterator, to_uint8_tensor

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
"""ImageNet RGB channel means used for normalisation."""
_IMAGENET_STD = [0.229, 0.224, 0.225]
"""ImageNet RGB channel standard deviations used for normalisation."""


class VisualModelWrapper(ABC):
    """Abstract base class for frame-wise visual feature extractors.

    Subclasses must implement :meth:`preprocess` (any supported input →
    model-ready tensor) and :meth:`inference` (model tensor → feature tensor).

    Supported input types for :meth:`__call__`:

    * ``torch.Tensor`` — ``(C, H, W)`` or ``(B, C, H, W)`` uint8 RGB; fastest
      path, no copies made until device transfer.
    * ``np.ndarray`` — ``(H, W, 3)`` or ``(B, H, W, 3)`` uint8 RGB.
    * ``Sequence[np.ndarray]`` — list of ``(H, W, 3)`` uint8 arrays.
    * ``Sequence[str | Path]`` — list of image file paths.

    Design contract:

    * :meth:`preprocess` — abstract; converts any supported input to a
      model-ready float tensor on ``self.device``.
    * :meth:`inference` — abstract; pure model forward pass; input already on
      ``self.device``.
    * :meth:`__call__` — ``preprocess`` → ``inference`` under
      ``torch.inference_mode``; returns a ``torch.Tensor``.
    * :meth:`dir_to_feature` / :meth:`track_to_feature` /
      :meth:`video_to_feature` — cached batch helpers.

    Example::

        class MyWrapper(VisualModelWrapper):
            def __init__(self, device_id=None):
                super().__init__(device_id)
                self.model = ...

            def preprocess(self, frames):
                x = self._to_uint8_tensor(frames).to(self.device)
                return (x.float() / 255.0 - MEAN) / STD

            def inference(self, tensor):
                return self.model(tensor)

        model = MyWrapper(device_id=0)
        tensor_out = model(video_tensor)   # torch.Tensor (B, D)

    """

    def __init__(self, device_id: int | None = None) -> None:
        self.device = get_torch_device(device_id)

    @staticmethod
    def _to_uint8_tensor(
        frames: torch.Tensor | Sequence,
    ) -> torch.Tensor:
        """Convert any supported input to a uint8 ``(B, 3, H, W)`` CPU tensor.

        Delegates to :func:`~exordium.video.core.io.to_uint8_tensor`.

        Args:
            frames: One of:

                * ``torch.Tensor (C, H, W)`` or ``(B, C, H, W)`` uint8
                * ``np.ndarray (H, W, 3)`` or ``(B, H, W, 3)`` uint8
                * ``str | Path`` — single image file path
                * ``Sequence[np.ndarray]`` of ``(H, W, 3)`` arrays
                * ``Sequence[str | Path]`` of image file paths

        Returns:
            uint8 tensor of shape ``(B, 3, H, W)`` on CPU.

        """
        return to_uint8_tensor(frames)

    @abstractmethod
    def preprocess(
        self,
        frames: torch.Tensor | Sequence,
    ) -> torch.Tensor:
        """Convert any supported input to a model-ready tensor.

        Call :meth:`_to_uint8_tensor` first to normalise the input type, then
        apply model-specific resize, crop, and normalisation as tensor
        operations.  The returned tensor must be on ``self.device``.

        Args:
            frames: Any supported input (see class docstring).

        Returns:
            Preprocessed float tensor on ``self.device``.

        """

    @abstractmethod
    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        The input tensor is already on ``self.device`` and preprocessed.

        Args:
            tensor: Float tensor of shape ``(B, …)`` on ``self.device``.

        Returns:
            Feature tensor of shape ``(B, D)``.

        """

    def __call__(
        self,
        frames: torch.Tensor | Sequence,
    ) -> torch.Tensor:
        """Preprocess and run inference, returning a feature tensor.

        Args:
            frames: Any supported input (see class docstring).

        Returns:
            Feature tensor of shape ``(B, D)`` on ``self.device``.

        """
        with torch.inference_mode():
            return self.inference(self.preprocess(frames))

    @load_or_create("st")
    def dir_to_feature(
        self,
        img_paths: list[str | Path],
        batch_size: int = 30,
        verbose: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract features from an ordered list of image files.

        Results are cached as a safetensors file: pass ``output_path`` and
        ``overwrite`` via ``_kwargs`` to control caching behaviour.

        Args:
            img_paths: Image file paths. Each file stem must be parseable
                as an integer frame ID.
            batch_size: Images to process per batch. Defaults to 30.
            verbose: Show a progress bar. Defaults to False.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Dict with keys ``"frame_ids"`` (``(N,)`` long tensor) and
            ``"features"`` (``(N, D)`` float tensor), both on CPU.

        """
        ids: list[int] = []
        features: list[torch.Tensor] = []

        for i in tqdm(
            range(0, len(img_paths), batch_size),
            total=-(-len(img_paths) // batch_size),
            desc=f"{type(self).__name__} extraction",
            disable=not verbose,
        ):
            batch = img_paths[i : i + batch_size]
            ids += [int(Path(p).stem) for p in batch]
            features.append(self(batch).cpu())

        return {
            "frame_ids": torch.tensor(ids, dtype=torch.long),
            "features": torch.cat(features, dim=0),
        }

    @load_or_create("st")
    def track_to_feature(
        self,
        track: Track,
        batch_size: int = 30,
        densify: bool = False,
        start_frame_id: int = 0,
        end_frame_id: int | None = None,
        fill: torch.Tensor | float = 0.0,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract features from all non-interpolated detections in a track.

        By default the output is **sparse**: one feature row per detection, so
        ``N <= num_frames`` when the subject was not detected on every frame.
        That is the honest default — it records only what was observed.

        Set ``densify=True`` to scatter those features onto a gap-free frame
        grid instead, which is what a temporal model needs to keep the face
        stream aligned with audio and text. Frames with no detection are filled
        with ``fill`` and flagged ``False`` in the returned ``mask``.

        Results are cached as a safetensors file: pass ``output_path`` and
        ``overwrite`` via ``_kwargs`` to control caching behaviour. Sparse and
        dense results differ, so give them **different** ``output_path``s.

        Args:
            track: Track containing a sequence of Detection objects.
            batch_size: Detections to process per batch. Defaults to 30.
            densify: If True, return a dense grid plus a validity mask. See
                :func:`~exordium.video.core.densify.densify`.
            start_frame_id: First frame of the dense grid, inclusive. Ignored
                unless ``densify``. Defaults to 0.
            end_frame_id: One past the last frame of the dense grid (exclusive).
                Ignored unless ``densify``. When ``None``, defaults to the last
                detected frame + 1, which **undercounts if the video continues
                past the last detection** — pass the video's frame count
                explicitly whenever it is known.
            fill: Value for frames with no detection — either a scalar or a
                ``(D,)`` tensor. Ignored unless ``densify``. Defaults to 0.0.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            When ``densify=False`` (default), a dict with ``"frame_ids"``
            (``(N,)`` long) and ``"features"`` (``(N, D)`` float), both on CPU.

            When ``densify=True``, a dict with ``"frame_ids"`` (``(T,)`` long,
            contiguous), ``"features"`` (``(T, D)``, ``fill`` where undetected)
            and ``"mask"`` (``(T,)`` bool, True where a detection existed) —
            the same contract
            :meth:`~exordium.video.deep.marlin.MarlinWrapper.track_to_feature`
            returns.

        """
        ids: list[int] = []
        features: list[torch.Tensor] = []

        for subset in batch_iterator(track, batch_size):
            if not subset:
                continue
            ids += [d.frame_id for d in subset]
            features.append(self([d.crop(square=True, extra_space=1.5) for d in subset]).cpu())

        frame_ids = torch.tensor(ids, dtype=torch.long)
        # An empty track yields no batches, so there is nothing to concatenate. The
        # feature width is still needed to build a well-formed (0, D) tensor, and it
        # is only knowable by asking the model — not every wrapper exposes it as an
        # attribute. One forward pass on a dummy crop is a cheap price for an edge case.
        if features:
            stacked = torch.cat(features, dim=0)
        else:
            probe = self(torch.zeros(1, 3, 224, 224, dtype=torch.uint8)).cpu()
            stacked = probe.new_zeros((0, probe.shape[-1]))

        if not densify:
            return {"frame_ids": frame_ids, "features": stacked}

        return densify_features(
            frame_ids,
            stacked,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            fill=fill,
        )

    @load_or_create("st")
    def video_to_feature(
        self,
        video_path: str | Path,
        batch_size: int = 30,
        verbose: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract features for every frame of a video file.

        Opens the video once with :class:`~exordium.video.core.io.Video`,
        iterates over batches of decoded frames, and concatenates the results.
        Results are cached as a safetensors file: pass ``output_path`` and
        ``overwrite`` via ``_kwargs`` to control caching behaviour.

        Args:
            video_path: Path to the input video file.
            batch_size: Frames to process per batch. Defaults to 30.
            verbose: Show a progress bar. Defaults to False.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Dict with keys ``"frame_ids"`` (``(T,)`` long tensor) and
            ``"features"`` (``(T, D)`` float tensor), both on CPU.

        """
        video_path = Path(video_path)
        ids: list[int] = []
        features: list[torch.Tensor] = []
        frame_id = 0

        with Video(video_path) as video:
            total_batches = -(-video.num_frames // batch_size)
            for batch in tqdm(
                video.iter_batches(batch_size=batch_size),
                total=total_batches,
                desc=f"{type(self).__name__} extraction",
                disable=not verbose,
            ):
                features.append(self(batch).cpu())
                ids += list(range(frame_id, frame_id + len(batch)))
                frame_id += len(batch)

        return {
            "frame_ids": torch.tensor(ids, dtype=torch.long),
            "features": torch.cat(features, dim=0),
        }
