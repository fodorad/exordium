"""Abstract base class for frame-wise visual feature extractors."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from exordium.utils.decorator import load_or_create
from exordium.utils.device import get_torch_device
from exordium.video.core.detection import Track
from exordium.video.core.io import Video, batch_iterator, image_to_np


class VisualModelWrapper(ABC):
    """Abstract base class for frame-wise visual feature extractors.

    Subclasses must implement :meth:`_preprocess` (numpy frames → model
    tensor) and :meth:`inference` (model tensor → feature tensor).  The
    shared :meth:`predict`, :meth:`dir_to_feature`, :meth:`track_to_feature`,
    and :meth:`video_to_feature` methods are provided here.

    Design contract:

    * :meth:`__call__` — strict tensor → tensor interface with
      ``torch.inference_mode`` applied.  The input tensor is moved to
      ``self.device`` automatically.
    * :meth:`inference` — abstract model forward pass.  Called by
      ``__call__`` after device placement.
    * :meth:`_preprocess` — abstract numpy-to-tensor conversion.  Called
      by :meth:`predict` before inference.
    * :meth:`predict` — convenience wrapper: numpy/path inputs → numpy
      output.  Handles loading, preprocessing, and device transfer.
    * :meth:`dir_to_feature` / :meth:`track_to_feature` / :meth:`video_to_feature`
      — cached batch extraction helpers built on top of :meth:`predict`.

    Example::

        class MyWrapper(VisualModelWrapper):
            def __init__(self, device_id=None):
                super().__init__(device_id)
                self.model = ...
                self.transform = ...

            def _preprocess(self, frames):
                return torch.stack([self.transform(f) for f in frames]).to(self.device)

            def inference(self, tensor):
                return self.model(tensor)

        model = MyWrapper(device_id=0)
        tensor_out = model(preprocessed_tensor)          # torch.Tensor
        array_out  = model.predict(numpy_frames)         # np.ndarray
        ids, feats = model.dir_to_feature(paths)         # cached
        ids, feats = model.track_to_feature(track)       # cached
        ids, feats = model.video_to_feature(video_path)  # cached

    """

    def __init__(self, device_id: int | None = None) -> None:
        """Initialise device.

        Args:
            device_id: GPU device index. ``None`` or negative uses CPU.

        """
        self.device = get_torch_device(device_id)

    @abstractmethod
    def _preprocess(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert a list of RGB numpy frames to a model-input tensor.

        The returned tensor should be on ``self.device`` and have whatever
        shape the concrete model expects (typically ``(B, C, H, W)``).

        Args:
            frames: Sequence of RGB uint8 arrays, each of shape
                ``(H, W, 3)``.

        Returns:
            Preprocessed batch tensor on ``self.device``.

        """

    @abstractmethod
    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        The input tensor is already on ``self.device``.

        Args:
            tensor: Preprocessed batch tensor of shape ``(B, …)``.

        Returns:
            Feature tensor of shape ``(B, D)``.

        """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Strict tensor → tensor interface.

        Accepts a single image tensor ``(C, H, W)`` or a batched tensor
        ``(B, C, H, W)``.  The tensor is moved to ``self.device`` and
        ``inference`` is run under ``torch.inference_mode``.

        Args:
            tensor: Image tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``
                on any device.

        Returns:
            Feature tensor of shape ``(1, D)`` or ``(B, D)`` on
            ``self.device``.

        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        with torch.inference_mode():
            return self.inference(tensor.to(self.device))

    def predict(
        self,
        frames: np.ndarray | str | Path | Sequence[np.ndarray | str | Path],
    ) -> np.ndarray:
        """Numpy / path inputs → numpy feature array.

        Accepts a single image (``np.ndarray`` or path) or a sequence of
        images.  Loads from disk when paths are provided, preprocesses,
        runs inference under ``torch.inference_mode``, and returns a CPU
        numpy array.

        Args:
            frames: A single RGB ``np.ndarray`` ``(H, W, 3)``, a file path,
                or a sequence of either.

        Returns:
            Feature array of shape ``(B, D)``.

        """
        if isinstance(frames, (np.ndarray, str, Path)):
            frames = [frames]
        frames_np = [image_to_np(f, "RGB") if isinstance(f, (str, Path)) else f for f in frames]
        with torch.inference_mode():
            return self.inference(self._preprocess(frames_np)).cpu().numpy()

    @load_or_create("pkl")
    def dir_to_feature(
        self,
        img_paths: list[str | Path],
        batch_size: int = 30,
        verbose: bool = False,
        **_kwargs,
    ) -> tuple[list[int], np.ndarray]:
        """Extract features from an ordered list of image files.

        Results are cached: pass ``output_path`` and ``overwrite`` via
        ``_kwargs`` to control caching behaviour.

        Args:
            img_paths: Image file paths. Each file stem must be parseable
                as an integer frame ID.
            batch_size: Images to process per batch. Defaults to 30.
            verbose: Show a progress bar. Defaults to False.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Tuple of ``(frame_ids, features)`` where ``features`` has shape
            ``(N, D)``.

        """
        ids: list[int] = []
        features: list[np.ndarray] = []

        for i in tqdm(
            range(0, len(img_paths), batch_size),
            total=-(-len(img_paths) // batch_size),
            desc=f"{type(self).__name__} extraction",
            disable=not verbose,
        ):
            batch = img_paths[i : i + batch_size]
            ids += [int(Path(p).stem) for p in batch]
            features.append(self.predict(batch))

        return ids, np.concatenate(features, axis=0)

    @load_or_create("pkl")
    def track_to_feature(
        self,
        track: Track,
        batch_size: int = 30,
        **_kwargs,
    ) -> tuple[list[int], np.ndarray]:
        """Extract features from all non-interpolated detections in a track.

        Results are cached: pass ``output_path`` and ``overwrite`` via
        ``_kwargs`` to control caching behaviour.

        Args:
            track: Track containing a sequence of Detection objects.
            batch_size: Detections to process per batch. Defaults to 30.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Tuple of ``(frame_ids, features)`` where ``features`` has shape
            ``(N, D)``.

        """
        ids: list[int] = []
        features: list[np.ndarray] = []

        for subset in batch_iterator(track, batch_size):
            valid = [d for d in subset if not d.is_interpolated]
            if not valid:
                continue
            ids += [d.frame_id for d in valid]
            features.append(self.predict([d.bb_crop_wide() for d in valid]))

        return ids, np.concatenate(features, axis=0)

    @load_or_create("pkl")
    def video_to_feature(
        self,
        video_path: str | Path,
        batch_size: int = 30,
        verbose: bool = False,
        **_kwargs,
    ) -> tuple[list[int], np.ndarray]:
        """Extract features for every frame of a video file.

        Opens the video once with :class:`~exordium.video.core.io.Video`,
        iterates over batches of decoded frames, and concatenates the results.
        Results are cached: pass ``output_path`` and ``overwrite`` via
        ``_kwargs`` to control caching behaviour.

        Args:
            video_path: Path to the input video file.
            batch_size: Frames to process per batch. Defaults to 30.
            verbose: Show a progress bar. Defaults to False.
            **_kwargs: Forwarded to ``load_or_create``.

        Returns:
            Tuple of ``(frame_ids, features)`` where ``frame_ids`` are
            zero-based indices and ``features`` has shape ``(T, D)``.

        """
        video_path = Path(video_path)
        ids: list[int] = []
        features: list[np.ndarray] = []
        frame_id = 0

        with Video(video_path) as video:
            total_batches = -(-video.num_frames // batch_size)
            for batch in tqdm(
                video.iter_batches(batch_size=batch_size),
                total=total_batches,
                desc=f"{type(self).__name__} extraction",
                disable=not verbose,
            ):
                # batch: list of tensors (C, H, W) in uint8 RGB
                frames = [f.permute(1, 2, 0).cpu().numpy() for f in batch]
                features.append(self.predict(frames))
                ids += list(range(frame_id, frame_id + len(frames)))
                frame_id += len(frames)

        return ids, np.concatenate(features, axis=0)
