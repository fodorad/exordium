"""Scatter sparse per-detection features onto a dense, regular time grid.

Feature extractors return one vector per *detection*, not per *frame*: a face
track covers only the frames where the subject was actually detected, so
:meth:`~exordium.video.deep.base.VisualModelWrapper.track_to_feature` yields
``(N, D)`` with ``N <= num_frames``. That sparse form is the right default —
it wastes nothing and stays honest about what was observed.

It is the wrong form for a *temporal model*. A sequence head consuming face,
audio and text streams needs them on a common, gap-free time axis, so timestep
*i* means the same instant in every modality. :func:`densify` performs that
conversion: it scatters the sparse features onto the frame grid, fills the
frames with no detection with a constant vector, and returns a boolean mask
saying which timesteps were real.

The mask is the load-bearing part. Zero-filled timesteps are *not* observations
of a zero-valued face; they are absences, and a model must be told to ignore
them (attention masking, packed sequences, masked pooling) rather than fitting
them. A dense tensor without its mask silently teaches the model that "no face
detected" is a meaningful feature value.

The output contract — ``frame_ids`` / ``features`` / ``mask`` — is deliberately
identical to the one
:meth:`~exordium.video.deep.marlin.MarlinWrapper.track_to_feature` already
produces, so densified frame-wise features and MARLIN's window-wise features
are interchangeable downstream.

.. note::

   Every **frame-wise** extractor inherits ``densify=`` from
   :meth:`~exordium.video.deep.base.VisualModelWrapper.track_to_feature` —
   FaRL, EmotiEffNet, FabNet, CLIP, DINOv2, AdaFace, SwinT, OpenGraphAU and
   SixDRepNet.

   :class:`~exordium.video.deep.marlin.MarlinWrapper` does **not** take the
   argument, and does not need it: MARLIN is a *video* model whose timestep is
   a 16-frame **window**, not a frame, so its ``frame_ids`` are window start
   indices (``0, 16, 32, ...``). It already emits this exact dense contract
   natively, zero-filling empty windows. Densifying it would require answering
   "which frame is window 2?", which has no honest answer — aligning MARLIN to
   a frame grid means *upsampling* its window features, a different operation.
"""

import torch

__all__ = ["densify"]


def densify(
    frame_ids: torch.Tensor,
    features: torch.Tensor,
    start_frame_id: int = 0,
    end_frame_id: int | None = None,
    fill: torch.Tensor | float = 0.0,
    strict: bool = False,
) -> dict[str, torch.Tensor]:
    """Scatter sparse per-detection features onto a dense frame grid.

    Builds the half-open frame range ``[start_frame_id, end_frame_id)``, places
    each feature at its own frame, and fills the rest with ``fill``.

    Densifying a *window* rather than the whole video keeps this O(window)
    instead of O(video): a long track through a 10-minute recording can be
    densified over a 250-frame snippet without materialising the full timeline.
    Whole-video densification is just the special case
    ``start_frame_id=0, end_frame_id=num_frames``.

    Args:
        frame_ids: ``(N,)`` long tensor of frame indices, as returned by
            ``track_to_feature``. Need not be sorted or contiguous.
        features: ``(N, D)`` float tensor of per-detection features, row-aligned
            with ``frame_ids``.
        start_frame_id: First frame of the output grid, inclusive. Defaults to 0.
        end_frame_id: One past the last frame of the output grid (exclusive).
            When ``None``, defaults to ``max(frame_ids) + 1`` — which
            **undercounts if the video continues past the last detection**, so
            pass it explicitly whenever the true length is known.
        fill: Value for frames with no detection. Either a scalar (broadcast
            across the feature dimension) or a ``(D,)`` tensor. Defaults to 0.0.
        strict: If True, raise when a frame id falls outside the output range.
            If False (default), such detections are dropped — a long track
            legitimately extends past a short window, so clipping is the normal
            case, not an error.

    Returns:
        Dict with, where ``T = end_frame_id - start_frame_id``:

        * ``"frame_ids"`` — ``(T,)`` long tensor, ``start_frame_id`` to
          ``end_frame_id - 1``. Absolute frame indices, not offsets into the
          window.
        * ``"features"`` — ``(T, D)`` tensor, ``fill`` where no detection
          exists. Same dtype as ``features``.
        * ``"mask"`` — ``(T,)`` bool tensor. ``True`` where a real detection
          landed on that frame, ``False`` where the value is ``fill``.

    Raises:
        ValueError: If ``features`` is not 2-D, if ``frame_ids`` and ``features``
            disagree on length, if ``end_frame_id <= start_frame_id``, if ``fill``
            is a tensor whose shape is not ``(D,)``, or — when ``strict`` — if any
            frame id falls outside ``[start_frame_id, end_frame_id)``.

    Example::

        from exordium.video.core.densify import densify

        sparse = model.track_to_feature(track)          # (N, D), N <= num_frames
        dense = densify(
            sparse["frame_ids"],
            sparse["features"],
            end_frame_id=num_frames,                    # full video
        )
        dense["features"]  # (num_frames, D), zeros where no face was detected
        dense["mask"]      # (num_frames,) bool — feed this to your temporal model

        # A 10-second snippet at 25 fps, without materialising the whole video:
        window = densify(
            sparse["frame_ids"], sparse["features"],
            start_frame_id=100, end_frame_id=350,
        )
        window["features"].shape  # (250, D)

    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (N, D), got shape {tuple(features.shape)}.")
    if frame_ids.shape[0] != features.shape[0]:
        raise ValueError(
            f"frame_ids and features disagree on length: "
            f"{frame_ids.shape[0]} ids vs {features.shape[0]} feature rows."
        )

    feature_dim = features.shape[1]

    if end_frame_id is None:
        # No explicit end: span up to the last detection. Undercounts a video that
        # continues past it, which is why the docstring pushes callers to pass it.
        end_frame_id = int(frame_ids.max()) + 1 if frame_ids.numel() else start_frame_id

    if end_frame_id < start_frame_id:
        raise ValueError(
            f"end_frame_id ({end_frame_id}) must be >= start_frame_id ({start_frame_id})."
        )

    num_timesteps = end_frame_id - start_frame_id

    if isinstance(fill, torch.Tensor):
        if fill.shape != (feature_dim,):
            raise ValueError(
                f"fill tensor must have shape ({feature_dim},) to match the feature "
                f"dimension, got {tuple(fill.shape)}."
            )
        dense = fill.to(dtype=features.dtype, device=features.device).expand(
            num_timesteps, feature_dim
        )
        dense = dense.clone()  # expand() gives a read-only view; scatter needs to write.
    else:
        dense = torch.full(
            (num_timesteps, feature_dim),
            float(fill),
            dtype=features.dtype,
            device=features.device,
        )

    mask = torch.zeros(num_timesteps, dtype=torch.bool, device=features.device)

    in_range = (frame_ids >= start_frame_id) & (frame_ids < end_frame_id)
    if strict and not bool(in_range.all()):
        out_of_range = frame_ids[~in_range]
        raise ValueError(
            f"{out_of_range.numel()} frame id(s) outside "
            f"[{start_frame_id}, {end_frame_id}), e.g. {out_of_range[:5].tolist()}."
        )

    # Absolute frame ids -> offsets into the window.
    positions = frame_ids[in_range].to(device=features.device) - start_frame_id
    dense[positions] = features[in_range.to(features.device)]
    mask[positions] = True

    return {
        "frame_ids": torch.arange(
            start_frame_id, end_frame_id, dtype=torch.long, device=features.device
        ),
        "features": dense,
        "mask": mask,
    }
