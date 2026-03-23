"""Bounding box utility functions for format conversion, cropping and IoU computation.

Tensor format conventions
--------------------------
* ``torch.Tensor`` images: shape ``(C, H, W)`` uint8 RGB — **channel-first**.
* ``np.ndarray`` images:   shape ``(H, W, C)`` uint8 RGB — **channel-last**.

Bounding box format notation
------------------------------
* ``xyxy``  — ``[x_min, y_min, x_max, y_max]`` pixel coords (top-left / bottom-right).
* ``xywh``  — ``[x_min, y_min, width, height]``.
* ``midwh`` — ``[x_center, y_center, width, height]``.

All format-conversion functions are **type-preserving**: a numpy input returns
numpy, a torch tensor input returns a torch tensor of the same device.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _hw(image: np.ndarray | torch.Tensor) -> tuple[int, int]:
    """Return (height, width) regardless of channel layout."""
    if isinstance(image, torch.Tensor):
        return int(image.shape[-2]), int(image.shape[-1])  # (C, H, W)
    return int(image.shape[0]), int(image.shape[1])  # (H, W, C)


def _slice(image: np.ndarray | torch.Tensor, y1, y2, x1, x2):
    """Slice spatial dims, preserving channel layout."""
    if isinstance(image, torch.Tensor):
        return image[:, y1:y2, x1:x2]
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Crop functions  (type-preserving: numpy ↔ tensor)
# ---------------------------------------------------------------------------


def crop_mid(
    image: np.ndarray | torch.Tensor,
    mid: np.ndarray | torch.Tensor,
    bb_size: int,
) -> np.ndarray | torch.Tensor:
    """Crop a square region from an image using a center point.

    Input type is preserved: numpy in → numpy out, tensor in → tensor out.

    Args:
        image: ``(H, W, C)`` uint8 RGB numpy array **or** ``(C, H, W)`` uint8
            RGB torch tensor.
        mid: Center point ``[x, y]`` of shape ``(2,)``.
        bb_size: Side length of the square crop in pixels.

    Returns:
        Cropped image in the same format as the input, clipped to image bounds.

    """
    h, w = _hw(image)
    half = bb_size // 2
    x, y = int(mid[0]), int(mid[1])
    x1, y1 = max(0, min(x - half, w)), max(0, min(y - half, h))
    x2, y2 = max(0, min(x + half, w)), max(0, min(y + half, h))
    return _slice(image, y1, y2, x1, x2)


def crop_xyxy(
    image: np.ndarray | torch.Tensor,
    bb_xyxy: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Crop a region from an image using xyxy bounding box coordinates.

    Input type is preserved: numpy in → numpy out, tensor in → tensor out.

    Args:
        image: ``(H, W, C)`` uint8 RGB numpy array **or** ``(C, H, W)`` uint8
            RGB torch tensor.
        bb_xyxy: Bounding box ``[x_min, y_min, x_max, y_max]`` of shape ``(4,)``.

    Returns:
        Cropped image in the same format as the input, clipped to image bounds.

    """
    h, w = _hw(image)
    x1 = max(0, min(int(bb_xyxy[0]), w))
    y1 = max(0, min(int(bb_xyxy[1]), h))
    x2 = max(0, min(int(bb_xyxy[2]), w))
    y2 = max(0, min(int(bb_xyxy[3]), h))
    return _slice(image, y1, y2, x1, x2)


def center_crop(
    img: np.ndarray | torch.Tensor,
    crop_size: tuple[int, int],
) -> np.ndarray | torch.Tensor:
    """Apply a center crop to an image.

    Input type is preserved: numpy in → numpy out, tensor in → tensor out.

    Args:
        img: ``(H, W, C)`` uint8 RGB numpy array **or** ``(C, H, W)`` uint8
            RGB torch tensor.
        crop_size: Target ``(height, width)`` in pixels.

    Returns:
        Center-cropped image in the same format as the input.

    """
    if isinstance(img, torch.Tensor):
        return TF.center_crop(img, list(crop_size))
    c_h, c_w = img.shape[0] // 2, img.shape[1] // 2
    half_h, half_w = crop_size[0] // 2, crop_size[1] // 2
    return img[c_h - half_h : c_h + half_h, c_w - half_w : c_w + half_w, :]


def apply_10_crop(
    img: np.ndarray | torch.Tensor,
    crop_size: tuple[int, int],
) -> np.ndarray | torch.Tensor:
    """Apply 10-crop augmentation to an image.

    Produces four corner crops plus a center crop, each for the original and
    its horizontal mirror — 10 crops in total.

    Input type is preserved: numpy in → numpy out, tensor in → tensor out.

    Args:
        img: ``(H, W, C)`` uint8 RGB numpy array **or** ``(C, H, W)`` uint8
            RGB torch tensor.
        crop_size: Target ``(height, width)`` for each crop.

    Returns:
        * tensor: ``(10, C, crop_h, crop_w)``
        * numpy:  ``(10, crop_h, crop_w, C)``

    """
    ch, cw = crop_size
    ih, iw = _hw(img)

    if isinstance(img, torch.Tensor):
        flip = TF.hflip(img)
        crops = [
            img[:, :ch, :cw],
            img[:, :ch, iw - cw :],
            img[:, ih - ch :, :cw],
            img[:, ih - ch :, iw - cw :],
            TF.center_crop(img, [ch, cw]),
            flip[:, :ch, :cw],
            flip[:, :ch, iw - cw :],
            flip[:, ih - ch :, :cw],
            flip[:, ih - ch :, iw - cw :],
            TF.center_crop(flip, [ch, cw]),
        ]
        return torch.stack(crops)

    flip = img[:, ::-1, :]
    crops = [
        img[:ch, :cw, :],
        img[:ch, iw - cw :, :],
        img[ih - ch :, :cw, :],
        img[ih - ch :, iw - cw :, :],
        center_crop(img, (ch, cw)),
        flip[:ch, :cw, :],
        flip[:ch, iw - cw :, :],
        flip[ih - ch :, :cw, :],
        flip[ih - ch :, iw - cw :, :],
        center_crop(flip, (ch, cw)),
    ]
    return np.stack(crops)


# ---------------------------------------------------------------------------
# IoU  (type-agnostic — scalar float output)
# ---------------------------------------------------------------------------


def iou_xywh(
    bb_xywh1: np.ndarray | torch.Tensor,
    bb_xywh2: np.ndarray | torch.Tensor,
) -> float:
    """Calculate IoU for two bounding boxes in xywh format.

    Accepts both numpy arrays and torch tensors; always returns a Python float.

    Args:
        bb_xywh1: First bounding box ``[x_min, y_min, width, height]`` of
            shape ``(4,)``.
        bb_xywh2: Second bounding box ``[x_min, y_min, width, height]`` of
            shape ``(4,)``.

    Returns:
        Intersection-over-Union in ``[0.0, 1.0]``.

    """
    bb_xyxy1 = xywh2xyxy(bb_xywh1)
    bb_xyxy2 = xywh2xyxy(bb_xywh2)

    x1 = max(float(bb_xyxy1[0]), float(bb_xyxy2[0]))
    y1 = max(float(bb_xyxy1[1]), float(bb_xyxy2[1]))
    x2 = min(float(bb_xyxy1[2]), float(bb_xyxy2[2]))
    y2 = min(float(bb_xyxy1[3]), float(bb_xyxy2[3]))

    area1 = float(bb_xywh1[2]) * float(bb_xywh1[3])
    area2 = float(bb_xywh2[2]) * float(bb_xywh2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def iou_xyxy(
    bb_xyxy1: np.ndarray | torch.Tensor,
    bb_xyxy2: np.ndarray | torch.Tensor,
) -> float:
    """Calculate IoU for two bounding boxes in xyxy format.

    Accepts both numpy arrays and torch tensors; always returns a Python float.

    Args:
        bb_xyxy1: First bounding box ``[x_min, y_min, x_max, y_max]`` of
            shape ``(4,)``.
        bb_xyxy2: Second bounding box ``[x_min, y_min, x_max, y_max]`` of
            shape ``(4,)``.

    Returns:
        Intersection-over-Union in ``[0.0, 1.0]``.

    """
    return iou_xywh(xyxy2xywh(bb_xyxy1), xyxy2xywh(bb_xyxy2))


# ---------------------------------------------------------------------------
# Format conversions  (type-preserving: numpy → numpy, tensor → tensor)
# ---------------------------------------------------------------------------


def xyxy2xywh(bb_xyxy: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert bounding box coordinates from xyxy to xywh format.

    The output type mirrors the input type (numpy or torch tensor).

    Args:
        bb_xyxy: Bounding box ``[x_min, y_min, x_max, y_max]``.
            Shape ``(4,)`` for a single box or ``(N, 4)`` for a batch.

    Returns:
        Bounding box ``[x_min, y_min, width, height]`` with the same
        shape and type as the input.

    """
    if isinstance(bb_xyxy, torch.Tensor):
        squeezed = bb_xyxy.ndim == 1
        bb = bb_xyxy.unsqueeze(0) if squeezed else bb_xyxy
        xmin, ymin, xmax, ymax = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        out = torch.stack([xmin, ymin, xmax - xmin, ymax - ymin], dim=-1).round().long()
        return out.squeeze(0) if squeezed else out
    else:
        bb = np.asarray(bb_xyxy)
        squeezed = bb.ndim == 1
        bb = bb[None] if squeezed else bb
        xmin, ymin, xmax, ymax = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        out = np.round(np.column_stack((xmin, ymin, xmax - xmin, ymax - ymin))).astype(int)
        return out.squeeze() if squeezed else out


def xywh2xyxy(bb_xywh: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert bounding box coordinates from xywh to xyxy format.

    The output type mirrors the input type (numpy or torch tensor).

    Args:
        bb_xywh: Bounding box ``[x_min, y_min, width, height]``.
            Shape ``(4,)`` for a single box or ``(N, 4)`` for a batch.

    Returns:
        Bounding box ``[x_min, y_min, x_max, y_max]`` with the same
        shape and type as the input.

    """
    if isinstance(bb_xywh, torch.Tensor):
        squeezed = bb_xywh.ndim == 1
        bb = bb_xywh.unsqueeze(0) if squeezed else bb_xywh
        xmin, ymin, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        out = torch.stack([xmin, ymin, xmin + w, ymin + h], dim=-1).round().long()
        return out.squeeze(0) if squeezed else out
    else:
        bb = np.asarray(bb_xywh)
        squeezed = bb.ndim == 1
        bb = bb[None] if squeezed else bb
        xmin, ymin, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        out = np.round(np.column_stack((xmin, ymin, xmin + w, ymin + h))).astype(int)
        return out.squeeze() if squeezed else out


def midwh2xywh(bb_midwh: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert bounding box coordinates from midwh to xywh format.

    The output type mirrors the input type (numpy or torch tensor).

    Args:
        bb_midwh: Bounding box ``[x_center, y_center, width, height]``.
            Shape ``(4,)`` for a single box or ``(N, 4)`` for a batch.

    Returns:
        Bounding box ``[x_min, y_min, width, height]`` with the same
        shape and type as the input.

    """
    if isinstance(bb_midwh, torch.Tensor):
        squeezed = bb_midwh.ndim == 1
        bb = bb_midwh.unsqueeze(0) if squeezed else bb_midwh
        xc, yc, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        out = torch.stack([xc - 0.5 * w, yc - 0.5 * h, w, h], dim=-1).round().long()
        return out.squeeze(0) if squeezed else out
    else:
        bb = np.asarray(bb_midwh)
        squeezed = bb.ndim == 1
        bb = bb[None] if squeezed else bb
        xc, yc, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        out = np.round(np.column_stack((xc - 0.5 * w, yc - 0.5 * h, w, h))).astype(int)
        return out.squeeze() if squeezed else out


def xywh2midwh(bb_xywh: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert bounding box coordinates from xywh to midwh format.

    The output type mirrors the input type (numpy or torch tensor).

    Args:
        bb_xywh: Bounding box ``[x_min, y_min, width, height]``.
            Shape ``(4,)`` for a single box or ``(N, 4)`` for a batch.

    Returns:
        Bounding box ``[x_center, y_center, width, height]`` with the same
        shape and type as the input.

    """
    if isinstance(bb_xywh, torch.Tensor):
        squeezed = bb_xywh.ndim == 1
        bb = bb_xywh.unsqueeze(0) if squeezed else bb_xywh
        xmin, ymin, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        out = torch.stack([xmin + 0.5 * w, ymin + 0.5 * h, w, h], dim=-1).round().long()
        return out.squeeze(0) if squeezed else out
    else:
        bb = np.asarray(bb_xywh)
        squeezed = bb.ndim == 1
        bb = bb[None] if squeezed else bb
        xmin, ymin, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        out = np.round(np.column_stack((xmin + 0.5 * w, ymin + 0.5 * h, w, h))).astype(int)
        return out.squeeze() if squeezed else out


def xyxy2full(bb_xyxy: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert a bounding box to all four corner coordinates.

    The output type mirrors the input type (numpy or torch tensor).

    Args:
        bb_xyxy: Bounding box ``[x_tl, y_tl, x_br, y_br]`` of shape ``(4,)``.

    Returns:
        Four corner points ``[[x_tl, y_tl], [x_tr, y_tr], [x_bl, y_bl], [x_br, y_br]]``
        of shape ``(4, 2)``.

    """
    if isinstance(bb_xyxy, torch.Tensor):
        x_tl, y_tl, x_br, y_br = bb_xyxy[0], bb_xyxy[1], bb_xyxy[2], bb_xyxy[3]
        return torch.stack([x_tl, y_tl, x_br, y_tl, x_tl, y_br, x_br, y_br]).reshape(4, 2)
    else:
        x_tl, y_tl, x_br, y_br = bb_xyxy
        return np.array([x_tl, y_tl, x_br, y_tl, x_tl, y_br, x_br, y_br]).reshape(4, 2)


# ---------------------------------------------------------------------------
# Visualization  (cv2-based, always returns BGR numpy)
# ---------------------------------------------------------------------------


def visualize_bb(
    image: np.ndarray | torch.Tensor,
    bb_xyxy: np.ndarray | torch.Tensor,
    probability: float,
    output_path: os.PathLike | None = None,
) -> np.ndarray | torch.Tensor:
    """Draw a bounding box and detection confidence score on an image.

    Accepts both numpy arrays and torch tensors; returns the same type as
    the input.  cv2 drawing is performed on an intermediate BGR numpy array.

    Args:
        image: Input image — ``np.ndarray (H, W, C)`` uint8 **BGR** or
            ``torch.Tensor (C, H, W)`` uint8 **RGB**.
        bb_xyxy: Bounding box ``[x_min, y_min, x_max, y_max]`` of shape ``(4,)``.
            Both numpy arrays and torch tensors are accepted.
        probability: Detection confidence score to display as a label.
        output_path: Path to save the annotated image.  ``None`` skips saving.

    Returns:
        Annotated image — same type as ``image``.  Tensor output is
        ``(C, H, W)`` uint8 **RGB**; numpy output is ``(H, W, C)`` uint8 **BGR**.

    Raises:
        ValueError: If ``bb_xyxy`` does not have shape ``(4,)``.

    """
    if isinstance(image, torch.Tensor):
        is_tensor = True
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        is_tensor = False
        img_bgr = image.copy()

    # Normalise bb to (4,) int
    if isinstance(bb_xyxy, torch.Tensor):
        bb_xyxy = bb_xyxy.cpu().numpy()
    bb_xyxy = np.asarray(bb_xyxy)
    if bb_xyxy.shape != (4,):
        raise ValueError(f"Expected bounding box with shape (4,) got instead {bb_xyxy.shape}.")

    bb = np.rint(bb_xyxy).astype(int)
    label = str(round(float(probability), 2))
    pt1 = (int(bb[0]), int(bb[1]))
    pt2 = (int(bb[2]), int(bb[3]))
    cv2.rectangle(img_bgr, pt1, pt2, (255, 0, 0), 2)
    cv2.putText(
        img_bgr,
        label,
        (pt1[0] - 5, pt1[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_bgr)

    if is_tensor:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb).permute(2, 0, 1)
    return img_bgr
