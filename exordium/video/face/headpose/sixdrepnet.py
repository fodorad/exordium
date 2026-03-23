"""Head pose estimation via SixDRepNet (6D rotation representation).

Vendored inference components from the 6DRepNet repository are embedded
directly in this module to avoid the packaging bugs in the ``sixdrepnet``
PyPI package (bare script-style ``import utils`` that break when the
package is imported normally).  Only the pieces required for inference are
included: the RepVGG-B1g2 backbone, the SixDRepNet head, and the 6D-rotation
math utilities.

Weights for ``SixDRepNetWrapper`` are downloaded automatically on first use.

Example::

    wrapper = SixDRepNetWrapper(device_id=0)
    headpose = wrapper(face_tensor)   # (B, 3) tensor — [yaw, pitch, roll] degrees

Vendored source: https://github.com/thohemp/6DRepNet
Vendored author: Thorsten Hempel
Vendored license: MIT
Reference: Hempel et al., "6D Rotation Representation For Unconstrained Head Pose Estimation",
ICASSP 2022.
"""

from __future__ import annotations

import math
from math import cos, sin
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from exordium.video.deep.base import _IMAGENET_MEAN, _IMAGENET_STD, VisualModelWrapper


class SixDRepNetWrapper(VisualModelWrapper):
    """6D rotation representation head pose estimator.

    Calls the underlying ``nn.Module`` directly with a batched float tensor
    for efficient GPU/CPU inference, bypassing the slow single-image PIL loop
    of the original ``SixDRepNet_Detector.predict()``.

    Supported input types for :meth:`__call__`:

    * ``torch.Tensor`` — ``(C, H, W)`` or ``(B, C, H, W)`` uint8 RGB
    * ``np.ndarray`` — ``(H, W, 3)`` or ``(B, H, W, 3)`` uint8 RGB
    * ``Sequence[np.ndarray]`` — list of ``(H, W, 3)`` uint8 arrays
    * ``Sequence[str | Path]`` — list of image file paths

    Design contract:

    * :meth:`preprocess` — resize/normalise to ImageNet 224×224; stays on
      ``self.device``; returns ``(B, 3, 224, 224)`` float32.
    * :meth:`inference` — single batched forward pass through the RepVGG
      backbone + Euler conversion; returns ``(B, 3)`` ``[yaw, pitch, roll]``
      in degrees on ``self.device``.
    * :meth:`__call__` — chains both under ``torch.inference_mode``
      (inherited from :class:`~exordium.video.deep.base.VisualModelWrapper`).

    Args:
        device_id: GPU device index.  ``None`` or ``-1`` uses CPU.

    Note:
        Based on `6DRepNet <https://github.com/thohemp/6DRepNet>`_ —
        "6D Rotation Representation For Unconstrained Head Pose Estimation",
        Hempel et al., ICASSP 2022.

    """

    def __init__(self, device_id: int | None = None) -> None:
        super().__init__(device_id)
        self.model = _SixDRepNetModel.from_pretrained(self.device)
        self._mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

    def preprocess(self, frames) -> torch.Tensor:
        """Resize and normalise face crops to SixDRepNet input convention.

        SixDRepNet expects 224×224 inputs normalised to ImageNet mean/std.
        Face crops in a batch commonly have different spatial sizes (each
        detection is a different-sized region); this method resizes every
        crop individually before stacking so no prior alignment is needed.

        Args:
            frames: Any input accepted by
                :meth:`~exordium.video.deep.base.VisualModelWrapper._to_uint8_tensor`.
                When ``frames`` is a ``Sequence[np.ndarray]``, crops may have
                different ``(H, W)`` sizes.

        Returns:
            Float tensor of shape ``(B, 3, 224, 224)`` on ``self.device``.

        """
        if isinstance(frames, (list, tuple)) and not isinstance(frames[0], (str, Path)):
            resized = [
                TF.resize(
                    self._to_uint8_tensor(f).to(self.device),
                    [224, 224],
                    antialias=True,
                )
                for f in frames
            ]
            x = torch.cat(resized, dim=0)
        else:
            x = self._to_uint8_tensor(frames).to(self.device)
            x = TF.resize(x, [224, 224], antialias=True)

        x = x.float().div(255)
        return (x - self._mean) / self._std

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run a single batched forward pass and return Euler angles in degrees.

        Args:
            tensor: Float tensor of shape ``(B, 3, 224, 224)`` on
                ``self.device``, normalised to ImageNet mean/std.

        Returns:
            Float tensor of shape ``(B, 3)`` containing
            ``[yaw, pitch, roll]`` in degrees on ``self.device``.

        """
        rot_mat = self.model(tensor)  # (B, 3, 3)
        euler = _compute_euler_angles_from_rotation_matrices(rot_mat) * (180.0 / torch.pi)
        # sixdrepnet convention: euler cols = [pitch, yaw, roll]; reorder to [yaw, pitch, roll]
        return torch.stack([euler[:, 1], euler[:, 0], euler[:, 2]], dim=1)


def draw_headpose_axis(
    img: np.ndarray | torch.Tensor,
    headpose: tuple[float, float, float] | np.ndarray | torch.Tensor,
    tdx: int | None = None,
    tdy: int | None = None,
    size: int = 100,
    output_path: str | Path | None = None,
) -> np.ndarray | torch.Tensor:
    """Draw 3D head-pose axes (X/Y/Z) projected onto a 2D RGB image.

    Accepts ``(H, W, 3)`` uint8 RGB numpy arrays or ``(3, H, W)`` uint8
    RGB torch tensors; returns the same type.

    Args:
        img: Input RGB image — ``np.ndarray (H, W, 3)`` or
            ``torch.Tensor (3, H, W)`` uint8.
        headpose: Head pose angles ``(yaw, pitch, roll)`` in degrees.
            Accepts a tuple, numpy array, or 1-D / 1-row torch tensor.
        tdx: X-coordinate of the axis origin. ``None`` uses the image centre.
        tdy: Y-coordinate of the axis origin. ``None`` uses the image centre.
        size: Axis length in pixels. Defaults to 100.
        output_path: Path to save the annotated image. ``None`` skips saving.

    Returns:
        Copy of ``img`` (RGB) with head-pose axes drawn, same type as input
        (red=X/yaw, green=Y/pitch, blue=Z/roll).

    """
    is_tensor = isinstance(img, torch.Tensor)
    if isinstance(img, torch.Tensor):
        img_np: np.ndarray = img.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = cast("np.ndarray", img)

    if isinstance(headpose, torch.Tensor):
        yaw, pitch, roll = headpose.flatten().tolist()
    else:
        yaw, pitch, roll = float(headpose[0]), float(headpose[1]), float(headpose[2])

    pitch_r = math.radians(pitch)
    yaw_r = -math.radians(yaw)  # sixdrepnet draw_axis convention
    roll_r = math.radians(roll)

    if tdx is None or tdy is None:
        h, w = img_np.shape[:2]
        tdx, tdy = w // 2, h // 2

    x1 = size * (cos(yaw_r) * cos(roll_r)) + tdx
    y1 = size * (cos(pitch_r) * sin(roll_r) + cos(roll_r) * sin(pitch_r) * sin(yaw_r)) + tdy
    x2 = size * (-cos(yaw_r) * sin(roll_r)) + tdx
    y2 = size * (cos(pitch_r) * cos(roll_r) - sin(pitch_r) * sin(yaw_r) * sin(roll_r)) + tdy
    x3 = size * sin(yaw_r) + tdx
    y3 = size * (-cos(yaw_r) * sin(pitch_r)) + tdy

    out = img_np.copy()
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.line(out_bgr, (tdx, tdy), (int(x1), int(y1)), (0, 0, 255), 3)  # X — red
    cv2.line(out_bgr, (tdx, tdy), (int(x2), int(y2)), (0, 255, 0), 3)  # Y — green
    cv2.line(out_bgr, (tdx, tdy), (int(x3), int(y3)), (255, 0, 0), 2)  # Z — blue
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), out_bgr)

    if is_tensor:
        return torch.from_numpy(out_rgb).permute(2, 0, 1)
    return out_rgb


def draw_headpose_cube(
    img: np.ndarray | torch.Tensor,
    headpose: tuple[float, float, float] | np.ndarray | torch.Tensor,
    tdx: int | None = None,
    tdy: int | None = None,
    size: int = 100,
    output_path: str | Path | None = None,
) -> np.ndarray | torch.Tensor:
    """Draw a 3D head-pose cube projected onto a 2D RGB image.

    The cube visualises head orientation in the style of the original
    SixDRepNet repository: the **back face** (away from viewer) is drawn in
    red, the **pillars** connecting back to front in blue, and the **front
    face** (where the person looks) in green.

    Uses the same rotation convention as :func:`draw_headpose_axis`:
    ``yaw_r = -yaw * π/180``.

    Accepts ``(H, W, 3)`` uint8 RGB numpy arrays or ``(3, H, W)`` uint8
    RGB torch tensors; returns the same type.

    Args:
        img: Input RGB image — ``np.ndarray (H, W, 3)`` or
            ``torch.Tensor (3, H, W)`` uint8.
        headpose: Head pose angles ``(yaw, pitch, roll)`` in degrees.
            Accepts a tuple, numpy array, or 1-D / 1-row torch tensor.
        tdx: X-coordinate of the cube anchor (centre of the back face).
            ``None`` uses the image centre.
        tdy: Y-coordinate of the cube anchor. ``None`` uses image centre.
        size: Cube edge length in pixels. Defaults to 100.
        output_path: Path to save the annotated image. ``None`` skips saving.

    Returns:
        Copy of ``img`` (RGB) with the head-pose cube drawn, same type as input.

    """
    is_tensor = isinstance(img, torch.Tensor)
    if isinstance(img, torch.Tensor):
        img_np: np.ndarray = img.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = cast("np.ndarray", img)

    if isinstance(headpose, torch.Tensor):
        yaw, pitch, roll = headpose.flatten().tolist()
    else:
        yaw, pitch, roll = float(headpose[0]), float(headpose[1]), float(headpose[2])

    p = math.radians(pitch)
    y = -math.radians(yaw)  # sixdrepnet convention
    r = math.radians(roll)

    if tdx is None or tdy is None:
        h, w = img_np.shape[:2]
        tdx, tdy = w // 2, h // 2

    fx = tdx - 0.5 * size
    fy = tdy - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + fx
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + fy
    x2 = size * (-cos(y) * sin(r)) + fx
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + fy
    x3 = size * sin(y) + fx
    y3 = size * (-cos(y) * sin(p)) + fy

    def pt(px, py):
        return (int(px), int(py))

    bk = pt(fx, fy)
    b1 = pt(x1, y1)
    b2 = pt(x2, y2)
    b12 = pt(x2 + x1 - fx, y2 + y1 - fy)
    f0 = pt(x3, y3)
    f1 = pt(x3 + x1 - fx, y3 + y1 - fy)
    f2 = pt(x3 + x2 - fx, y3 + y2 - fy)
    f12 = pt(x3 + x1 + x2 - 2 * fx, y3 + y1 + y2 - 2 * fy)

    out = img_np.copy()
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.line(out, bk, b1, (0, 0, 255), 3)  # back face — red
    cv2.line(out, bk, b2, (0, 0, 255), 3)
    cv2.line(out, b1, b12, (0, 0, 255), 3)
    cv2.line(out, b2, b12, (0, 0, 255), 3)
    cv2.line(out, bk, f0, (255, 0, 0), 2)  # pillars — blue
    cv2.line(out, b1, f1, (255, 0, 0), 2)
    cv2.line(out, b2, f2, (255, 0, 0), 2)
    cv2.line(out, b12, f12, (255, 0, 0), 2)
    cv2.line(out, f0, f1, (0, 255, 0), 3)  # front face — green
    cv2.line(out, f0, f2, (0, 255, 0), 3)
    cv2.line(out, f1, f12, (0, 255, 0), 3)
    cv2.line(out, f2, f12, (0, 255, 0), 3)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), out)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    if is_tensor:
        return torch.from_numpy(out_rgb).permute(2, 0, 1)
    return out_rgb


# ===========================================================================
# Vendored: Squeeze-and-Excitation block
# Source: sixdrepnet/backbone/se_block.py
# ===========================================================================


class _SEBlock(nn.Module):  # pragma: no cover
    def __init__(self, input_channels: int, internal_neurons: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = F.relu(self.down(x))
        x = torch.sigmoid(self.up(x))
        return inputs * x.view(-1, self.input_channels, 1, 1)


# ===========================================================================
# Vendored: RepVGG backbone
# Source: sixdrepnet/backbone/repvgg.py
# ===========================================================================


def _conv_bn(  # pragma: no cover
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 1,
) -> nn.Sequential:
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(out_channels))
    return result


class _RepVGGBlock(nn.Module):  # pragma: no cover
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        deploy: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()
        self.se: nn.Module = _SEBlock(out_channels, out_channels // 16) if use_se else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            )
            self.rbr_dense = _conv_bn(
                in_channels, out_channels, kernel_size, stride, padding, groups
            )
            self.rbr_1x1 = _conv_bn(in_channels, out_channels, 1, stride, padding_11, groups)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def switch_to_deploy(self) -> None:
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.rbr_dense.conv.in_channels,
            self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for p in self.parameters():
            p.detach_()
        del self.rbr_dense, self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        if hasattr(self, "id_tensor"):
            del self.id_tensor
        self.deploy = True

    def _get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        k3, b3 = self._fuse_bn(self.rbr_dense)
        k1, b1 = self._fuse_bn(self.rbr_1x1)
        ki, bi = self._fuse_bn(self.rbr_identity)
        return k3 + self._pad_1x1(k1) + ki, b3 + b1 + bi  # ty: ignore[invalid-return-type]

    @staticmethod
    def _pad_1x1(k: torch.Tensor | int) -> torch.Tensor | int:
        if isinstance(k, int):
            return k
        return F.pad(k, [1, 1, 1, 1])

    def _fuse_bn(self, branch: nn.Module | None) -> tuple[torch.Tensor | int, torch.Tensor | int]:
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight  # type: ignore[union-attr]
            rm = branch.bn.running_mean  # type: ignore[union-attr]
            rv = branch.bn.running_var  # type: ignore[union-attr]
            gamma = branch.bn.weight  # type: ignore[union-attr]
            beta = branch.bn.bias  # type: ignore[union-attr]
            eps = branch.bn.eps  # type: ignore[union-attr]
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kv = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kv[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kv).to(branch.weight.device)
            kernel = self.id_tensor
            rm, rv, gamma, beta, eps = (
                branch.running_mean,
                branch.running_var,
                branch.weight,
                branch.bias,
                branch.eps,
            )
        std = (rv + eps).sqrt()  # ty: ignore[unsupported-operator]
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - rm * gamma / std  # ty: ignore[unsupported-operator]


class _RepVGG(nn.Module):  # pragma: no cover
    def __init__(
        self,
        num_blocks: list[int],
        num_classes: int = 1000,
        width_multiplier: list[float] | None = None,
        override_groups_map: dict[int, int] | None = None,
        deploy: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        assert width_multiplier is not None and len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or {}
        self.use_se = use_se
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.cur_layer_idx = 1

        self.stage0 = _RepVGGBlock(
            3, self.in_planes, 3, stride=2, padding=1, deploy=deploy, use_se=use_se
        )
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            g = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                _RepVGGBlock(
                    self.in_planes,
                    planes,
                    3,
                    stride=s,
                    padding=1,
                    groups=g,
                    deploy=self.deploy,
                    use_se=self.use_se,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return self.linear(x.view(x.size(0), -1))


_G2_MAP = {layer: 2 for layer in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]}
"""Mapping of RepVGG layer indices to group count 2 for weight reparameterisation."""

_WEIGHT_URL = "https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth"
"""Download URL for the 6DRepNet pretrained weights (300W-LP + AFLW2000)."""


# ===========================================================================
# Vendored: SixDRepNet nn.Module
# Source: sixdrepnet/model.py
# ===========================================================================


class _SixDRepNetModel(nn.Module):  # pragma: no cover
    """RepVGG-B1g2 backbone with a 6D-rotation linear head (inference only).

    Vendored from https://github.com/thohemp/6DRepNet (model.py).
    Weights are fetched via :meth:`from_pretrained`.
    """

    def __init__(self, deploy: bool = True) -> None:
        super().__init__()
        backbone = _RepVGG(
            num_blocks=[4, 6, 16, 1],
            num_classes=1000,
            width_multiplier=[2.0, 2.0, 2.0, 4.0],
            override_groups_map=_G2_MAP,
            deploy=deploy,
        )
        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ("rbr_dense" in n or "rbr_reparam" in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        self.linear_reg = nn.Linear(last_channel, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return _compute_rotation_matrix_from_ortho6d(self.linear_reg(x))

    @classmethod
    def from_pretrained(cls, device: torch.device) -> _SixDRepNetModel:
        model = cls(deploy=True)
        state = torch.hub.load_state_dict_from_url(_WEIGHT_URL, map_location=device)
        model.load_state_dict(state)
        model.eval()
        model.to(device)
        return model


# ===========================================================================
# Vendored: 6D-rotation math utilities
# Source: sixdrepnet/utils.py
# ===========================================================================


def _normalize_vector(v: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    v_mag = torch.clamp(v.pow(2).sum(1, keepdim=True).sqrt(), min=1e-8)
    return v / v_mag.expand_as(v)


def _cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return torch.stack([i, j, k], dim=1)


def _compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    x = _normalize_vector(poses[:, 0:3])
    z = _normalize_vector(_cross_product(x, poses[:, 3:6]))
    y = _cross_product(z, x)
    return torch.cat([x.unsqueeze(2), y.unsqueeze(2), z.unsqueeze(2)], dim=2)


def _compute_euler_angles_from_rotation_matrices(
    R: torch.Tensor,
) -> torch.Tensor:  # pragma: no cover
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = (sy < 1e-6).float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = torch.zeros_like(R[:, 1, 0])

    out = torch.zeros(R.shape[0], 3, device=R.device, dtype=R.dtype)
    out[:, 0] = x * (1 - singular) + xs * singular
    out[:, 1] = y * (1 - singular) + ys * singular
    out[:, 2] = z * (1 - singular) + zs * singular
    return out
