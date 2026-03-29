"""emotion2vec+ speech emotion feature extractor wrapper.

Wraps the emotion2vec+ model (Data2VecMulti architecture) for frame-level
speech emotion feature extraction.  The model is pre-trained on speech emotion
data and produces 768-dimensional features at ~50 Hz.

Weights are downloaded on first use from the HuggingFace Hub and cached
locally.  The architecture is vendored as pure PyTorch — no fairseq or
funasr dependency is required.

References:
    * https://github.com/ddlBoJack/emotion2vec
    * Ma et al. (2024), *emotion2vec: Self-Supervised Pre-Training for
      Speech Emotion Representation*, ACL 2024.


Vendored model architecture
---------------------------

The building blocks below are a minimal, inference-only reimplementation of the
Data2VecMulti audio model.  Only the forward path required for feature
extraction is retained; all training-specific code (masking, EMA teacher, loss
computation) is removed.

The architecture is:

1. **ConvFeatureExtractor** — 7-layer 1-D CNN that converts raw 16 kHz
   waveforms to 512-d frame embeddings at ~50 Hz.
2. **ProjectionLayer** — LayerNorm + Linear projecting 512-d → 768-d.
3. **Relative positional encoder** — 5 grouped-conv layers with ALiBi-style
   learned scaling.
4. **Pre-net (context encoder)** — 4 transformer blocks.
5. **Main encoder** — 8 transformer blocks (AltBlock with ALiBi attention).

The checkpoint stores weights under a ``d2v_model.`` prefix which is stripped
during loading by :func:`_load_emotion2vec_state_dict`.

Original code license: MIT (Facebook / Alibaba DAMO Academy).

Note:
    Coverage is intentionally excluded for this section — it is
    third-party architecture code, tested only through the wrapper.
"""

import logging
import math
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exordium import WEIGHT_DIR
from exordium.audio.base import AudioModelWrapper
from exordium.utils.ckpt import download_file
from exordium.utils.decorator import load_or_create

logger = logging.getLogger(__name__)
"""Module-level logger."""

EMOTION2VEC_SAMPLE_RATE = 16000
"""Required audio sample rate for emotion2vec (16 000 Hz)."""

_WEIGHT_URL = "https://huggingface.co/emotion2vec/emotion2vec_plus_seed/resolve/main/model.pt"
"""URL for downloading the emotion2vec+ seed checkpoint."""

EMOTION2VEC_FEATURE_DIM = 768
"""Output feature dimension of the emotion2vec+ seed model."""


# ===================================================================== #
#  Vendored building blocks (pure PyTorch, no fairseq)                  #
# ===================================================================== #


class _TransposeLast(nn.Module):  # pragma: no cover
    """Transpose the last two dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-2, -1)


class _SamePad(nn.Module):  # pragma: no cover
    """Remove one trailing time-step when the kernel size is even."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove:
            x = x[..., :-1]
        return x


def _make_conv_block(  # pragma: no cover
    in_ch: int,
    out_ch: int,
    kernel: int,
    stride: int,
    bias: bool,
    layer_norm: bool,
) -> nn.Module:
    """Build one conv-feature-extractor block.

    The structure mirrors fairseq's ``ConvFeatureExtractionModel`` so that
    state_dict keys align with the checkpoint:
    ``conv_layers.{i}.0`` (Conv1d), ``conv_layers.{i}.2.1`` (LayerNorm).
    """
    conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=bias)
    if layer_norm:
        norm = nn.Sequential(_TransposeLast(), nn.LayerNorm(out_ch), _TransposeLast())
        return nn.Sequential(conv, nn.Identity(), norm, nn.GELU())
    return nn.Sequential(conv, nn.Identity(), nn.Identity(), nn.GELU())


class _ConvFeatureExtractor(nn.Module):  # pragma: no cover
    """Stack of 1-D convolutions that downsample raw audio to frame features.

    The default spec ``[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2``
    converts 16 kHz audio to ~50 Hz, 512-d representations.
    """

    def __init__(
        self,
        conv_spec: list[tuple[int, int, int]] | None = None,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if conv_spec is None:
            conv_spec = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        self.conv_spec = conv_spec

        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, s in conv_spec:
            layers.append(_make_conv_block(in_ch, out_ch, k, s, bias=False, layer_norm=layer_norm))
            in_ch = out_ch
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, T)
        for layer in self.conv_layers:
            x = layer(x)
        return x  # (B, C, T_frames)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.float()
        for _, k, s in self.conv_spec:
            lengths = torch.floor((lengths - k) / s + 1)
        return lengths.long()


def _build_relative_positional_encoder(  # pragma: no cover
    embed_dim: int = 768,
    kernel_size: int = 19,
    groups: int = 16,
    depth: int = 5,
) -> nn.Sequential:
    """Build convolutional relative positional encoding.

    Returns a plain ``nn.Sequential`` whose state_dict keys are flat
    integer indices (``0``, ``1.0``, …) matching the fairseq checkpoint.
    """
    blocks: list[nn.Module] = [_TransposeLast()]
    for _ in range(depth):
        blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    embed_dim, embed_dim, kernel_size, padding=kernel_size // 2, groups=groups
                ),
                _SamePad(kernel_size),
                _TransposeLast(),
                nn.LayerNorm(embed_dim, elementwise_affine=False),
                _TransposeLast(),
                nn.GELU(),
            )
        )
    blocks.append(_TransposeLast())
    return nn.Sequential(*blocks)


class _AltAttention(nn.Module):  # pragma: no cover
    """Multi-head attention with optional ALiBi bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        alibi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias

        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Mlp(nn.Module):  # pragma: no cover
    """Two-layer MLP with GELU (mirrors timm Mlp)."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class _DropPath(nn.Module):  # pragma: no cover
    """Stochastic depth (drop path) regularisation."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) > self.drop_prob
        return x / (1 - self.drop_prob) * keep


class _AltBlock(nn.Module):  # pragma: no cover
    """Transformer block used by emotion2vec (Data2VecMulti)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        post_mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.layer_norm_first = layer_norm_first
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _AltAttention(
            dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop=mlp_drop)
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        alibi_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x), padding_mask, alibi_bias))
            r = x = self.mlp(self.norm2(x))
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
        return x, t


class _BlockEncoder(nn.Module):  # pragma: no cover
    """Sequential transformer blocks with optional layer-drop and final norm."""

    def __init__(
        self,
        blocks: nn.ModuleList,
        norm: nn.Module | None,
        layer_norm_first: bool,
        layerdrop: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.norm = norm
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
        alibi_bias: torch.Tensor | None,
        alibi_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)

        x = self.dropout(x)

        for i, blk in enumerate(self.blocks):
            ab = alibi_bias
            if ab is not None and alibi_scale is not None:
                scale = alibi_scale[i] if alibi_scale.size(0) > 1 else alibi_scale.squeeze(0)
                ab = ab * scale.type_as(ab)
            x, _ = blk(x, padding_mask, ab)

        if self.norm is not None and self.layer_norm_first:
            x = self.norm(x)

        return x


def _get_alibi_bias(  # pragma: no cover
    batch_size: int,
    time_steps: int,
    heads: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Compute static ALiBi position bias.

    Returns:
        Tensor of shape ``(1, heads, time_steps, time_steps)``.

    """
    slopes = torch.tensor(_get_alibi_slopes(heads), dtype=dtype, device=device)
    pos = torch.arange(time_steps, dtype=dtype, device=device)
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T)
    bias = rel.unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)  # (H, T, T)
    return bias.unsqueeze(0).expand(batch_size, -1, -1, -1)


def _get_alibi_slopes(heads: int) -> list[float]:  # pragma: no cover
    """Compute ALiBi slopes for *heads* attention heads."""

    def _slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(heads).is_integer():
        return _slopes_power_of_2(heads)
    closest_pow2 = 2 ** math.floor(math.log2(heads))
    slopes = _slopes_power_of_2(closest_pow2)
    extra = _slopes_power_of_2(2 * closest_pow2)
    slopes.extend(extra[0::2][: heads - closest_pow2])
    return slopes


class _Emotion2vecModel(nn.Module):  # pragma: no cover
    """Pure-PyTorch emotion2vec model for inference-only feature extraction.

    This re-implements the ``Data2VecMultiModel`` + ``AudioEncoder`` forward
    path with no fairseq / funasr dependency.  Only the layers needed for
    ``extract_features`` are built; training-specific modules (EMA teacher,
    decoder, masking) are omitted.

    Default hyper-parameters match the **emotion2vec+ seed** checkpoint.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        prenet_depth: int = 4,
        num_extra_tokens: int = 10,
        conv_spec: list[tuple[int, int, int]] | None = None,
        pos_kernel: int = 19,
        pos_groups: int = 16,
        pos_depth: int = 5,
    ) -> None:
        super().__init__()
        if conv_spec is None:
            conv_spec = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        feature_dim = conv_spec[-1][0]

        self.embed_dim = embed_dim
        self.num_extra_tokens = num_extra_tokens
        self.conv_spec = conv_spec

        self.local_encoder = _ConvFeatureExtractor(conv_spec, layer_norm=True)

        self.project_features = nn.Sequential(
            _TransposeLast(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embed_dim),
        )

        self.relative_positional_encoder = _build_relative_positional_encoder(
            embed_dim,
            kernel_size=pos_kernel,
            groups=pos_groups,
            depth=pos_depth,
        )

        self.extra_tokens = nn.Parameter(torch.zeros(1, num_extra_tokens, embed_dim))
        self.alibi_scale = nn.Parameter(torch.ones(1, 1, num_heads, 1, 1))

        prenet_blocks = nn.ModuleList(
            [
                _AltBlock(embed_dim, num_heads, mlp_ratio, layer_norm_first=False)
                for _ in range(prenet_depth)
            ]
        )
        self.context_encoder = _BlockEncoder(
            prenet_blocks,
            norm=nn.LayerNorm(embed_dim),
            layer_norm_first=False,
            layerdrop=0.0,
            dropout=0.0,
        )

        self.blocks = nn.ModuleList(
            [
                _AltBlock(embed_dim, num_heads, mlp_ratio, layer_norm_first=False)
                for _ in range(depth)
            ]
        )

        self.norm: nn.Module | None = None

    def _compute_padding_mask(
        self,
        x_conv: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if padding_mask is None:
            return None
        input_lengths = (~padding_mask).long().sum(-1)
        output_lengths = self.local_encoder.output_lengths(input_lengths)
        B, T = x_conv.shape[:2]
        mask = torch.zeros(B, T, dtype=torch.bool, device=x_conv.device)
        for i, length in enumerate(output_lengths):
            mask[i, length:] = True
        if not mask.any():
            return None
        return mask

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list | None]:
        """Extract frame-level features from raw waveforms.

        Args:
            source: Raw 16 kHz waveform of shape ``(B, T_samples)``.
            padding_mask: Optional bool mask of shape ``(B, T_samples)`` where
                True indicates padding.

        Returns:
            Dict with key ``"x"`` containing the feature tensor of shape
            ``(B, T_frames, 768)``, and ``"padding_mask"`` (bool or None).

        """
        x_conv = self.local_encoder(source)
        x = self.project_features(x_conv)
        padding_mask = self._compute_padding_mask(x, padding_mask)

        x_pos = self.relative_positional_encoder(x)

        B, T, _ = x.shape
        alibi_bias = _get_alibi_bias(
            batch_size=B,
            time_steps=T + self.num_extra_tokens,
            heads=self.alibi_scale.shape[2],
            dtype=torch.float32,
            device=x.device,
        )
        alibi_scale = self.alibi_scale.clamp_min(0)

        x = x + x_pos

        if self.num_extra_tokens > 0:
            extra = self.extra_tokens.expand(B, -1, -1)
            x = torch.cat([extra, x], dim=1)
            if padding_mask is not None:
                padding_mask = F.pad(padding_mask, (self.num_extra_tokens, 0))

        prenet_depth = len(self.context_encoder.blocks)
        if alibi_scale.size(0) == 1:
            prenet_alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
            ctx_alibi_scale = None
        else:
            ctx_alibi_scale = alibi_scale[:prenet_depth]
            prenet_alibi_bias = alibi_bias

        x = self.context_encoder(x, padding_mask, prenet_alibi_bias, ctx_alibi_scale)

        if alibi_scale.size(0) > 1:
            main_alibi_scale = alibi_scale[prenet_depth:]
        else:
            main_alibi_scale = alibi_scale

        for i, blk in enumerate(self.blocks):
            ab = alibi_bias
            if ab is not None and main_alibi_scale is not None:
                if main_alibi_scale.size(0) > 1:
                    scale = main_alibi_scale[i]
                else:
                    scale = main_alibi_scale.squeeze(0)
                ab = ab * scale.type_as(ab)
            x, _ = blk(x, padding_mask=padding_mask, alibi_bias=ab)

        if self.norm is not None:
            x = self.norm(x)

        if self.num_extra_tokens > 0:
            x = x[:, self.num_extra_tokens :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, self.num_extra_tokens :]

        return {"x": x, "padding_mask": padding_mask}


def _load_emotion2vec_state_dict(path: str) -> dict[str, torch.Tensor]:  # pragma: no cover
    """Load an emotion2vec+ checkpoint and remap keys.

    The checkpoint stores the full training state; only the ``model`` dict is
    used.  Keys prefixed with ``d2v_model.`` are mapped to the corresponding
    attribute in :class:`_Emotion2vecModel`, and the
    ``modality_encoders.AUDIO.`` prefix is flattened out.

    The classification head (``proj.*``) is dropped.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_sd = ckpt["model"]

    new_sd: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if not k.startswith("d2v_model."):
            continue
        k = k[len("d2v_model.") :]
        k = k.replace("modality_encoders.AUDIO.", "")
        new_sd[k] = v

    return new_sd


# ===================================================================== #
#  Public wrapper                                                       #
# ===================================================================== #


class Emotion2vecWrapper(AudioModelWrapper):
    """Wrapper for emotion2vec+ seed frame-level feature extraction.

    Extracts 768-dimensional speech emotion features at ~50 Hz from 16 kHz
    mono audio.  The model uses a Data2VecMulti architecture (convolutional
    feature extractor + transformer encoder) pre-trained on curated academic
    speech emotion datasets.

    Model weights are downloaded automatically on first use.

    Args:
        device_id: GPU device index.  ``-1`` or ``None`` uses CPU.

    Example::

        from exordium.audio.emotion2vec import Emotion2vecWrapper

        model = Emotion2vecWrapper(device_id=0)

        # From file — returns (T, 768) numpy array
        features = model.audio_to_feature("speech.wav")

        # From tensor — returns (T, 768) torch.Tensor
        waveform = torch.randn(16000)   # 1 second at 16 kHz
        features = model(waveform)

    """

    def __init__(self, device_id: int = -1) -> None:
        super().__init__(device_id)

        # Download weights
        weight_dir = WEIGHT_DIR / "emotion2vec"
        local_path = weight_dir / "model.pt"
        download_file(_WEIGHT_URL, local_path)

        # Build and load model
        state_dict = _load_emotion2vec_state_dict(str(local_path))
        self.model = _Emotion2vecModel()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info("emotion2vec+ seed loaded to %s.", self.device)

    def __call__(
        self,
        waveform: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Extract emotion features from a waveform.

        Args:
            waveform: Mono audio signal of shape ``(T,)`` (single) or
                ``(B, T)`` (batch of **same-length** signals) at 16 kHz.

        Returns:
            Feature tensor of shape ``(T', 768)`` for 1-D input or
            ``(B, T', 768)`` for 2-D input, on ``self.device``.

        Raises:
            ValueError: If waveform is not 1-D or 2-D.

        """
        waveform = torch.as_tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim != 2:
            raise ValueError(f"Expected shape (T,) or (B, T), got {tuple(waveform.shape)}.")

        squeeze = waveform.shape[0] == 1 and waveform.ndim == 2

        waveform = waveform.to(self.device)
        with torch.inference_mode():
            out = self.model.extract_features(waveform)

        features = cast("torch.Tensor", out["x"])  # (B, T', 768)
        if squeeze:
            features = features.squeeze(0)  # (T', 768)
        return features

    @load_or_create("npy")
    def audio_to_feature(
        self,
        audio: Path | str | np.ndarray | torch.Tensor,
        **_kwargs,
    ) -> np.ndarray:
        """Extract emotion2vec features from an audio path or waveform.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
                If a path is given, audio is loaded and resampled to 16 kHz.
            **_kwargs: Passed to :func:`~exordium.utils.decorator.load_or_create`
                (``output_path``, ``overwrite``).

        Returns:
            Feature array of shape ``(T, 768)`` where T is time frames
            (~50 per second).

        """
        waveform = self._prepare_waveform(audio, EMOTION2VEC_SAMPLE_RATE)
        return self(waveform).detach().cpu().numpy()

    def batch_audio_to_features(
        self,
        audios: list[Path | str | np.ndarray | torch.Tensor],
        **_kwargs,
    ) -> list[np.ndarray]:
        """Extract emotion2vec features from multiple audio inputs.

        Variable-length inputs are zero-padded to the longest waveform.
        Output frames corresponding to padding are trimmed.

        Args:
            audios: List of audio file paths, numpy arrays, or torch tensors.

        Returns:
            List of feature arrays, each of shape ``(T_i, 768)``.

        """
        waveforms = [self._prepare_waveform(a, EMOTION2VEC_SAMPLE_RATE) for a in audios]
        padded, lengths = self._pad_waveforms(waveforms)

        # Build a padding mask (True = padding)
        padding_mask = torch.ones_like(padded, dtype=torch.bool)
        for i, length in enumerate(lengths):
            padding_mask[i, :length] = False

        padded = padded.to(self.device)
        padding_mask = padding_mask.to(self.device)

        with torch.inference_mode():
            out = self.model.extract_features(padded, padding_mask=padding_mask)

        features = cast("torch.Tensor", out["x"])  # (B, T_max_frames, 768)
        out_mask: torch.Tensor | None = cast("torch.Tensor | None", out["padding_mask"])

        results: list[np.ndarray] = []
        for i in range(len(waveforms)):
            if out_mask is not None:
                valid = (~out_mask[i]).sum().item()
                results.append(features[i, :valid].detach().cpu().numpy())
            else:
                results.append(features[i].detach().cpu().numpy())

        return results

    @torch.inference_mode()
    def inference(self, waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract features in inference mode, returning a tensor.

        Args:
            waveform: Mono audio signal of shape ``(T,)`` at 16 kHz.

        Returns:
            Feature tensor of shape ``(T', 768)`` on ``self.device``.

        """
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        out = self.model.extract_features(waveform)
        return cast("torch.Tensor", out["x"]).squeeze(0)
