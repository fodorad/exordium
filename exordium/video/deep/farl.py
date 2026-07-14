"""FaRL universal facial representation wrapper.

Wraps the FaRL (Facial Representation Learning) vision encoder for frame-wise
feature extraction.  FaRL is a CLIP ViT-B/16 further pre-trained on LAION-Face
(20M face image-text pairs) with combined image-text contrastive learning and
masked image modelling, yielding a *general-purpose* face representation that
transfers well to attribute prediction, parsing, alignment and expression tasks.

Unlike :class:`~exordium.video.deep.emotieffnet.EmotiEffNetWrapper` — which is
fine-tuned for the 8 AffectNet expression classes and therefore discards cues
irrelevant to that head — FaRL is trained without task supervision.  This makes
it the better complement for traits that are not pure expression (apparent
personality, sentiment), and the two are designed to be concatenated.

Weights ship as a raw OpenAI-CLIP ``state_dict``.  This wrapper converts it to
the HuggingFace ``CLIPVisionModelWithProjection`` layout on load, so no
``openai-clip`` dependency is needed: the fused ``in_proj_weight`` attention
matrices are split into separate q/k/v projections and the OpenAI module names
are remapped.  FaRL's masked-image-modelling heads (``lm_transformer``,
``lm_head``, ``mask_token``) exist only for pre-training and are discarded.

License:
    FaRL code and weights are released under the MIT License by Microsoft.

References:
    * https://github.com/FacePerceiver/FaRL
    * Zheng et al. (2022), *General Facial Representation Learning in a
      Visual-Linguistic Manner*, CVPR (Oral).  https://arxiv.org/abs/2112.03109
"""

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_weight
from exordium.video.deep.base import VisualModelWrapper

logger = logging.getLogger(__name__)
"""Module-level logger."""

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
"""CLIP RGB channel means. FaRL inherits OpenAI CLIP's normalisation."""
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
"""CLIP RGB channel standard deviations. FaRL inherits OpenAI CLIP's normalisation."""

_WEIGHT_URL = (
    "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/{name}.pth"
)
"""Original FaRL release URL, kept as the fallback behind the mirror.

GitHub rate-limits unauthenticated requests from shared CI address ranges, so the
``fodorad/exordium-weights`` mirror is tried first.
"""

_MODELS: dict[str, dict] = {
    "ep16": {
        "filename": "FaRL-Base-Patch16-LAIONFace20M-ep16",
        "epochs": 16,
    },
    "ep64": {
        "filename": "FaRL-Base-Patch16-LAIONFace20M-ep64",
        "epochs": 64,
    },
}
"""Supported FaRL variants.

Both are ViT-B/16 trained on LAION-Face 20M and differ only in pre-training
length.  ``ep16`` is the checkpoint used in the CVPR paper; ``ep64`` is the
longer-trained release the authors recommend for downstream transfer.
"""

_DEFAULT_MODEL = "ep64"
"""Default FaRL variant — the longer-trained checkpoint."""

_IMG_SIZE = 224
"""FaRL pre-training resolution (ViT-B/16 → 14x14 patches + CLS)."""

_FEATURE_DIM = 512
"""Dimension of the projected FaRL image embedding."""

_HIDDEN_SIZE = 768
"""ViT-B hidden width, before the CLIP projection."""

_NUM_LAYERS = 12
"""ViT-B transformer depth."""

_NUM_HEADS = 12
"""ViT-B attention head count."""


def _vision_config() -> CLIPVisionConfig:
    """Build the CLIP ViT-B/16 vision config that FaRL's weights expect.

    Returns:
        A :class:`~transformers.CLIPVisionConfig` matching FaRL's architecture.

    """
    return CLIPVisionConfig(
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_HIDDEN_SIZE * 4,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS,
        image_size=_IMG_SIZE,
        patch_size=16,
        projection_dim=_FEATURE_DIM,
    )


def _build_vision_tower() -> Any:
    """Instantiate the CLIP ViT-B/16 vision tower FaRL's weights expect.

    Returns ``Any`` for the same reason
    :func:`~exordium.utils.ckpt.build_hf_model` does: ``transformers`` wraps
    ``PreTrainedModel.to`` in a decorator whose signature static analysis
    misreads, so a precisely-typed handle makes an ordinary device move look like
    a type error.

    Returns:
        A randomly-initialised :class:`~transformers.CLIPVisionModelWithProjection`.

    """
    return CLIPVisionModelWithProjection(_vision_config())


def _convert_openai_to_hf(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a FaRL (OpenAI-CLIP) vision state_dict to the HuggingFace layout.

    Three things differ between the two conventions:

    * **Module names** — OpenAI uses ``visual.transformer.resblocks.N.*``;
      HuggingFace uses ``vision_model.encoder.layers.N.*``.
    * **Attention packing** — OpenAI stores one fused ``in_proj_weight`` of shape
      ``(3*D, D)``; HuggingFace expects separate ``q_proj`` / ``k_proj`` /
      ``v_proj``.  The fused matrix is split in q, k, v order.
    * **Projection orientation** — OpenAI's ``visual.proj`` is ``(D, P)`` and is
      applied as ``x @ proj``; HuggingFace's ``visual_projection`` is an
      ``nn.Linear`` holding ``(P, D)``, so it must be transposed.

    Keys under ``visual.lm_*`` / ``visual.ln_lm`` / ``visual.mask_token`` belong
    to FaRL's masked-image-modelling pre-training head and are dropped — they
    have no counterpart in the encoder used for feature extraction.

    Args:
        state_dict: The ``"state_dict"`` entry of a FaRL ``.pth`` checkpoint.

    Returns:
        A state_dict loadable into
        :class:`~transformers.CLIPVisionModelWithProjection` with no missing or
        unexpected keys.

    """
    out: dict[str, torch.Tensor] = {
        "vision_model.embeddings.class_embedding": state_dict["visual.class_embedding"],
        "vision_model.embeddings.patch_embedding.weight": state_dict["visual.conv1.weight"],
        "vision_model.embeddings.position_embedding.weight": state_dict[
            "visual.positional_embedding"
        ],
        "vision_model.pre_layrnorm.weight": state_dict["visual.ln_pre.weight"],
        "vision_model.pre_layrnorm.bias": state_dict["visual.ln_pre.bias"],
        "vision_model.post_layernorm.weight": state_dict["visual.ln_post.weight"],
        "vision_model.post_layernorm.bias": state_dict["visual.ln_post.bias"],
        # (D, P) applied as x @ proj  ->  nn.Linear weight of shape (P, D).
        "visual_projection.weight": state_dict["visual.proj"].t().contiguous(),
    }

    for i in range(_NUM_LAYERS):
        src = f"visual.transformer.resblocks.{i}."
        dst = f"vision_model.encoder.layers.{i}."

        # Split the fused QKV projection: rows are ordered [q; k; v].
        weight = state_dict[f"{src}attn.in_proj_weight"]
        bias = state_dict[f"{src}attn.in_proj_bias"]
        dim = weight.shape[0] // 3
        for idx, name in enumerate(("q", "k", "v")):
            lo, hi = idx * dim, (idx + 1) * dim
            out[f"{dst}self_attn.{name}_proj.weight"] = weight[lo:hi]
            out[f"{dst}self_attn.{name}_proj.bias"] = bias[lo:hi]

        out[f"{dst}self_attn.out_proj.weight"] = state_dict[f"{src}attn.out_proj.weight"]
        out[f"{dst}self_attn.out_proj.bias"] = state_dict[f"{src}attn.out_proj.bias"]
        out[f"{dst}layer_norm1.weight"] = state_dict[f"{src}ln_1.weight"]
        out[f"{dst}layer_norm1.bias"] = state_dict[f"{src}ln_1.bias"]
        out[f"{dst}layer_norm2.weight"] = state_dict[f"{src}ln_2.weight"]
        out[f"{dst}layer_norm2.bias"] = state_dict[f"{src}ln_2.bias"]
        out[f"{dst}mlp.fc1.weight"] = state_dict[f"{src}mlp.c_fc.weight"]
        out[f"{dst}mlp.fc1.bias"] = state_dict[f"{src}mlp.c_fc.bias"]
        out[f"{dst}mlp.fc2.weight"] = state_dict[f"{src}mlp.c_proj.weight"]
        out[f"{dst}mlp.fc2.bias"] = state_dict[f"{src}mlp.c_proj.bias"]

    return out


class FarlWrapper(VisualModelWrapper):
    """FaRL universal face representation extractor.

    Extracts L2-normalised 512-d embeddings from face crops using the FaRL
    ViT-B/16 vision encoder.  Because FaRL is trained with language supervision
    on face images rather than fine-tuned on an expression label set, its
    features stay useful for tasks an expression classifier would throw away —
    apparent personality, sentiment, and attributes.

    Model weights are downloaded automatically on first use from the official
    FaRL GitHub release (MIT licensed) and cached under ``WEIGHT_DIR``.

    Supported variants:

    +------------------+--------+----------+
    | ``model_name``   | Epochs | Feat dim |
    +==================+========+==========+
    | ``ep16``         | 16     | 512      |
    +------------------+--------+----------+
    | ``ep64``         | 64     | 512      |
    +------------------+--------+----------+

    Args:
        model_name: FaRL variant — ``"ep16"`` (paper checkpoint) or ``"ep64"``
            (longer pre-training, better transfer).  Defaults to ``"ep64"``.
        device_id: GPU device index.  ``None`` or negative uses CPU.
        pretrained: If ``False``, build the architecture with random weights and
            download nothing.  Shapes and call paths are identical, but the
            features are meaningless — intended for tests and offline use.

    Raises:
        ValueError: If ``model_name`` is not one of the supported variants.

    Example::

        from exordium.video.deep.farl import FarlWrapper

        model = FarlWrapper(model_name="ep64", device_id=0)

        # single face crop -> (1, 512)
        features = model(face_crop)

        # face track -> one embedding per non-interpolated detection
        result = model.track_to_feature(track)
        # result["frame_ids"]: (N,)      result["features"]: (N, 512)

    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device_id: int | None = None,
        pretrained: bool = True,
    ) -> None:
        if model_name not in _MODELS:
            raise ValueError(f"Invalid model_name: {model_name!r}. Choose from {sorted(_MODELS)}.")
        super().__init__(device_id)

        self.feature_dim: int = _FEATURE_DIM
        self.img_size: int = _IMG_SIZE

        # Built from an explicit config rather than a Hub id: FaRL ships raw OpenAI
        # CLIP weights, so there is no HF repo to pull an architecture from.
        model = _build_vision_tower()
        if pretrained:
            filename = _MODELS[model_name]["filename"]
            local_path = download_weight(
                f"{filename}.pth",
                WEIGHT_DIR / "farl",
                upstream_url=_WEIGHT_URL.format(name=filename),
            )
            # The release checkpoint holds only tensors, so it loads under the safe
            # weights_only reader — no arbitrary unpickling.
            checkpoint = torch.load(str(local_path), map_location="cpu", weights_only=True)
            model.load_state_dict(_convert_openai_to_hf(checkpoint["state_dict"]))
        else:
            logger.info("Building FaRL architecture with random weights (no checkpoint).")

        model.to(self.device)
        model.eval()
        self.model: nn.Module = model

        self._mean = torch.tensor(_CLIP_MEAN, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor(_CLIP_STD, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

        logger.info(
            "FaRL (%s) loaded to %s  —  img_size=%d, feature_dim=%d",
            model_name,
            self.device,
            self.img_size,
            self.feature_dim,
        )

    def preprocess(self, frames: torch.Tensor | Sequence) -> torch.Tensor:
        """Resize and normalise face crops for FaRL.

        Applies bicubic resize to 224x224 and CLIP mean/std normalisation
        (FaRL inherits OpenAI CLIP's constants, *not* ImageNet's).

        Args:
            frames: Any input supported by
                :meth:`~exordium.video.deep.base.VisualModelWrapper._to_uint8_tensor`.

        Returns:
            Float tensor of shape ``(B, 3, 224, 224)`` on ``self.device``.

        """
        x = self._to_uint8_tensor(frames).to(self.device)
        x = TF.resize(
            x,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        x = x.float().div(255)
        return (x - self._mean) / self._std

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """FaRL vision encoder forward pass.

        Runs the ViT, projects the pooled output into the shared image-text
        embedding space, and L2-normalises the result.

        Args:
            tensor: Preprocessed face tensor of shape ``(B, 3, 224, 224)`` on
                ``self.device``.

        Returns:
            L2-normalised feature tensor of shape ``(B, 512)``.

        """
        embeds = self.model(pixel_values=tensor).image_embeds  # (B, 512)
        return embeds / embeds.norm(dim=-1, keepdim=True)
