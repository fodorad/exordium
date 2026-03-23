"""OpenGraphAU action unit detection wrapper."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_weight, load_checkpoint
from exordium.video.deep.base import _IMAGENET_MEAN, _IMAGENET_STD, VisualModelWrapper
from exordium.video.deep.swint import swin_transformer_tiny

AU_REGISTRY: list[tuple[str, str]] = [
    ("1", "Inner brow raiser"),
    ("2", "Outer brow raiser"),
    ("4", "Brow lowerer"),
    ("5", "Upper lid raiser"),
    ("6", "Cheek raiser"),
    ("7", "Lid tightener"),
    ("9", "Nose wrinkler"),
    ("10", "Upper lip raiser"),
    ("11", "Nasolabial deepener"),
    ("12", "Lip corner puller"),
    ("13", "Sharp lip puller"),
    ("14", "Dimpler"),
    ("15", "Lip corner depressor"),
    ("16", "Lower lip depressor"),
    ("17", "Chin raiser"),
    ("18", "Lip pucker"),
    ("19", "Tongue show"),
    ("20", "Lip stretcher"),
    ("22", "Lip funneler"),
    ("23", "Lip tightener"),
    ("24", "Lip pressor"),
    ("25", "Lips part"),
    ("26", "Jaw drop"),
    ("27", "Mouth stretch"),
    ("32", "Lip bite"),
    ("38", "Nostril dilator"),
    ("39", "Nostril compressor"),
    ("L1", "Left Inner brow raiser"),
    ("R1", "Right Inner brow raiser"),
    ("L2", "Left Outer brow raiser"),
    ("R2", "Right Outer brow raiser"),
    ("L4", "Left Brow lowerer"),
    ("R4", "Right Brow lowerer"),
    ("L6", "Left Cheek raiser"),
    ("R6", "Right Cheek raiser"),
    ("L10", "Left Upper lip raiser"),
    ("R10", "Right Upper lip raiser"),
    ("L12", "Left Nasolabial deepener"),
    ("R12", "Right Nasolabial deepener"),
    ("L14", "Left Dimpler"),
    ("R14", "Right Dimpler"),
]
"""Registry of all 41 Action Units — list of ``(au_id, display_name)`` tuples."""

AU_ids: list[str] = [au_id for au_id, _ in AU_REGISTRY]
"""Ordered list of AU code strings (``'1'``, ``'2'``, ... ``'L1'``, ``'R1'``, ...)."""
AU_names: list[str] = [name for _, name in AU_REGISTRY]
"""Ordered list of AU display names aligned with :data:`AU_ids`."""

_OPENGRAPHAU_WEIGHTS: dict[int, str] = {
    1: "opengraphau-swint-1s_weights.pth",
    2: "opengraphau-swint-2s_weights.pth",
}


class OpenGraphAuWrapper(VisualModelWrapper):
    """OpenGraphAU facial action unit detection wrapper.

    Detects 41 facial action units (27 main + 14 sub) using a graph neural
    network on top of a Swin Transformer Tiny backbone.

    Two pre-trained weight variants are available, both trained on the hybrid
    BP4D + DISFA dataset:

    * ``stage=1`` — ANFL: Adaptive Node Feature Learning with a dynamic
      sparse graph (``opengraphau-swint-1s_weights.pth``)
    * ``stage=2`` — MEFL: Multi-dimensional Edge Feature Learning with a
      gated GCN over full n×n edge features, higher accuracy
      (``opengraphau-swint-2s_weights.pth``)

    Weights are downloaded automatically from
    ``fodorad/exordium-weights`` on Hugging Face Hub on first use.

    Args:
        stage: Training stage variant, ``1`` or ``2``.  Defaults to ``2``.
        device_id: GPU device index.  ``None`` or negative uses CPU.

    Raises:
        ValueError: If ``stage`` is not ``1`` or ``2``.

    """

    def __init__(
        self,
        stage: int = 2,
        device_id: int | None = None,
    ):
        super().__init__(device_id)
        if stage not in _OPENGRAPHAU_WEIGHTS:
            raise ValueError(f"stage must be 1 or 2, got {stage!r}")
        self.stage = stage
        self.local_path = download_weight(_OPENGRAPHAU_WEIGHTS[stage], WEIGHT_DIR / "opengraphau")
        model = _MEFARG(num_main_classes=27, num_sub_classes=14, stage=stage)
        model.load_state_dict(load_checkpoint(self.local_path), strict=False)
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frames) -> torch.Tensor:
        """Resize, centre-crop and normalise to ImageNet stats.

        Args:
            frames: Any input supported by
                :meth:`~exordium.video.deep.base.VisualModelWrapper._to_uint8_tensor`.

        Returns:
            Float tensor of shape ``(B, 3, 224, 224)`` on ``self.device``.

        """
        x = self._to_uint8_tensor(frames).to(self.device)
        x = TF.resize(x, [256], antialias=True)
        x = TF.center_crop(x, [224])
        x = x.float().div(255)
        return (x - self._mean) / self._std

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """OpenGraphAU forward pass.

        Args:
            tensor: Preprocessed face tensor of shape ``(B, 3, 224, 224)``
                on ``self.device``.

        Returns:
            Action unit intensity tensor of shape ``(B, 41)`` with values
            roughly in ``[0, 1]``.

        """
        return self.model(tensor)

    def predict_au(self, frames) -> torch.Tensor:
        """Preprocess and run inference, returning AU intensities as a tensor.

        Args:
            frames: Any input supported by :meth:`preprocess`.

        Returns:
            AU intensity tensor of shape ``(B, 41)``.

        """
        return self(frames).cpu()


######################################################################################
#                                                                                    #
#   Code: https://github.com/lingjivoo/OpenGraphAU                                   #
#   Authors: Cheng Luo, Siyang Song, Weicheng Xie, Linlin Shen, Hatice Gunes         #
#   Reference: "Learning Multi-dimensional Edge Feature-based AU Relation             #
#               Graph for Facial Action Unit Recognition", IJCAI-ECAI 2022            #
#                                                                                    #
######################################################################################


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _LinearBlock(nn.Module):  # pragma: no cover
    """Linear → BN → ReLU block operating on the last dimension."""

    def __init__(self, in_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2.0 / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


def _bn_init(bn):  # pragma: no cover
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()


class _CrossAttn(nn.Module):  # pragma: no cover
    """Scaled dot-product cross-attention."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.linear_q.weight.data.normal_(0, math.sqrt(2.0 / (in_channels // 2)))
        self.linear_k.weight.data.normal_(0, math.sqrt(2.0 / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2.0 / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        attn = self.attend(torch.matmul(query, key.transpose(-2, -1)) * self.scale)
        return torch.matmul(attn, value)


class _GEM(nn.Module):  # pragma: no cover
    """Graph Edge Modeling — extracts pairwise edge features via two CrossAttn passes."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = _CrossAttn(in_channels)
        self.ARM = _CrossAttn(in_channels)
        self.edge_proj = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(num_classes * num_classes)
        self.edge_proj.weight.data.normal_(0, math.sqrt(2.0 / in_channels))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, class_feature, global_feature):
        B, N, D, C = class_feature.shape
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D, C)
        feat = self.FAM(class_feature, global_feature)
        feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C)
        feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
        feat = self.ARM(feat_start, feat_end)
        return self.bn(self.edge_proj(feat))


# ---------------------------------------------------------------------------
# Stage 1 (ANFL) — Adaptive Node Feature Learning
# ---------------------------------------------------------------------------


def _normalize_digraph(A):  # pragma: no cover
    """Symmetric normalisation D^{-1/2} A D^{-1/2} for a batched adjacency matrix."""
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim=-1)
    degs_inv_sqrt = node_degrees**-0.5
    norm_degs_matrix = torch.eye(n, device=A.device)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    return torch.bmm(torch.bmm(norm_degs_matrix, A), norm_degs_matrix)


class _GNN_S1(nn.Module):  # pragma: no cover
    """Dynamic sparse GNN for stage 1 (ANFL).

    Builds a top-K adjacency from node similarity (dots / cosine / l1 / l2),
    normalises it, and aggregates:  X' = ReLU(X + BN(V(X) + A U(X)))
    """

    def __init__(self, in_channels, num_classes, neighbor_num=4, metric="dots"):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.neighbor_num = neighbor_num
        self.metric = metric
        self.relu = nn.ReLU()
        self.U = nn.Linear(in_channels, in_channels)
        self.V = nn.Linear(in_channels, in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)
        self.U.weight.data.normal_(0, math.sqrt(2.0 / in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2.0 / in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape
        if self.metric == "dots":
            si = x.detach()
            si = torch.einsum("b i j, b j k -> b i k", si, si.transpose(1, 2))
            thr = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= thr).float()
        elif self.metric == "cosine":
            si = F.normalize(x.detach(), p=2, dim=-1)
            si = torch.einsum("b i j, b j k -> b i k", si, si.transpose(1, 2))
            thr = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= thr).float()
        elif self.metric == "l1":
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.sqrt(torch.pow(si.transpose(1, 2) - si, 2).sum(dim=-1))
            thr = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= thr).float()
        elif self.metric == "l2":
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si).sum(dim=-1)
            thr = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= thr).float()
        else:
            raise ValueError(f"Unknown metric: {self.metric!r}")
        A = _normalize_digraph(adj)
        aggregate = torch.einsum("b i j, b j k -> b i k", A, self.V(x))
        return self.relu(x + self.bnv(aggregate + self.U(x)))


class _Head_S1(nn.Module):  # pragma: no cover
    """Stage-1 AU classification head (ANFL — node features only)."""

    def __init__(
        self, in_channels, num_main_classes=27, num_sub_classes=14, neighbor_num=4, metric="dots"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes
        self.sub_list = [0, 1, 2, 4, 7, 8, 11]
        self.relu = nn.ReLU()

        self.main_class_linears = nn.ModuleList(
            [_LinearBlock(in_channels, in_channels) for _ in range(num_main_classes)]
        )
        self.gnn = _GNN_S1(in_channels, num_main_classes, neighbor_num=neighbor_num, metric=metric)
        self.main_sc = nn.Parameter(torch.zeros(num_main_classes, in_channels))
        self.sub_sc = nn.Parameter(torch.zeros(num_sub_classes, in_channels))
        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)

    def forward(self, x):
        # AFG: per-AU linear projections
        f_u = torch.cat([layer(x).unsqueeze(1) for layer in self.main_class_linears], dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG: dynamic graph aggregation
        f_v = self.gnn(f_v)
        return self._classify(f_v)

    def _classify(self, f_v):
        _, n, c = f_v.shape
        main_sc = F.normalize(self.relu(self.main_sc), p=2, dim=-1)
        main_cl = (F.normalize(f_v, p=2, dim=-1) * main_sc.view(1, n, c)).sum(dim=-1)
        sub_cl = []
        for i, idx in enumerate(self.sub_list):
            au = F.normalize(f_v[:, idx], p=2, dim=-1)
            sc_l = F.normalize(self.relu(self.sub_sc[2 * i]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[2 * i + 1]), p=2, dim=-1)
            sub_cl += [
                (au * sc_l.view(1, c)).sum(dim=-1)[:, None],
                (au * sc_r.view(1, c)).sum(dim=-1)[:, None],
            ]
        return torch.cat([main_cl, torch.cat(sub_cl, dim=-1)], dim=-1)


# ---------------------------------------------------------------------------
# Stage 2 (MEFL) — Multi-dimensional Edge Feature Learning
# ---------------------------------------------------------------------------


def _create_e_matrix(n):  # pragma: no cover
    """Build start (n²×n) and end (n²×n) incidence matrices for the gated GCN."""
    end = torch.zeros(n * n, n)
    for i in range(n):
        end[i * n : (i + 1) * n, i] = 1
    start = torch.eye(n).repeat(n, 1)
    return start, end


class _GNNLayer(nn.Module):  # pragma: no cover
    """Single Gated GCN layer — jointly updates node and edge features."""

    def __init__(self, in_channels, num_classes, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.U = nn.Linear(in_channels, in_channels, bias=False)
        self.V = nn.Linear(in_channels, in_channels, bias=False)
        self.A = nn.Linear(in_channels, in_channels, bias=False)
        self.B = nn.Linear(in_channels, in_channels, bias=False)
        self.E = nn.Linear(in_channels, in_channels, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.act = nn.ReLU()
        self.bnv = nn.BatchNorm1d(num_classes)
        self.bne = nn.BatchNorm1d(num_classes * num_classes)
        scale = math.sqrt(2.0 / in_channels)
        for linear in (self.U, self.V, self.A, self.B, self.E):
            linear.weight.data.normal_(0, scale)
        _bn_init(self.bnv)
        _bn_init(self.bne)

    def forward(self, x, edge, start, end):
        # Edge update
        edge = edge + self.act(
            self.bne(
                torch.einsum("ev, bvc -> bec", end, self.A(x))
                + torch.einsum("ev, bvc -> bec", start, self.B(x))
                + self.E(edge)
            )
        )
        # Sigmoid gate + per-class softmax normalisation
        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = self.softmax(e.view(b, self.num_classes, self.num_classes, c)).view(b, -1, c)
        # Node update
        Ujx = torch.einsum("ev, bvc -> bec", start, self.V(x))
        x = self.U(x) + torch.einsum("ve, bec -> bvc", end.t(), e * Ujx) / self.num_classes
        x = x + self.act(self.bnv(x))  # residual added inside bnv path
        return x, edge


class _GNN_S2(nn.Module):  # pragma: no cover
    """Two-layer Gated GCN for stage 2 (MEFL).

    Incidence matrices ``start`` / ``end`` are registered as non-parameter
    buffers so they move to the correct device automatically with ``.to()``.
    """

    def __init__(self, in_channels, num_classes, layer_num=2):
        super().__init__()
        self.num_classes = num_classes
        start, end = _create_e_matrix(num_classes)
        self.register_buffer("start", start)
        self.register_buffer("end", end)
        self.graph_layers = nn.ModuleList(
            [_GNNLayer(in_channels, num_classes) for _ in range(layer_num)]
        )

    def forward(self, x, edge):
        for layer in self.graph_layers:
            x, edge = layer(x, edge, self.start, self.end)
        return x, edge


class _Head_S2(nn.Module):  # pragma: no cover
    """Stage-2 AU classification head (MEFL — node + edge features)."""

    def __init__(self, in_channels, num_main_classes=27, num_sub_classes=14):
        super().__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes
        self.sub_list = [0, 1, 2, 4, 7, 8, 11]
        self.relu = nn.ReLU()

        self.main_class_linears = nn.ModuleList(
            [_LinearBlock(in_channels, in_channels) for _ in range(num_main_classes)]
        )
        self.edge_extractor = _GEM(in_channels, num_main_classes)
        self.gnn = _GNN_S2(in_channels, num_main_classes, layer_num=2)
        self.main_sc = nn.Parameter(torch.zeros(num_main_classes, in_channels))
        self.sub_sc = nn.Parameter(torch.zeros(num_sub_classes, in_channels))
        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)

    def forward(self, x):
        # AFG: per-AU linear projections
        f_u = torch.cat([layer(x).unsqueeze(1) for layer in self.main_class_linears], dim=1)
        f_v = f_u.mean(dim=-2)
        # Edge extraction + Gated GCN
        f_e = self.edge_extractor(f_u, x).mean(dim=-2)
        f_v, _ = self.gnn(f_v, f_e)
        return self._classify(f_v)

    def _classify(self, f_v):
        _, n, c = f_v.shape
        main_sc = F.normalize(self.relu(self.main_sc), p=2, dim=-1)
        main_cl = (F.normalize(f_v, p=2, dim=-1) * main_sc.view(1, n, c)).sum(dim=-1)
        sub_cl = []
        for i, idx in enumerate(self.sub_list):
            au = F.normalize(f_v[:, idx], p=2, dim=-1)
            sc_l = F.normalize(self.relu(self.sub_sc[2 * i]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[2 * i + 1]), p=2, dim=-1)
            sub_cl += [
                (au * sc_l.view(1, c)).sum(dim=-1)[:, None],
                (au * sc_r.view(1, c)).sum(dim=-1)[:, None],
            ]
        return torch.cat([main_cl, torch.cat(sub_cl, dim=-1)], dim=-1)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class _MEFARG(nn.Module):  # pragma: no cover
    """MEFARG backbone + graph head, selectable between stage 1 and stage 2.

    Args:
        num_main_classes: Number of main AU classes. Default: 27.
        num_sub_classes: Number of sub AU classes. Default: 14.
        stage: ``1`` for ANFL (dynamic sparse GNN), ``2`` for MEFL
            (gated GCN with explicit edge features). Default: 2.
        neighbor_num: Top-K neighbours for the stage-1 dynamic graph.
            Ignored for stage 2. Default: 4.
        metric: Similarity metric for the stage-1 dynamic graph
            (``"dots"``, ``"cosine"``, ``"l1"``, ``"l2"``).
            Ignored for stage 2. Default: ``"dots"``.

    """

    def __init__(
        self,
        num_main_classes: int = 27,
        num_sub_classes: int = 14,
        stage: int = 2,
        neighbor_num: int = 4,
        metric: str = "dots",
    ):
        super().__init__()
        self.backbone = swin_transformer_tiny()
        self.in_channels = self.backbone.num_features  # 768 for SwinT-Tiny
        self.out_channels = self.in_channels // 2  # 384
        self.backbone.head = None

        self.global_linear = _LinearBlock(self.in_channels, self.out_channels)

        if stage == 1:
            self.head = _Head_S1(
                self.out_channels,
                num_main_classes,
                num_sub_classes,
                neighbor_num=neighbor_num,
                metric=metric,
            )
        else:
            self.head = _Head_S2(self.out_channels, num_main_classes, num_sub_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_linear(x)
        return self.head(x)
