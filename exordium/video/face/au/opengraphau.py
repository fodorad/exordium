import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_file, load_checkpoint
from exordium.video.deep.base import VisualModelWrapper
from exordium.video.deep.resnet import resnet18, resnet50, resnet101
from exordium.video.deep.swin import (
    swin_transformer_base,
    swin_transformer_small,
    swin_transformer_tiny,
)

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

AU_ids: list[str] = [au_id for au_id, _ in AU_REGISTRY]
AU_names: list[str] = [name for _, name in AU_REGISTRY]


class OpenGraphAuWrapper(VisualModelWrapper):
    """OpenGraphAU facial action unit detection wrapper.

    Detects 41 facial action units (27 main + 14 sub) using a graph neural
    network on top of a Swin Transformer or ResNet backbone.  Model weights
    are downloaded automatically on first use.

    Args:
        backbone_name: Feature backbone name.  One of
            ``"swin_transformer_tiny"`` (default), ``"swin_transformer_small"``,
            ``"swin_transformer_base"``, ``"resnet18"``, ``"resnet50"``,
            ``"resnet101"``.
        device_id: GPU device index.  ``None`` or negative uses CPU.

    """

    def __init__(self, backbone_name: str = "swin_transformer_tiny", device_id: int | None = None):
        super().__init__(device_id)
        self.remote_path = "https://github.com/fodorad/exordium/releases/download/v1.0.0/opengraphau-swint-1s_weights.pth"
        self.local_path = WEIGHT_DIR / "opengraphau" / self.remote_path.split("/")[-1]
        download_file(self.remote_path, self.local_path)
        model = MEFARG(
            num_main_classes=27,
            num_sub_classes=14,
            backbone=backbone_name,
            neighbor_num=4,
            metric="dots",
        )
        model.load_state_dict(load_checkpoint(self.local_path), strict=False)
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_inference = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _preprocess(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert RGB numpy frames to a (B, 3, 224, 224) tensor on self.device.

        Args:
            frames: RGB uint8 arrays each of shape (H, W, 3).

        Returns:
            Tensor of shape (B, 3, 224, 224) on self.device.

        """
        return torch.stack([self.transform(Image.fromarray(np.uint8(f))) for f in frames]).to(
            self.device
        )

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """OpenGraphAU forward pass.

        Args:
            tensor: Preprocessed face tensor of shape (B, 3, 224, 224)
                on self.device.

        Returns:
            Action unit feature tensor of shape (B, 41).

        """
        if tensor.ndim != 4:  # pragma: no cover
            raise ValueError(f"Expected (B, C, H, W) tensor, got shape {tensor.shape}.")
        feature = self.model(tensor)
        if feature.shape[-1] != 41:  # pragma: no cover
            raise ValueError(f"Expected output shape (B, 41), got {feature.shape}.")
        return feature

    def inference_from_tensor(self, samples: torch.Tensor) -> torch.Tensor:
        """Run OpenGraphAU on a pre-loaded normalised tensor.

        Applies only the resize/crop normalization (skips PIL conversion).
        Useful when the caller already has a float tensor.

        Args:
            samples: Tensor of shape (B, C, H, W) of face crops,
                in RGB format and [0, 1] pixel value range.

        Returns:
            Action unit feature tensor of shape (B, 41).

        """
        with torch.inference_mode():
            samples = self.transform_inference(samples)
            return self.model(samples)

    def image_to_feature(self, image_path: str) -> np.ndarray:
        """Extract features from a single image file.

        Args:
            image_path: Path to the image file.

        Returns:
            Feature array of shape (41,).

        """
        from exordium.video.core.io import image_to_np

        return self.predict([image_to_np(image_path, "RGB")])[0]


######################################################################################
#                                                                                    #
#   Code: https://github.com/lingjivoo/OpenGraphAU                                   #
#   Authors: Cheng Luo, Siyang Song, Weicheng Xie, Linlin Shen, Hatice Gunes         #
#   Reference: "Learning Multi-dimensional Edge Feature-based AU Relation             #
#               Graph for Facial Action Unit Recognition", IJCAI-ECAI 2022            #
#                                                                                    #
######################################################################################


class LinearBlock(nn.Module):  # pragma: no cover
    """Linear block module with batch normalization and activation."""

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
        """Apply linear transformation with normalization and activation.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.

        """
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


def bn_init(bn):  # pragma: no cover
    """Initialize batch normalization layer.

    Args:
        bn: Batch normalization layer to initialize.

    """
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()


def normalize_digraph(A):  # pragma: no cover
    """Normalize directed graph adjacency matrix.

    Used in stage 1 (ANFL).

    Args:
        A: Adjacency matrix of shape (b, n, n).

    Returns:
        Normalized adjacency matrix.

    """
    # Used in stage 1 (ANFL)
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim=-1)
    degs_inv_sqrt = node_degrees**-0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix, A), norm_degs_matrix)
    return norm_A


def create_e_matrix(n):  # pragma: no cover
    """Create edge matrices for graph construction.

    Used in stage 2 (MEFL).

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (start, end) edge matrices.

    """
    # Used in stage 2 (MEFL)
    end = torch.zeros((n * n, n))
    for i in range(n):
        end[i * n : (i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n, 1)
    return start, end


class CrossAttn(nn.Module):  # pragma: no cover
    """cross attention Module."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2.0 / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2.0 / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2.0 / in_channels))

    def forward(self, y, x):
        """Apply cross-attention mechanism.

        Args:
            y: Query tensor.
            x: Key and value tensor.

        Returns:
            Attention output tensor.

        """
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out


class GEM(nn.Module):  # pragma: no cover
    """Graph edge modeling module."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(self.in_channels)
        self.ARM = CrossAttn(self.in_channels)
        self.edge_proj = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(self.num_classes * self.num_classes)

        self.edge_proj.weight.data.normal_(0, math.sqrt(2.0 / in_channels))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, class_feature, global_feature):
        """Compute edge features from class and global features.

        Args:
            class_feature: Class feature tensor.
            global_feature: Global feature tensor.

        Returns:
            Edge features tensor.

        """
        B, N, D, C = class_feature.shape
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D, C)
        feat = self.FAM(class_feature, global_feature)
        feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C)
        feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
        feat = self.ARM(feat_start, feat_end)
        edge = self.bn(self.edge_proj(feat))
        return edge


class GNN(nn.Module):  # pragma: no cover
    """Graph neural network module for node feature aggregation."""

    def __init__(self, in_channels, num_classes, neighbor_num=4, metric="dots"):
        super().__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors
        #   for each node feature.
        # metric: metric for assessing node similarity.
        #   Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels, self.in_channels)
        self.V = nn.Linear(self.in_channels, self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2.0 / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2.0 / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        """Aggregate node features using graph neural network.

        Args:
            x: Node feature tensor of shape (b, n, c).

        Returns:
            Aggregated node features.

        """
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == "dots":
            si = x.detach()
            si = torch.einsum("b i j , b j k -> b i k", si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(
                b, n, 1
            )
            adj = (si >= threshold).float()

        elif self.metric == "cosine":
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum("b i j , b j k -> b i k", si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(
                b, n, 1
            )
            adj = (si >= threshold).float()

        elif self.metric == "l1":
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.pow(si.transpose(1, 2) - si, 2)
            si = torch.sqrt(si.sum(dim=-1))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(
                b, n, 1
            )
            adj = (si <= threshold).float()

        elif self.metric == "l2":
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(
                b, n, 1
            )
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum("b i j, b j k->b i k", A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


class Head(nn.Module):  # pragma: no cover
    """Action unit classification head."""

    def __init__(
        self, in_channels, num_main_classes=27, num_sub_classes=14, neighbor_num=4, metric="dots"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes

        main_class_linear_layers = []

        for i in range(self.num_main_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            main_class_linear_layers += [layer]
        self.main_class_linears = nn.ModuleList(main_class_linear_layers)

        self.gnn = GNN(
            self.in_channels, self.num_main_classes, neighbor_num=neighbor_num, metric=metric
        )
        self.main_sc = nn.Parameter(
            torch.FloatTensor(torch.zeros(self.num_main_classes, self.in_channels))
        )

        self.sub_sc = nn.Parameter(
            torch.FloatTensor(torch.zeros(self.num_sub_classes, self.in_channels))
        )
        self.sub_list = [0, 1, 2, 4, 7, 8, 11]

        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)

    def forward(self, x):
        """Classify action units from feature tensor.

        Args:
            x: Feature tensor.

        Returns:
            Action unit classification logits.

        """
        # AFG
        f_u = []
        for i, layer in enumerate(self.main_class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape

        main_sc = self.main_sc
        main_sc = self.relu(main_sc)
        main_sc = F.normalize(main_sc, p=2, dim=-1)
        main_cl = F.normalize(f_v, p=2, dim=-1)
        main_cl = (main_cl * main_sc.view(1, n, c)).sum(dim=-1)

        sub_cl = []
        for i, index in enumerate(self.sub_list):
            au_l = 2 * i
            au_r = 2 * i + 1
            main_au = F.normalize(f_v[:, index], p=2, dim=-1)

            sc_l = F.normalize(self.relu(self.sub_sc[au_l]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[au_r]), p=2, dim=-1)

            cl_l = (main_au * sc_l.view(1, c)).sum(dim=-1)
            cl_r = (main_au * sc_r.view(1, c)).sum(dim=-1)
            sub_cl.append(cl_l[:, None])
            sub_cl.append(cl_r[:, None])
        sub_cl = torch.cat(sub_cl, dim=-1)
        cl = torch.cat([main_cl, sub_cl], dim=-1)
        return cl


class MEFARG(nn.Module):  # pragma: no cover
    """Multi-dimensional Edge Feature-based AU Relation Graph model."""

    def __init__(
        self,
        num_main_classes=27,
        num_sub_classes=14,
        backbone="swin_transformer_base",
        neighbor_num=4,
        metric="dots",
    ):
        super().__init__()
        if "transformer" in backbone:
            if backbone == "swin_transformer_tiny":
                self.backbone = swin_transformer_tiny()
            elif backbone == "swin_transformer_small":
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None
        elif "resnet" in backbone:
            if backbone == "resnet18":
                self.backbone = resnet18()
            elif backbone == "resnet101":
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_main_classes, num_sub_classes, neighbor_num, metric)

    def forward(self, x):
        """Forward pass through the MEFARG model.

        Args:
            x: Input tensor of shape (b, c, h, w).

        Returns:
            Action unit predictions.

        """
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl = self.head(x)
        return cl
