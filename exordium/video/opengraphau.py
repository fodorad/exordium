from pathlib import Path
from typing import Sequence
from PIL import Image
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from exordium import WEIGHT_DIR
from exordium.video.swin_transformers import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from exordium.video.resnet import resnet18, resnet101, resnet50
from exordium.utils.decorator import load_or_create
from exordium.utils.ckpt import download_file
from exordium.video.io import batch_iterator
from exordium.video.detection import Track


AU_names = [
    'Inner brow raiser',
    'Outer brow raiser',
    'Brow lowerer',
    'Upper lid raiser',
    'Cheek raiser',
    'Lid tightener',
    'Nose wrinkler',
    'Upper lip raiser',
    'Nasolabial deepener',
    'Lip corner puller',
    'Sharp lip puller',
    'Dimpler',
    'Lip corner depressor',
    'Lower lip depressor',
    'Chin raiser',
    'Lip pucker',
    'Tongue show',
    'Lip stretcher',
    'Lip funneler',
    'Lip tightener',
    'Lip pressor',
    'Lips part',
    'Jaw drop',
    'Mouth stretch',
    'Lip bite',
    'Nostril dilator',
    'Nostril compressor',
    'Left Inner brow raiser',
    'Right Inner brow raiser',
    'Left Outer brow raiser',
    'Right Outer brow raiser',
    'Left Brow lowerer',
    'Right Brow lowerer',
    'Left Cheek raiser',
    'Right Cheek raiser',
    'Left Upper lip raiser',
    'Right Upper lip raiser',
    'Left Nasolabial deepener',
    'Right Nasolabial deepener',
    'Left Dimpler',
    'Right Dimpler'
]


AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12',
          '13', '14', '15', '16', '17', '18', '19', '20', '22',
		  '23', '24', '25', '26', '27', '32', '38', '39',
          'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6',
          'L10', 'R10', 'L12', 'R12', 'L14', 'R14']


class image_eval(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model


class OpenGraphAuWrapper:

    def __init__(self, backbone_name: str = "swin_transformer_tiny", gpu_id: int = 0):
        self.device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu'
        self.remote_path = 'https://github.com/fodorad/exordium/releases/download/v1.0.0/opengraphau-swint-1s_weights.pth'
        self.local_path = WEIGHT_DIR / 'opengraphau' / Path(self.remote_path).name
        download_file(self.remote_path, self.local_path)
        model = MEFARG(num_main_classes=27, num_sub_classes=14, backbone=backbone_name, neighbor_num=4, metric="dots")
        self.model = load_state_dict(model, self.local_path)
        self.model.to(self.device)
        self.model.eval()
        self.transform = image_eval(img_size=256, crop_size=224)

    def __call__(self, faces_rgb: Sequence[np.ndarray]) -> np.ndarray:
        samples = torch.stack([self.transform(Image.fromarray(np.uint8(image))) for image in faces_rgb]).to(self.device) # (B, C, H, W) == (B, 3, 224, 224)

        if samples.ndim != 4:
            raise Exception(f'Invalid input shape. Expected sample shape is (B, C, H, W) got instead {samples.shape}.')

        with torch.no_grad():
            feature = self.model(samples)

        feature = feature.detach().cpu().numpy()

        if not feature.shape == (samples.shape[0], 41):
            raise Exception(f'Invalid output shape. Expected feature shape is {(samples.shape[0], 41)} got instead {feature.shape}.')

        return feature

    @load_or_create('pkl')
    def track_to_feature(self, track: Track, batch_size: int = 30, **kwargs) -> tuple[list, np.ndarray]:
        ids, features = [], []
        for subset in batch_iterator(track, batch_size):
            ids += [detection.frame_id for detection in subset if not detection.is_interpolated]
            samples = [detection.bb_crop_wide() for detection in subset if not detection.is_interpolated] # (B, H, W, C)
            feature = self(samples)
            features.append(feature)
        features = np.concatenate(features, axis=0)
        return ids, features


######################################################################################
#                                                                                    #
#   Code: https://github.com/lingjivoo/OpenGraphAU                                   #
#   Authors: Cheng Luo, Siyang Song, Weicheng Xie, Linlin Shen, Hatice Gunes         #
#   Reference: "Learning Multi-dimensional Edge Feature-based AU Relation			 #
# 				Graph for Facial Action Unit Recognition", IJCAI-ECAI 2022   		 #
#                                                                                    #
######################################################################################


class LinearBlock(nn.Module):
    def __init__(self, in_features,out_features=None,drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()


#Used in stage 1 (ANFL)
def normalize_digraph(A):
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim = -1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix,A),norm_degs_matrix)
    return norm_A


#Used in stage 2 (MEFL)
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end


class CrossAttn(nn.Module):
    """ cross attention Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out



class GEM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(self.in_channels)
        self.ARM = CrossAttn(self.in_channels)
        self.edge_proj = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(self.num_classes * self.num_classes)

        self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, class_feature, global_feature):
        B, N, D, C = class_feature.shape
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D, C)
        feat = self.FAM(class_feature, global_feature)
        feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C)
        feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
        feat = self.ARM(feat_start, feat_end)
        edge = self.bn(self.edge_proj(feat))
        return edge



# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNNLayer(nn.Module):

    def __init__(self, in_channels, num_classes, dropout_rate = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U = nn.Linear(dim_in, dim_out, bias=False)
        self.V = nn.Linear(dim_in, dim_out, bias=False)
        self.A = nn.Linear(dim_in, dim_out, bias=False)
        self.B = nn.Linear(dim_in, dim_out, bias=False)
        self.E = nn.Linear(dim_in, dim_out, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)

        self.bnv = nn.BatchNorm1d(num_classes)
        self.bne = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)


    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U.weight.data.normal_(0, scale)
        self.V.weight.data.normal_(0, scale)
        self.A.weight.data.normal_(0, scale)
        self.B.weight.data.normal_(0, scale)
        self.E.weight.data.normal_(0, scale)

        bn_init(self.bnv)
        bn_init(self.bne)


    def forward(self, x, edge, start, end):

        res = x
        Vix = self.A(x)  # V x d_out
        Vjx = self.B(x)  # V x d_out
        e = self.E(edge)  # E x d_out
        # print(e.shape)
        # print(x.shape)
        # print(start.shape)
        # print(end.shape)

        edge = edge + self.act(self.bne(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b,self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = res + self.act(self.bnv(x))

        return x, edge


# GAT GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, layer_num = 2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        graph_layers = []
        for i in range(layer_num):
            layer = GNNLayer(self.in_channels, self.num_classes)
            graph_layers += [layer]

        self.graph_layers = nn.ModuleList(graph_layers)


    def forward(self, x, edge):
        dev = x.get_device()
        if dev >= 0:
            self.start = self.start.to(dev)
            self.end = self.end.to(dev)
        for i, layer in enumerate(self.graph_layers):
            x, edge = layer(x, edge, self.start, self.end)
        return x, edge


class Head(nn.Module):
    def __init__(self, in_channels, num_main_classes = 27, num_sub_classes = 14):
        super().__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes

        main_class_linear_layers = []

        for i in range(self.num_main_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            main_class_linear_layers += [layer]
        self.main_class_linears = nn.ModuleList(main_class_linear_layers)

        self.edge_extractor = GEM(self.in_channels, num_main_classes)
        self.gnn = GNN(self.in_channels, num_main_classes, 2)


        self.main_sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_main_classes, self.in_channels)))
        self.sub_sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_sub_classes, self.in_channels)))
        self.sub_list = [0,1,2,4,7,8,11]

        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)


    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.main_class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)

        f_e = self.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)
        f_v, f_e = self.gnn(f_v, f_e)


        b, n, c = f_v.shape

        main_sc = self.main_sc
        main_sc = self.relu(main_sc)
        main_sc = F.normalize(main_sc, p=2, dim=-1)
        main_cl = F.normalize(f_v, p=2, dim=-1)
        main_cl = (main_cl * main_sc.view(1, n, c)).sum(dim=-1)

        sub_cl = []
        for i, index in enumerate(self.sub_list):
            au_l = 2*i
            au_r = 2*i + 1
            main_au = F.normalize(f_v[:, index], p=2, dim=-1)

            sc_l = F.normalize(self.relu(self.sub_sc[au_l]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[au_r]), p=2, dim=-1)

            cl_l = (main_au * sc_l.view(1, c)).sum(dim=-1)
            cl_r = (main_au * sc_r.view(1, c)).sum(dim=-1)
            sub_cl.append(cl_l[:,None])
            sub_cl.append(cl_r[:,None])
        sub_cl = torch.cat(sub_cl, dim=-1)
        cl = torch.cat([main_cl, sub_cl], dim=-1)
        return cl


class MEFARG(nn.Module):

    def __init__(self, num_main_classes = 27, num_sub_classes = 14, backbone='swin_transformer_base'):
        super().__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_main_classes, num_sub_classes)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl = self.head(x)
        return cl


class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.pow(si.transpose(1, 2) - si,2)
            si = torch.sqrt(si.sum(dim=-1))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        elif self.metric == 'l2':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_channels, num_main_classes = 27, num_sub_classes = 14, neighbor_num=4, metric='dots'):
        super().__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes

        main_class_linear_layers = []

        for i in range(self.num_main_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            main_class_linear_layers += [layer]
        self.main_class_linears = nn.ModuleList(main_class_linear_layers)

        self.gnn = GNN(self.in_channels, self.num_main_classes,neighbor_num=neighbor_num,metric=metric)
        self.main_sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_main_classes, self.in_channels)))

        self.sub_sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_sub_classes, self.in_channels)))
        self.sub_list = [0,1,2,4,7,8,11]

        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)

    def forward(self, x):
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
            au_l = 2*i
            au_r = 2*i + 1
            main_au = F.normalize(f_v[:, index], p=2, dim=-1)

            sc_l = F.normalize(self.relu(self.sub_sc[au_l]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[au_r]), p=2, dim=-1)

            cl_l = (main_au * sc_l.view(1, c)).sum(dim=-1)
            cl_r = (main_au * sc_r.view(1, c)).sum(dim=-1)
            sub_cl.append(cl_l[:,None])
            sub_cl.append(cl_r[:,None])
        sub_cl = torch.cat(sub_cl, dim=-1)
        cl = torch.cat([main_cl, sub_cl], dim=-1)
        return cl


class MEFARG(nn.Module):
    def __init__(self, num_main_classes = 27, num_sub_classes = 14, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super().__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
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
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl = self.head(x)
        return cl


if __name__ == '__main__':
    from exordium.video.io import image2np
    img_rgb = image2np('data/tmp/10025.jpg')
    m = OpenGraphAuWrapper(gpu_id=-1)
    au = m([img_rgb])
    print('output shape:', au.shape)
    print('output:', au)