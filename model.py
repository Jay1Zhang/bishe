#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import math
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models
from torch.nn.modules.container import ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils import *


# 0 - Input Module
# This model is used to generate primary input embedding of one modality
class InputModel(nn.Module):
    def __init__(self, mod, dropout=0.4):
        super(InputModel, self).__init__()

        self.u = 65     # 输入特征矩阵 (S, E)
        self.d = 32     # 规整后的矩阵 (u, d)
        if mod == 'a':
            self.seq_len = 220
            self.feat_dim = 73
            self.conv1 = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.d, kernel_size=28, stride=3)
        elif mod == 'v':
            self.seq_len = 350
            self.feat_dim = 512
            self.conv1 = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.d, kernel_size=30, stride=5)
        else:
            self.seq_len = 610
            self.feat_dim = 200
            self.conv1 = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.d, kernel_size=34, stride=9)

        # self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.d)
        self.relu = F.relu
        self.ln = nn.LayerNorm(self.d)

    def forward(self, src):
        # Input:    src(N, S, E)
        # Output:   out(N, u, d)
        x = src.permute(0, 2, 1)    # pytorch是对最后一维做卷积的, 因此需要把S换到最后
        x = self.conv1(x)           # (N, S, E) -> (N, d, u)
        # x = self.dropout(x)               # 1. dropout
        # x = self.relu(self.bn(x))         # 2. bn+relu
        # out = self.ln(x.permute(0, 2, 1)) # 3. ln
        out = x.permute(0, 2, 1)    # (N, u, d)
        return out


# Transformer Encoder
# positional encoder + transformer encoder
class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers=1, dropout=0.4):
        super(TransformerModel, self).__init__()
        # self.model_type = 'Transformer'
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # self.pos_encoder = LearnedPositionalEncoding(ninp, dropout)

        # 1-layer transformer encoder
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        # Input:    src(S, N, E:16)
        # Output:   out(T, N, E:16)
        src *= math.sqrt(self.ninp)
        # positional encoder
        src = self.pos_encoder(src)
        # transformer encoder
        output = self.transformer_encoder(src)
        return output


# positional encoding
# 无可学习参数的PositionEncoding层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# The model generate the compact latent embedding of one modality
class LatentModule(nn.Module):
    def __init__(self, mod, dropout=0.4):
        super(LatentModule, self).__init__()

        # question: it means?
        # answer:
        nfeat = 32
        nhead = 4   # number of head attention
        nhid = 16   # number of hidden layers
        nlayers = 1     # transformer encoder layers

        self.feat_exa = InputModel(mod, dropout)
        self.transformer_emb = TransformerModel(nfeat, nhead, nhid, nlayers, dropout)

    def forward(self, src):
        # Input:    src(N, u, d),
        #           where N: batch size, u: sequence length, d: dimension of feature
        # Output:   out(N, u, d)
        feats = self.feat_exa(src)
        seq = self.transformer_emb(feats.transpose(0, 1))  # seq: (u, N, d)
        out = F.relu(seq)
        # max_pool: (u, N, d) -> (N, d)
        # out = torch.max(seq, dim=0)[0]    # (N, d)
        return out.transpose(0, 1)


# 2 - Align Module
#   - 2.1 Bi-Attention
class BiAttnModel(nn.Module):
    def __init__(self):
        super(BiAttnModel, self).__init__()

    # Input:    latent_emb_mod 3 x (N, u, d)
    # func:     首先由单模态特征矩阵F生成三个双模态融合信息
    #           但如果直接将这三个 Attn(u, 2d) 的矩阵拼接，特征维度将大大上升
    #           因此采用一种全局注意力机制对这部分融合信息进行进一步的筛选，压缩到(u, 2d)
    # Output:   CCA-Bi(N, u, 2d)
    def forward(self, latent_emb_mod):
        a_emb, v_emb, l_emb = latent_emb_mod['a'], latent_emb_mod['v'], latent_emb_mod['l']
        attnAV, attnAL, attnVL = self.biAttn(a_emb, v_emb), self.biAttn(a_emb, l_emb), self.biAttn(v_emb, l_emb)

        return attnAV, attnAL, attnVL

    def biAttn(self, feat1, feat2):
        # Input:    feat1, feat2 (N, u, d)
        # Output:   Attn12(N, u, 2d)
        # 矩阵相乘，生成跨模态信息矩阵, (u, u)
        B12 = torch.matmul(feat1, feat2.transpose(1, 2))
        B21 = torch.matmul(feat2, feat1.transpose(1, 2))
        # softmax, 得到相关矩阵注意力分布, (u, u)
        N1 = F.softmax(B12, dim=1)
        N2 = F.softmax(B21, dim=1)
        # 矩阵乘法，生成注意力表征矩阵, (u, d)
        O1 = torch.matmul(N1, feat2)
        O2 = torch.matmul(N2, feat1)
        # 逐元素相乘，得到互注意力信息矩阵, (u, d)
        A1 = torch.mul(O1, feat1)
        A2 = torch.mul(O2, feat2)
        # concat, 融合信息表征, (u, 2d)
        Attn12 = torch.cat([A1, A2], dim=2)

        return Attn12


#   - 2.2 Tri-Attention
class TriAttnModel(nn.Module):
    def __init__(self):
        super(TriAttnModel, self).__init__()
        self.d = 32
        self.fc1 = nn.Linear(2 * self.d, self.d)
        self.fc2 = nn.Linear(2 * self.d, self.d)
        self.fc3 = nn.Linear(2 * self.d, self.d)
        self.relu = torch.relu
        self.softmax = F.softmax

    # Input:    latent_emb_mod 3 x (N, u, d)
    # Output:   CCA-Tri(N, u, 3d)
    def forward(self, latent_emb_mod):
        # 单模态特征矩阵
        F_a, F_v, F_l = latent_emb_mod['a'], latent_emb_mod['v'], latent_emb_mod['l']
        # 浅层融合的双模态信息矩阵
        F_av, F_al, F_vl = torch.cat([F_a, F_v], dim=2), torch.cat([F_a, F_l], dim=2), torch.cat([F_v, F_l], dim=2)
        F_av = self.relu(self.fc1(F_av))
        F_al = self.relu(self.fc2(F_al))
        F_vl = self.relu(self.fc3(F_vl))
        # 三模态融合特征
        attn_lav = self.triAttn(F_l, F_av)
        attn_val = self.triAttn(F_v, F_al)
        attn_avl = self.triAttn(F_a, F_vl)
        CCA = torch.cat([attn_lav, attn_val, attn_avl], dim=2)
        return CCA

    def triAttn(self, F1, F23):
        # Input:    F1 (u, d), F23 (u, d)
        # func:     输入单模态特征矩阵与双模态浅层融合特征，通过计算其注意力分布得到最终的三模态融合信息
        # Output:   Attn123 (u, d)
        # 跨模态信息矩阵, (u, u)
        C1 = torch.matmul(F1, F23.transpose(1, 2))
        # 注意力分布, (u, u)
        P1 = self.softmax(C1, dim=1)
        # 交互注意力表征信息, (u, d)
        T1 = torch.matmul(P1, F1)
        # 三模态融合信息, (u, d)
        Attn123 = torch.mul(T1, F23)

        return Attn123


class AlignModule(nn.Module):
    def __init__(self):
        super(AlignModule, self).__init__()
        self.bi_attn_model = BiAttnModel()
        self.tri_attn_model = TriAttnModel()

        self.d = 32
        self.fc1 = nn.Linear(2 * self.d, 2 * self.d)
        self.tanh = torch.tanh
        self.fc2 = nn.Linear(2 * self.d, 1, bias=False)
        self.softmax = F.softmax

    # Input:    latent_emb_mod 3 x (N, u, d)
    # func:     首先由单模态特征矩阵F生成三个双模态融合信息
    #           但如果直接将这三个 Attn(u, 2d) 的矩阵拼接，特征维度将大大上升
    #           因此采用一种全局注意力机制对这部分融合信息进行进一步的筛选，压缩到(u, 2d)
    # Output:   CCA-Bi(N, u, 2d) + CCA-Tri(N, u, 3d)
    def forward(self, latent_emb_mod):
        # 双模态交互特征
        attnAV, attnAL, attnVL = self.bi_attn_model(latent_emb_mod)
        seq_len = attnAV.size()[1]
        CCA = []
        for i in range(seq_len):
            Bi = torch.cat([attnAV[:, 0:1, :], attnAL[:, 0:1, :], attnVL[:, 0:1, :]], dim=1)
            Ci = self.fc2(self.tanh(self.fc1(Bi)))
            alpha = self.softmax(Ci, dim=0)
            CCA_i = torch.matmul(alpha.transpose(1, 2), Bi)
            CCA.append(CCA_i)
        bi_attn_emb = torch.cat(CCA, dim=1)
        # 三模态交互特征
        # tri_attn_emb = self.tri_attn_model(latent_emb_mod)
        tri_attn_emb = None

        return bi_attn_emb, tri_attn_emb


# 3 - Heterogeneity Module
# the uni-modal reference models, used to calculate $L^ref_m$ and update weights: $w_m$
class RefModel(nn.Module):
    def __init__(self, nfeat=32, nhid=16, dropout=0.4):
        super(RefModel, self).__init__()

        ninp = nfeat
        nout = 1
        self.fc1 = nn.Linear(ninp, 2 * nhid)
        self.fc2 = nn.Linear(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)
        self.sigm = nn.Sigmoid()

    def forward(self, latent_emb):
        # get $Y^ref_m$ with $H^latent_m$
        x = torch.mean(latent_emb, dim=1)
        x1 = self.fc1(x)
        x2 = F.relu(self.dropout(x1))
        x3 = self.fc2(x2)
        x4 = F.relu(x3)
        x5 = self.fc3(x4)
        out = self.sigm(x5)
        return out


class HetModule(nn.Module):
    def __init__(self, MODS, nfeat=32, nhid=16, dropout=0.4):
        super(HetModule, self).__init__()

        self.MODS = MODS
        self.ref_models = {mod: RefModel(nfeat, nhid, dropout) for mod in MODS}

    def forward(self, latent_emb_mod):
        # get $Y^ref_m$ with $H^latent_m$
        ref_emb_mod = {}
        for mod, latent_emb in latent_emb_mod.items():
            ref_emb_mod[mod] = self.ref_models[mod](latent_emb)

        return ref_emb_mod


# 4 - Persuasiveness Module
class PersModel(nn.Module):
    def __init__(self, nmod=3, nfeat=32, dropout=0.4):
        super(PersModel, self).__init__()

        # input: align_emb (5 * nfeat)  het_emb (1 * nfeat), debate meta-data (2)
        ninp = (1 + 2) * nfeat + 2
        nout = 1
        self.fc1 = nn.Linear(ninp, 2 * ninp)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2 * ninp, nout)
        self.sigm = nn.Sigmoid()

    # Input: attn_emb(N, 2d + 3d), het_emb(N, d), meta_emb(N, 2)
    def forward(self, align_emb, het_emb, meta_emb):
        x1 = torch.cat([align_emb, het_emb, meta_emb], dim=1)
        x2 = self.fc1(x1)
        x3 = F.relu(self.dropout(x2))
        x4 = self.fc2(x3)
        out = self.sigm(x4)
        return out

