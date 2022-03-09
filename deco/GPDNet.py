#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

"""
@Author: Antonio Alliegro
@Contact: antonio.alliegro@polito.it
@File: GPDNet.py
@Source: https://github.com/antoalli/Deco
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from deco.deco_utils import Identity


def compute_graph(h):
    sq_norms = h * h
    sq_norms = torch.sum(sq_norms, 2)
    D = torch.abs(
        sq_norms.unsqueeze(2) + sq_norms.unsqueeze(1) - 2 * torch.matmul(h, h.transpose(1, 2)))  # (B, N, N)
    return D


def myroll(h, shift=0, axis=2):
    h_len = h.size()[2]
    d = int(h_len - shift)
    return torch.cat((h[:, :, d:], h[:, :, :d]), dim=axis)


def index_points(batch, idx):
    """
  Input:
      batch: input batch data, [B, N, C]
      idx: sample index data, [B, S]
  Return:
      new_batch: indexed data, [B, S, C] """

    device = batch.device
    B = batch.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_batch = batch[batch_indices, idx, :]
    return new_batch


class GConv(nn.Module):
    def __init__(self, in_feat, fnet_feat, out_feat,
                 rank_theta, stride_th1, stride_th2, min_nn):
        super(GConv, self).__init__()
        self.in_feat = in_feat
        self.fnet_feat = fnet_feat
        self.out_feat = out_feat
        self.rank_theta = rank_theta
        self.stride_th1 = stride_th1
        self.stride_th2 = stride_th2
        self.min_nn = min_nn

        # create_gconv_variables_dn() :
        self.W_flayer_th0 = torch.nn.Parameter(data=torch.Tensor(in_feat, fnet_feat), requires_grad=True)
        torch.nn.init.xavier_normal_(self.W_flayer_th0, gain=1.0)
        self.b_flayer_th0 = torch.nn.Parameter(data=torch.Tensor(1, fnet_feat), requires_grad=True)
        torch.nn.init.zeros_(self.b_flayer_th0)

        self.W_flayer_th1 = torch.nn.Parameter(data=torch.Tensor(fnet_feat, int(stride_th1) * rank_theta),
                                               requires_grad=True)
        torch.nn.init.normal_(self.W_flayer_th1, mean=0, std=1.0 / (np.sqrt(fnet_feat + 0.0) * np.sqrt(in_feat + 0.0)))
        self.b_flayer_th1 = torch.nn.Parameter(data=torch.Tensor(1, rank_theta, in_feat), requires_grad=True)
        torch.nn.init.zeros_(self.b_flayer_th1)

        self.W_flayer_th2 = torch.nn.Parameter(data=torch.Tensor(fnet_feat, int(stride_th2) * rank_theta),
                                               requires_grad=True)
        torch.nn.init.normal_(self.W_flayer_th2, mean=0, std=1.0 / (np.sqrt(fnet_feat + 0.0) * np.sqrt(in_feat + 0.0)))
        self.b_flayer_th2 = torch.nn.Parameter(data=torch.Tensor(1, rank_theta, out_feat), requires_grad=True)
        torch.nn.init.zeros_(self.b_flayer_th2)

        self.W_flayer_thl = torch.nn.Parameter(data=torch.Tensor(fnet_feat, rank_theta), requires_grad=True)
        torch.nn.init.normal_(self.W_flayer_thl, mean=0, std=1.0 / np.sqrt(rank_theta + 0.0))
        self.b_flayer_thl = torch.nn.Parameter(data=torch.Tensor(1, rank_theta), requires_grad=True)
        torch.nn.init.zeros_(self.b_flayer_thl)

    def forward(self, h, D):
        B, N, f = h.size()
        assert f == self.in_feat
        assert D is not None

        # if D is None:
        #     # No previous graph computed..
        #     D = compute_graph(h)

        _, top_idx = torch.topk(-D, k=(self.min_nn + 1),
                                dim=-1)  # (-D is fundamental for following line (i.e. 22)) - [B, N, min_nn + 1]
        # not that 'repeat_interleave' is not available in torch < 1.1.0
        top_idx2 = top_idx[:, :, 0].repeat_interleave(self.min_nn, 1)  # expected: [B, N*min_nn]
        top_idx = top_idx[:, :, 1:].reshape(-1, N * self.min_nn)  # [B, N, min_nn] -> [B, N*min_nn]
        x_tilde1 = index_points(h, top_idx)  # [B, N*min_nn, in_feat]
        x_tilde2 = index_points(h, top_idx2)  # [B, N*min_nn, in_feat]
        labels = x_tilde1 - x_tilde2  # [B, N*min_nn, in_feat]
        x_tilde1 = x_tilde1.view(-1, self.in_feat)  # [B*N*min_nn, in_feat]
        labels = labels.view(-1, self.in_feat)  # [B*N*min_nn, in_feat]
        d_labels = torch.sum(labels * labels, dim=1).view(-1, self.min_nn)
        # print("labels: ", labels.size())
        # print("d_labels: ", d_labels.size())

        # FORWARD
        labels = F.leaky_relu(torch.matmul(labels, self.W_flayer_th0) + self.b_flayer_th0, negative_slope=0.02)
        labels_exp = labels.unsqueeze(1)  # (B*K, 1, F)

        labels1 = labels_exp + 0.0  # [B*K, 3, F=in_feat]
        for ss in range(1, int(self.in_feat / self.stride_th1)):
            labels1 = torch.cat((labels1, myroll(labels_exp, shift=(ss + 1) * self.stride_th1, axis=2)),
                                axis=1)  # (B*K, dlm1/stride, dlm1)

        labels2 = labels_exp + 0.0  # [B*K, 3, F=in_feat]
        for ss in range(1, int(self.out_feat / self.stride_th2)):
            labels2 = torch.cat((labels2, myroll(labels_exp, shift=(ss + 1) * self.stride_th2, axis=2)),
                                axis=1)  # (B*K, dl/stride, dlm1)

        theta1 = torch.matmul(labels1.view(-1, self.in_feat), self.W_flayer_th1)  # (B*K*dlm1/stride, R*stride)
        theta1 = theta1.view(-1, self.rank_theta, self.in_feat) + self.b_flayer_th1

        theta2 = torch.matmul(labels2.view(-1, self.in_feat), self.W_flayer_th2)  # (B*K*dl/stride, R*stride)
        theta2 = theta2.view(-1, self.rank_theta, self.out_feat) + self.b_flayer_th2

        thetal = torch.matmul(labels, self.W_flayer_thl) + self.b_flayer_thl  # (B*K, R)
        thetal = thetal.unsqueeze(2)  # (B*K, R, 1)

        x = torch.matmul(theta1, x_tilde1.unsqueeze(2))  # (B*K, R, 1)
        x = torch.mul(x, thetal)
        # print("theta2: ", theta2.size())
        # print("theta2 permute: ", theta2.permute(0, 2, 1).size())
        # print("x: ", x.size())
        x = torch.matmul(theta2.permute(0, 2, 1), x)[:, :, 0]  # (B*K, dl)

        x = x.view(-1, self.min_nn, self.out_feat)  # (N, d, dl)
        x = torch.mul(x, torch.exp(-torch.div(d_labels, 10)).unsqueeze(2))
        # x = torch.sum(x, 1).reshape(-1, N, self.out_feat)  # [B, N, out_feat]
        x = torch.sum(x, 1).reshape(-1, self.out_feat, N)  # [B, out_feat, N]
        return x


class ConvLayer(nn.Module):
    def __init__(self, layer_conf, bn=True, last=False):
        super(ConvLayer, self).__init__()

        self.block_config = layer_conf
        self.last = last

        self.gconv = GConv(in_feat=layer_conf['in_feat'],
                           fnet_feat=layer_conf['fnet_feat'],
                           out_feat=layer_conf['out_feat'],
                           rank_theta=layer_conf['rank_theta'],
                           stride_th1=layer_conf['stride_th1'],
                           stride_th2=layer_conf['stride_th2'],
                           min_nn=layer_conf['min_nn'], )

        # SL: self loop
        # SELF LOOP: feature del punto i-esimo / centroide...
        self.sl_W = nn.Parameter(data=torch.Tensor(layer_conf['out_feat'], layer_conf['in_feat'], 1),
                                 requires_grad=True)  # SL Weights
        self.sl_b = nn.Parameter(data=torch.Tensor(layer_conf['out_feat'], 1), requires_grad=True)  # SL bias
        nn.init.xavier_normal_(self.sl_W, gain=1.0)
        nn.init.zeros_(self.sl_b)

        if not last:
            self.BN_layer = nn.BatchNorm1d(layer_conf['out_feat']) if bn else Identity()
            self.non_lin = nn.LeakyReLU(negative_slope=0.2)

    def lnl_aggregation(self, h_l, h_nl, b):
        assert h_l.size() == h_nl.size()
        return torch.div(h_l + h_nl, self.block_config['min_nn'] + 1) + b

    def forward(self, t):
        h, D = t  # [B, N, in_feat], None or [B, N, N]
        if D is None:
            # print("Computing graph")
            D = compute_graph(h)

        h_nl = self.gconv(h, D)  # [B, N, in_feat] -> [B, out_feat, N]
        h_sl = F.conv1d(
            input=h.permute(0, 2, 1), weight=self.sl_W, bias=None, stride=1)  # [B, N, in_feat] -> [B, out_feat, N]

        # print("h_nl ", h_nl.size())
        # print("h_sl ", h_sl.size())

        h = self.lnl_aggregation(h_sl, h_nl, self.sl_b)

        if self.last:
            """ If last layer should be no BatchNorm nor LeakyReLU """
            return h.permute(0, 2, 1), D  # ([BS, N, out_feat: in_channels], D)

        h = self.non_lin(self.BN_layer(h)).permute(0, 2, 1)  # [BS, out_feat, N] -> [BS, N, out_feat]
        return h, D  # ([BS, N, out_feat], D) returns tuple (since it is in nn.Sequential) to next gconv!


def get_mlp_1d(channels, bn=True):
    """
    MLP with Conv1D (kernel_size = 1) + BatchNorm1D + LeakyRelu """

    return Seq(*[
        Seq(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=(not bn)),
            nn.BatchNorm1d(channels[i]) if bn else Identity(),
            nn.LeakyReLU(negative_slope=0.2),
            ) for i in range(1, len(channels))])


class GPDLocalFE(nn.Module):
    def __init__(self, configuration):
        super(GPDLocalFE, self).__init__()

        # PRE
        pre_layers = configuration["pre_Nfeat"]
        self.pre_nn_layers = get_mlp_1d(pre_layers)

        # RESIDUAL BLOCKS
        self.residual_block1 = Seq()
        for j in range(configuration["conv_n_layers"]):
            self.residual_block1.add_module(name="conv_{}_layer".format(j),
                                            module=ConvLayer(layer_conf=configuration["conv_layer"]))

        self.residual_block2 = Seq()
        for j in range(configuration["conv_n_layers"]):
            self.residual_block2.add_module(name="conv_{}_layer".format(j),
                                            module=ConvLayer(layer_conf=configuration["conv_layer"]))

        # LAST
        last_l_conf = configuration["lconv_layer"]
        self.l_conv = ConvLayer(layer_conf=last_l_conf, last=True)

        # Xavier Weights initialization: nn.Conv1d, nn.Linear, BatchNorm1d
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):  # very similar to resnet
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, points):
        points = points.permute(0, 2, 1)  # [B, N, 3] -> [B, 3, N]

        # PRE - Simply MLP1Ds
        h = self.pre_nn_layers(points).permute(0, 2, 1)  # [B, N, in_channels] -> [B, N, in_feat]

        # RESIDUAL BLOCK 1
        x1, _ = self.residual_block1((h, None))  # None: compute graph 1, [B, N, in_feat] -> [B, N, out_feat]
        h = x1 + h  # residual 1

        # RESIDUAL BLOCK 2
        x2, _ = self.residual_block2((h, None))  # computing graph 2
        h = x2 + h  # residual 2

        # LAST BLOCK out_feat -> in_channels
        x3, _ = self.l_conv((h, None))  # computing graph 3
        return x3
