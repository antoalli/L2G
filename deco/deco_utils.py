#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

"""
@Author: Antonio Alliegro
@Contact: antonio.alliegro@polito.it
@File: deco_utils.py
@Source: https://github.com/antoalli/Deco
"""

import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Identity(nn.Module):
    """ Simple Identity layer """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def weights_init_normal(m):
    """ Weights initialization with normal distribution.. Xavier """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def sanitize_model_dict(state_dict, to_remove_str='module'):
    # remove 'module' prefix if the model has been saved as DataParallel object
    new_state_dict = OrderedDict()
    remove_len = len(to_remove_str) + 1
    for k, v in state_dict.items():
        if str(k).startswith(to_remove_str):
            name = k[remove_len:]  # remove to_remove_str
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, in_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9:
            # If normals are present.. do not consider them in graph building!
            idx = knn(x[:, 6:], k=k)
        else:
            idx = knn(x, k=k)  # (batch_size, in_points, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature  # (batch_size, 2*num_dims, in_points, k)


def batched_index_select(x, dim, index):
    for i in range(1, len(x.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(x, dim, index)


def mlp(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]),
                      nn.BatchNorm1d(channels[i]),
                      nn.LeakyReLU(negative_slope=0.2),
                      ) for i in range(1, len(channels))])


class GlobalFeat(nn.Module):
    def __init__(self, k=30, emb_dims=1024, bn=True):
        super(GlobalFeat, self).__init__()
        self.k = k
        self.bn = bn

        # Removed batch normalization since GPNet requires training with bs=1
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64) if bn else Identity(),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64) if bn else Identity(),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128) if bn else Identity(),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256) if bn else Identity(),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(emb_dims) if bn else Identity(),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        bs, dim, npoints = x.size()
        assert dim == 3

        x = get_graph_feature(x, k=self.k)  # (bs, 3, npoints) -> (bs, 3*2, npoints, k)
        x = self.conv1(x)  # (bs, 3*2, npoints, k) -> (bs, 64, npoints, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (bs, 64, npoints, k) -> (bs, 64, npoints)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (bs, 64, npoints)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (bs, 128, npoints)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (bs, 256, npoints)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (bs, 64+64+128+256, npoints) ==> (bs, emb_dims, npoints)

        x = self.conv5(x)  # (bs, 64+64+128+256, in_points) -> (bs, emb_dims, npoints)
        x = F.adaptive_max_pool1d(x, 1).view(bs, -1)  # (bs, emb_dims, npoints) -> (bs, emb_dims)

        return x
