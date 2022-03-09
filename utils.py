#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
from collections import OrderedDict


def gather_by_idxs(source, idx):
    """"
    Input:
        source: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = source.device
    B = source.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = source[batch_indices, idx, :]
    return new_points


def set_random_seed(seed):
    """ set random seed """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_dataparallel(state_dict):
    res = False
    for k, _ in state_dict.items():
        if str(k).startswith('module'):
            res = True
            break
    return res


def sanitize_model_dict(state_dict, to_remove_str='module'):
    new_state_dict = OrderedDict()
    remove_len = len(to_remove_str) + 1
    for k, v in state_dict.items():
        if str(k).startswith(to_remove_str):
            name = k[remove_len:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def count_params(model):
    tot_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num params: {tot_params}, Trainable: {trainable_params}")
    return tot_params, trainable_params


def plot_grad_flow(named_parameters):
    """
    Credits to RoshanRane: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8
    :param named_parameters:
    :return:
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            if p.grad is None:
                warnings.warn('Grad is none for layer {}.'.format(n), RuntimeWarning)
                continue
            layers.append(n[:15])
            ave_grads.append(p.grad.cpu().abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()


class IOStream:
    """
    Generic Text Logger
    """
    def __init__(self, path):
        self.f = open(path, 'a')
        self.blue = lambda x: '\033[94m' + x + '\033[0m'
        self.red = lambda x: '\033[31m' + x + '\033[0m'

    def cprint(self, text, color=None):
        if color is not None and (color == 'b' or color == 'blue'):
            print(self.blue(text))
        elif color is not None and (color == 'r' or color == 'red'):
            print(self.red(text))
        else:
            print(text)

        self.f.write(text + '\n')
        self.f.flush()

    def fprint(self, text):
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def safe_make_dirs(dirs):
    if not isinstance(dirs, list):
        dirs = [dirs]
    for currdir in dirs:
        if not os.path.exists(currdir):
            print(f"Creating directory: {currdir}")
            os.makedirs(currdir)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_bold(pstring):
    print(bcolors.BOLD + pstring + bcolors.ENDC)


def print_warn(pstring):
    print(bcolors.WARNING + pstring + bcolors.ENDC)


def print_fail(pstring):
    print(bcolors.FAIL + pstring + bcolors.ENDC)


def print_ok(pstring):
    print(bcolors.OKBLUE + pstring + bcolors.ENDC)
