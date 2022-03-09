import torch.nn as nn


class Identity(nn.Module):
    """ Simple Identity layer """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
