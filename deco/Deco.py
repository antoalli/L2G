#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

"""
@Author: Antonio Alliegro
@Contact: antonio.alliegro@polito.it
@File: Deco.py
@Source: https://github.com/antoalli/Deco
"""

from deco.GPDNet import *
from deco.deco_utils import *


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


class GPDLocalFeat(nn.Module):
    def __init__(self, configuration, bn=True):
        super(GPDLocalFeat, self).__init__()

        self.bn = bn

        self.pre_nn_layers = get_mlp_1d([3, 33, 66, 99], bn=bn)

        self.residual_block1 = Seq()
        for j in range(configuration["conv_n_layers"]):
            self.residual_block1.add_module(name="conv_{}_layer".format(j),
                                            module=ConvLayer(layer_conf=configuration["conv_layer"], bn=bn))
        self.residual_block2 = Seq()
        for j in range(configuration["conv_n_layers"]):
            self.residual_block2.add_module(name="conv_{}_layer".format(j),
                                            module=ConvLayer(layer_conf=configuration["conv_layer"], bn=bn))
        self._initialize_weights()

    def _initialize_weights(self):
        """ Init: only affecting standard pytorch layer (Convs and BatchNorm) """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):  # very similar to resnet
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, points):
        assert points.size(1) == 3

        # Pre
        h = self.pre_nn_layers(points)  # -> [B, in_feat, N]
        h = h.permute(0, 2, 1)

        # Res1
        x1, _ = self.residual_block1((h, None))  # None: compute graph 1, [B, N, in_feat] -> [B, N, out_feat]
        h = x1 + h  # residual 1

        # Res2
        x2, _ = self.residual_block2((h, None))  # computing graph 2
        h = x2 + h  # residual 2
        return h.permute(0, 2, 1)


class GLEncoder(nn.Module):
    def __init__(self, conf, bn=True, return_global=False):
        super(GLEncoder, self).__init__()

        self.global_encoder = GlobalFeat(
            k=conf['global_encoder']['nearest_neighboors'],
            emb_dims=conf['global_encoder']['latent_dim'],
            bn=bn)

        self.local_encoder = GPDLocalFeat(
            configuration=conf['GPD_local'],
            bn=bn)

        global_emb_dim = conf['global_encoder']['latent_dim']
        output_dim = conf["aggr_out_feat"]  # local + global output embedding size

        self.nn_local = nn.Conv1d(
            in_channels=99,
            out_channels=output_dim,
            kernel_size=1,
            bias=(bn is False))

        self.nn_global = nn.Linear(
            in_features=global_emb_dim,
            out_features=output_dim,
            bias=(bn is False))

        self.nn_aggr_bn = nn.BatchNorm1d(output_dim) if bn else Identity()
        self.nn_aggr_relu = nn.LeakyReLU(negative_slope=0.2)

        self.return_global = return_global

    def forward(self, points):
        """
        Parameters
        ----------
        points: tensor([bs, dim, npoints])

        Returns
        -------

        """

        if points.shape[1] != 3:
            points = points.permute(0, 2, 1)

        bs = points.size(0)

        """ Features from the two branch encoder """
        l_feat = self.local_encoder(points)  # [bs, 99, npoints]
        g_feat = self.global_encoder(points)  # [bs, 1024]

        """ Local + Global Aggregation """
        l_feat_aggr = self.nn_local(l_feat)  # [bs, out_dim, npoints]
        g_feat_aggr = self.nn_global(g_feat).unsqueeze(-1)  # [bs, out_dim, 1]
        feat = l_feat_aggr + g_feat_aggr
        feat = self.nn_aggr_bn(feat)
        feat = self.nn_aggr_relu(feat)

        if self.return_global:
            global_feat = torch.max(feat, dim=2)[0].view(bs, -1)  # [bs, out_dim, npoints] => [bs, out_dim]
            g_feat = g_feat.view(bs, -1)  # [bs, 1024]
            global_feat = torch.cat([global_feat, g_feat], dim=-1)  # [bs, out_dim + 1024]
            return feat, global_feat
        else:
            return feat


def build_deco_encoder(bn=True, config=None, return_global=False, no_pretext=False):
    assert config is not None, "config is none!!!"
    model = GLEncoder(conf=config, bn=bn, return_global=return_global)
    model.apply(weights_init_normal)  # affecting only non pretrained layers

    if no_pretext:
        print_warn("Deco Pretext Weights not loaded!!!")
        return model

    local_fe_fn = config['pretrain']['checkpoint_local_enco']
    global_fe_fn = config['pretrain']['checkpoint_global_enco']

    # TODO: use InstanceNorm instead of BN
    print("-" * 30)
    print_bold(f"DeCo config: {config}\n")

    # load local encoder (denoising) pretext
    if len(local_fe_fn) > 0:
        local_enco_dict = torch.load(local_fe_fn)['model_state_dict']
        loc_load_result = model.local_encoder.load_state_dict(local_enco_dict, strict=False)
        print_warn(f"Deco - local pretext loading: \n{str(loc_load_result)} \n")  # TODO: remove bn warnings
    else:
        print_fail("Deco - local enco weights NOT loaded \n")

    # load global encoder (contrastive) pretext
    if len(global_fe_fn) > 0:
        global_enco_dict = torch.load(global_fe_fn, )['global_encoder']
        glob_load_result = model.global_encoder.load_state_dict(
            global_enco_dict,
            strict=bn
        )
        print_warn(f"Deco - global pretext loading: \n{str(glob_load_result)} \n")  # TODO: remove bn warnings
    else:
        print_fail("Deco - global enco weights NOT loaded \n")

    print("-" * 30)
    print("")
    return model


if __name__ == '__main__':
    # just a test
    import yaml
    with open("deco/deco_config.yaml", "r") as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    model = build_deco_encoder(bn=False, config=config, return_global=True, no_pretext=False).cuda()
    points = torch.rand(1, 3, 9000).cuda()
    point_feat, glob_feat = model(points)
    print(f"point_feat: {point_feat.shape}")
    print(f"glob_feat: {glob_feat.shape}")


