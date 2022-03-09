import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule


def build_pn2_encoder(input_channels=0, bn=False, return_global=False):
    return Pointnet2MSG(input_channels=input_channels, bn=bn, return_global=return_global)


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels, bn, return_global):
        """
        input_channels: 0 if w/o normals
        bn: whether to use or not batch normalization
        """
        super().__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.return_global = return_global

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=True,
                bn=bn,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=True,
                bn=bn,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=True,
                bn=bn,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=True,
                bn=bn,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512], bn=bn))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        bs = pointcloud.size(0)
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        sa_glob_feat = l_features[-1].max(2)[0].view(bs, -1)  # [bs, 1024]

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        feat = l_features[0]  # per-point features after feature propagation layers [bs, 128, npoints]

        if self.return_global:
            # point -> glob
            global_feat = torch.max(feat, dim=2)[0].view(bs, -1)  # [bs, 128]
            global_feat = torch.cat([global_feat, sa_glob_feat], dim=-1)  # [bs, 128 + 1024]
            return feat, global_feat
        else:
            return feat


if __name__ == '__main__':
    model = build_pn2_encoder(input_channels=0, bn=False, return_global=True).cuda()
    t = torch.rand(32, 2048, 3).cuda()
    point_feat, glob_feat = model(t)
    print("point_feat ", point_feat.shape)
    print("glob_feat ", glob_feat.shape)
