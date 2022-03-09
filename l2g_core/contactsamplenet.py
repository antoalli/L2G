from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
from l2g_core.chamfer_distance.chamfer_distance import ChamferDistance
from l2g_core.utils.identity import Identity
from l2g_core.utils.hausdorff import directed_hausdorff
from l2g_core.soft_projection import SoftProjection
from l2g_core import sputils

assert torch.cuda.is_available(), "ContactSampleNet: CUDA is not available"


class ContactSampleNet(nn.Module):
    def __init__(
            self,
            num_out_points,
            bottleneck_size,
            group_size,
            initial_temperature=1.0,
            is_temperature_trainable=True,
            min_sigma=1e-2,
            input_shape="bnc",
            output_shape="bnc",
            bn=True,
            simp_loss='chamfer'
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.bn = bn
        self.name = "contactsamplenet"

        self.fc1 = nn.Linear(bottleneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * num_out_points)

        self.bn_fc1 = nn.BatchNorm1d(256) if bn else Identity()
        self.bn_fc2 = nn.BatchNorm1d(256) if bn else Identity()
        self.bn_fc3 = nn.BatchNorm1d(256) if bn else Identity()

        # projection and matching
        self.is_temperature_trainable = is_temperature_trainable
        self.project = SoftProjection(
            group_size, initial_temperature, is_temperature_trainable, min_sigma
        )
        self.simp_loss = simp_loss
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor, global_feat: torch.Tensor):

        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)  # point cloud input shape must be B x 3 X N

        y = global_feat
        y = F.relu(self.bn_fc1(self.fc1(y)))
        y = F.relu(self.bn_fc2(self.fc2(y)))
        y = F.relu(self.bn_fc3(self.fc3(y)))
        y = self.fc4(y)
        y = y.view(-1, 3, self.num_out_points)

        generated = y  # generated: points not guaranteed to be on the shape surface

        # project on whole input pointcloud
        proj = self.project(point_cloud=x, query_cloud=y)  # projected points are weighted combination of shape surface points
        match = None  # for each point in 'generated' the nearest (KNN-1) point on the shape surface (HARD SAMPLING)

        # HARD SAMPLING
        if not self.training:
            # Retrieve nearest neighbor indices
            _, idx = KNN(1, transpose_mode=False)(x.contiguous(), y.contiguous())

            # Convert to numpy arrays in B x N x 3 format. we assume 'bcn' format.
            x = x.permute(0, 2, 1).cpu().detach().numpy()
            y = y.permute(0, 2, 1).cpu().detach().numpy()

            idx = idx.cpu().detach().numpy()
            idx = np.squeeze(idx)
            idx, counts = np.unique(idx, return_counts=True)
            idx = np.reshape(idx, (1, -1))

            z = sputils.nn_matching(x, idx, idx.shape[1], complete_fps=False, counts=counts)

            # Matched points are in B x N x 3 format.
            match = torch.tensor(z, dtype=torch.float32).cuda()

        # Change to output shapes
        if self.output_shape == "bnc":
            generated = generated.permute(0, 2, 1)
            if proj is not None:
                proj = proj.permute(0, 2, 1)
        elif self.output_shape == "bcn" and match is not None:
            match = match.permute(0, 2, 1)
            match = match.contiguous()

        # Assert contiguous tensors
        generated = generated.contiguous()
        if proj is not None:
            proj = proj.contiguous()
        if match is not None:
            match = match.contiguous()

        if self.training:
            sampled = proj
        else:
            sampled = match

        return generated, sampled

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.
    def get_simplification_loss(self, samp_pc, ref_pc, gamma=1):
        """
        Computes the simplification loss (either chamfer or hausdorff)
        @param ref_pc: the ground truth points
        @param samp_pc: the sampled points
        @param gamma: the weight of the second term of the chamfer distance (if chamfer is used)
        @return: the selected loss
        """
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(ref_pc)

        if self.simp_loss == 'chamfer':
            if ref_pc.shape[2] != 3:
                ref_pc = ref_pc.permute(0, 2, 1)
                samp_pc = samp_pc.permute(0, 2, 1)
            assert ref_pc.shape[2] == 3 and samp_pc.shape[2] == 3, "Chamfer input shape must be B X N X 3."

            # print("samp_pc: ", samp_pc.shape)
            # print("ref_pc: ", ref_pc.shape)

            cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
            max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
            max_cost = torch.mean(max_cost)
            cost_p1_p2 = torch.mean(cost_p1_p2)
            cost_p2_p1 = torch.mean(cost_p2_p1)
            loss = cost_p1_p2 + max_cost + gamma * cost_p2_p1
            return cost_p1_p2, max_cost, cost_p2_p1, loss
        else:
            # Hausdorff loss
            if ref_pc.shape[1] != 3:
                ref_pc = ref_pc.permute(0, 2, 1)
                samp_pc = samp_pc.permute(0, 2, 1)
            assert ref_pc.shape[-1] == 3 and samp_pc.shape[-1] == 3, "Hausdorff input shape must be B X 3 X N."

            loss = directed_hausdorff(ref_pc, samp_pc)
            return 0, 0, 0, loss

    def get_projection_loss(self):
        sigma = self.project.sigma()
        if not self.training or not self.is_temperature_trainable:
            return torch.tensor(0).to(sigma)
        return sigma
