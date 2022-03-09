import os
import warnings
import yaml
import torch.nn as nn
import torch
from l2g_core.utils.identity import Identity
from l2g_core.utils.grasp_utils import reparametrize_grasps
from l2g_core import pytorch_utils as p_utils  # different version from the one used in pn2 modules!
from l2g_core.contactsamplenet import ContactSampleNet
from utils import gather_by_idxs
from utils import print_bold
from encoders import *


class PosePredictBatch(nn.Module):
    def __init__(self, input_dim=256, point_num=50, bn=False, use_tanh=True, feat_aggregator="avg"):
        super(PosePredictBatch, self).__init__()

        assert feat_aggregator in ["max", "avg", "w_avg"], f"feat_aggregator unknown: {feat_aggregator}"
        self.input_dim = input_dim

        if feat_aggregator == "max":
            self.feat_aggregator = nn.MaxPool2d((1, point_num), stride=(1, 1))
        elif feat_aggregator == "avg":
            self.feat_aggregator = nn.AvgPool2d((1, point_num), stride=(1, 1))
        else:
            # Conv2d used like that works like a learned weighted avrrage
            self.feat_aggregator = p_utils.Conv2d(input_dim, input_dim, kernel_size=(1, point_num), padding=0, bn=bn)

        # the first conv layer is shared between angle and second contact predictor
        self.conv1 = p_utils.Conv1d(input_dim, input_dim, 1, bn=bn)
        self.second_contact_predictor = nn.Sequential(
            p_utils.Conv1d(input_dim, 256, 1, bn=bn),
            p_utils.Conv1d(256, 3, 1, bn=bn, activation=None)
        )
        self.angle_predictor = nn.Sequential(
            p_utils.Conv1d(input_dim, 256, 1, bn=bn),
            p_utils.Conv1d(256, 1, 1, bn=bn, activation=None)
        )

        self.activation = nn.Tanh() if use_tanh else Identity()

    def forward(self, gather_feat):
        # feat_aggregator needs the input to be in the shape [B x C x M x K]
        feat = self.feat_aggregator(gather_feat.permute(0, 3, 1, 2)).squeeze(-1)
        feat = self.conv1(feat)

        second_contact = self.second_contact_predictor(feat)
        angle = self.angle_predictor(feat)

        second_contact = self.activation(second_contact)
        angle = self.activation(angle)

        return second_contact.permute(0, 2, 1), angle.permute(0, 2, 1)

    def get_prediction_loss(self, predicted, truth, angle_contribution=1.0):
        """
        # TODO: not true anymore... UPDATE DOCUMENTATION SOON (Alli 17 dic 2021)
        Computes the prediction loss as mean of min grasp distances between predicted and truth
        The grasp distance used is the one reported in (https://arxiv.org/pdf/1912.05604.pdf)
        @param predicted: the predicted grasps
        @param truth: the ground truth positive grasps in the dataset
        @param angle_contribution: the weight of the angle distance
        @return: the prediction loss
        """

        # swap first and second contact point and then concatenate with truth
        swapped_truth = torch.cat([truth[:, :, 3:6], truth[:, :, :3], truth[:, :, -1].unsqueeze(-1)], dim=-1)
        new_truth = torch.cat([truth, swapped_truth], dim=1)  # [bs, 2 x num_grasps, 3+3+1]
        all_contacts = new_truth[:, :, :3]  # [bs, 2 x num_grasps, 3]

        # predicted: [bs, num_grasp_pred, 3 + 3 + 1 ]
        predicted_contacts = predicted[:, :, :3]
        pred_truth_dist = torch.cdist(predicted_contacts, all_contacts)  # [bs, num_grasp_pred, 2 x num_grasps]
        # consideriamo corrispondenza 1:1 tra punto samplato e grasp
        _, min_dist_idxs = torch.min(pred_truth_dist, dim=-1)  # [bs, num_grasp_pred]

        # if (torch.max(min_dist_idxs).item() >= truth.shape[1]):
        #     print("makes sense to swap")

        per_sampled_points_truth = gather_by_idxs(new_truth, min_dist_idxs.long())  # [bs, num_grasp_pred, 3+3+1]

        # need to be in the (center, quaternion) parametrization to compute the loss
        rep_predicted = reparametrize_grasps(predicted)
        rep_truth = reparametrize_grasps(per_sampled_points_truth)

        predicted_centers = rep_predicted[:, :, :3].squeeze(0).contiguous()  # [M x 3]
        truth_centers = rep_truth[:, :, :3].squeeze(0).contiguous()  # [M x 3]
        # center_dist = torch.cdist(predicted_centers, truth_centers)  # [M x G]
        center_loss = nn.functional.pairwise_distance(predicted_centers, truth_centers, p=2).mean()

        eps = 1e-5
        predicted_quaternions = rep_predicted[:, :, 3:7].squeeze(0)  # [M x 4]
        truth_quaternions = rep_truth[:, :, 3:7].squeeze(0)  # [M x 4]
        scalar_product = torch.diagonal(torch.abs(torch.matmul(predicted_quaternions, truth_quaternions.t())))  # [M]
        quaternion_dist = torch.acos(torch.clamp(scalar_product, max=1 - eps))
        quaternion_loss = torch.mean(quaternion_dist)

        loss = center_loss + angle_contribution * quaternion_loss

        return center_loss, quaternion_loss, loss


class ContactAngleFeat(nn.Module):
    def __init__(self, in_dim=4, out_dim=128, bn=False, depth=2):
        assert depth in [2, 3], "the depth of this module could either be 2 or 3"
        super(ContactAngleFeat, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if depth == 3:
            self.layers = nn.Sequential(
                p_utils.Conv1d(in_dim, 64, 1, bn=bn),
                p_utils.Conv1d(64, 128, 1, bn=bn),
                p_utils.Conv1d(128, out_dim, bn=bn)
            )
        else:
            self.layers = nn.Sequential(
                p_utils.Conv1d(in_dim, 64, 1, bn=bn),
                p_utils.Conv1d(64, out_dim, bn=bn)
            )

    def forward(self, x: torch.Tensor):
        if x.shape[1] != self.in_dim:
            x = x.permute(0, 2, 1)

        assert x.shape[1] == self.in_dim, "Input tensor must be in shape [B x 4 x M]"

        feat = self.layers(x)
        return feat.permute(0, 2, 1)  # [B x M x C]


class GraspClassifier(nn.Module):
    def __init__(self, input_dim=256, point_num=50, bn=False, feat_aggregator="avg"):
        super(GraspClassifier, self).__init__()
        self.input_dim = input_dim

        if feat_aggregator == "max":
            self.feat_aggregator = nn.MaxPool2d((1, point_num), stride=(1, 1))
        elif feat_aggregator == "avg":
            self.feat_aggregator = nn.AvgPool2d((1, point_num), stride=(1, 1))
        elif feat_aggregator == "w_avg":
            # Conv2d works like a learned weighted average
            self.feat_aggregator = p_utils.Conv2d(input_dim, input_dim, kernel_size=(1, point_num), padding=0, bn=bn)
        else:
            raise ValueError(f"Wrong feat_aggregator: {feat_aggregator}")

        self.conv1 = p_utils.Conv1d(input_dim, 512, 1, bn=bn)
        self.score = nn.Sequential(
            p_utils.Conv1d(512, 256, 1, bn=bn),
            p_utils.Conv1d(256, 1, 1, bn=bn, activation=None),
        )
        self.sigmoid = nn.Sigmoid()
        self.loss_criterion = nn.BCELoss()

    def forward(self, gather_feat):
        feat = self.feat_aggregator(gather_feat.permute(0, 3, 1, 2)).squeeze(-1)
        feat = self.conv1(feat)
        score = self.sigmoid(self.score(feat))
        return score.permute(0, 2, 1)  # [B x M x 1]

    def get_classification_loss(self, predicted, truth):
        loss = self.loss_criterion(predicted, truth)
        return loss


class GraspSampleNet(nn.Module):
    def __init__(self,
                 feat_extractor='deco',
                 sampled_grasps=400,
                 sample_group_size=10,
                 simp_loss='chamfer',
                 train_temperature=True,
                 neigh_size=50,
                 use_all_grasp_info=False,
                 use_contact_angle_feat=True,
                 angle_feat_depth=2,
                 projected_feat_aggregation="w_avg",
                 bn=False,
                 use_tanh=True,
                 deco_config_path=None,
                 resume=False  # not loading pretexts in case of 'resume' or 'test'
                 ):
        super().__init__()

        if use_all_grasp_info:
            raise NotImplementedError('use_all_grasp_info: experimental!')

        self.sampled_grasps = sampled_grasps
        self.sample_group_size = sample_group_size
        self.projected_feat_aggregation = projected_feat_aggregation
        self.point_feat_dim = 128  # per-point features dim must be the same across all feature extractors
        self.neigh_size = neigh_size
        self.use_all_grasp_info = use_all_grasp_info
        self.use_contact_angle_feat = use_contact_angle_feat

        # Build feature extractor
        if feat_extractor == 'pointnet2':
            self.encoder = build_pn2_encoder(input_channels=0, bn=bn, return_global=True)
            self.glob_feat_dim = (128 + 1024)  # feat from last SA + pooled point-feat
        elif feat_extractor == 'deco':
            assert deco_config_path is not None and os.path.isfile(deco_config_path)
            with open(deco_config_path, "r") as cf:
                config = yaml.load(cf, Loader=yaml.FullLoader)
                print(f"DeCo encoder loaded config from {deco_config_path}: \n{config}")
            self.glob_feat_dim = (config['aggr_out_feat'] + 1024)  # feat from global branch + pooled point-feat
            self.encoder = build_deco_encoder(
                bn=bn, config=config, return_global=True, no_pretext=resume)
        else:
            raise ValueError(f"Unknown feat_extractor: {feat_extractor}")

        print(
            f"GraspSampleNet - feat_extractor: {feat_extractor}, "
            f"point_feat_dim: {self.point_feat_dim}, "
            f"glob_feat_dim: {self.glob_feat_dim}")

        # sampler:
        # input: shape global feature descriptor
        self.sampler = ContactSampleNet(
            num_out_points=self.sampled_grasps,
            bottleneck_size=self.glob_feat_dim,
            is_temperature_trainable=train_temperature,
            group_size=sample_group_size,
            bn=bn,
            simp_loss=simp_loss,
        )

        # grasp_predictor
        # input: sampled first contact neigh-feat [B x M x K x C]
        # output: c2: [B x M x 3], angle: [B x M x 1]
        self.grasp_predictor = PosePredictBatch(
            input_dim=self.point_feat_dim,
            point_num=self.neigh_size,
            bn=bn,
            feat_aggregator=projected_feat_aggregation,
            use_tanh=use_tanh
        )

        # grasp classifier
        in_dim = 7 if self.use_all_grasp_info else 4  # in_dim to grasp_cla - whether to append first sampled contact (7) or not (4)
        if self.use_contact_angle_feat:
            # default - as in GPNet!
            # mlp module expands: in_dim -> feat_dim
            self.contact_angle_feat_extractor = ContactAngleFeat(
                in_dim=in_dim, out_dim=self.point_feat_dim, bn=bn, depth=angle_feat_depth)
            grasp_classifier_input_dim = self.point_feat_dim
        else:
            # 2. keep the xyz coordinates and the angle as they are and later concatenate with neigh-like first contact feat
            self.contact_angle_feat_extractor = Identity()
            grasp_classifier_input_dim = self.point_feat_dim + in_dim

        # GraspClassifier takes as input
        # - the grasp features, made up of a combination of
        #    - the neigh-like features of the first contact point
        #    - either the features or the coordinates of the second contact + the angle
        # and predicts
        # - grasp success probability
        self.grasp_classifier = GraspClassifier(
            input_dim=grasp_classifier_input_dim,
            point_num=self.neigh_size,
            bn=bn,
            feat_aggregator=projected_feat_aggregation
        )

    def compute_neigh_features(self, x, y, point_feat, k, metric='euclidean'):
        """
        This function generates neighbourhood-like features for a given point cloud (y), based on point-wise features of a
        pre-computed reference point cloud (x). For each point in y, it looks for the k closest point in x and gather
        their features together.
        @param x: the reference point cloud [B x N x 3]
        @param y: the point cloud from which is needed to extract features from (may be the same as x) [B x M x 3]
        @param point_feat: the point features extracted from the reference pc [B x C x N]
        @param k : the neighborhood size
        @param metric: the distance metric to use ('euclidean' or 'feature')
        @return: the neighbourhood like features for y [B x M x K x C]
        """
        assert metric in ['euclidean', 'feature']
        # need to permute point feat to B x N x C
        if point_feat.shape[1] != x.shape[1]:
            point_feat = point_feat.permute(0, 2, 1)
        assert point_feat.shape[1] == x.shape[1], "Point features must be of the same size of the points."

        # for each projected point, I need to get the K closest point in the pc
        # first, compute the distances between the sampled and original points
        if metric == 'euclidean':
            distance_matrix = - torch.cdist(y, x)  # [B x M x N]
        else:
            raise NotImplementedError
        # topk returns a tuple, whose first element are the values and the second the indexes
        closest_index = distance_matrix.topk(k, dim=2)[1]  # [B x M x K]
        closest_index = closest_index.squeeze(0)

        gathered_projected_feat = point_feat[:, closest_index]  # [B x M x K x C]

        return gathered_projected_feat

    def forward(self, pc, gt_grasps=None, gt_sampling=None):
        """
        @param gt_sampling:
        @param pc: the point cloud [B x N x 3]
        @param gt_grasps: the ground truth grasps [B x G x (3 + 3 + 1)]
                            contact1 + contact2 + angle
        @param gt_sampling: the ground truth for the sampling phase [B x S x 3]
        @return:
            - generated: the points generated by the sampler before the projection
            - projected: the points projected on the pc shape (soft matching)
            - matched: the projected points matched with pc points through NN policy (hard matching)
        """

        """EXTRACT FEATURES"""
        point_feat, global_feat = self.encoder(pc)  # [B x C x N] and [B x C]
        # print(f"point_feat: {point_feat.size()}")
        # print(f"global_feat: {global_feat.size()}")

        """SAMPLE POINTS"""
        # sample the first contact point
        # first_generated: generated points (global_feat -> [B x M x 3])
        # first_sampled: projected points if is_training else matched (hard sampling at test time)
        first_generated, first_sampled = self.sampler(pc, global_feat)  # [B x M x 3], [B x M x 3]

        """GENERATE FEATURES FOR SAMPLED POINTS"""
        # need to extract features for each projected point obtained from the sampler
        # get the K (sample_group_size) closest point in the pc and their relative features and aggregate them
        # M : number of sampled points
        # C : feature dimension
        first_contact_feat = self.compute_neigh_features(pc, first_sampled, point_feat, self.neigh_size)  # [B x M x K x C]

        """GENERATE THE FINAL GRASP"""
        first_contact = first_sampled
        second_contact, angle = self.grasp_predictor(first_contact_feat)
        predicted_grasps = torch.cat((first_contact, second_contact, angle), dim=-1)  # [B x M x 7]

        # the grasp classifier works on different data at training or test time:
        #   - training: use gt_grasp data
        #   - test: use grasp generated data (the ones extracted before)
        """"EXTRACT FEATURES FOR THE SECOND CONTACT + ANGLE"""
        if self.training:
            assert gt_grasps is not None, "Need ground truth grasps at training time"
            gt_first_contact = gt_grasps[:, :, 0:3]
            gt_second_contact = gt_grasps[:, :, 3:6]
            gt_angle = gt_grasps[:, :, 6]

            first_contact_feat = self.compute_neigh_features(pc, gt_first_contact, point_feat, self.neigh_size)
            first_contact = gt_first_contact
            second_contact = gt_second_contact
            angle = gt_angle.unsqueeze(2)

        if self.use_all_grasp_info:
            to_expand = torch.cat((first_contact, second_contact, angle), dim=-1)
        else:
            to_expand = torch.cat((second_contact, angle), dim=-1)

        # to combine with gathered, need to expand to [B x M x K x C]
        second_contact_angle_feat = self.contact_angle_feat_extractor(to_expand)
        second_contact_angle_feat = second_contact_angle_feat.unsqueeze(2).expand(-1, -1, self.neigh_size, -1)

        """"CLASSIFY THE GRASP"""
        # grasp feat is the combination of gathered_feat from the first contact point neighborhood
        # and feat extracted from the second contact point + the angle
        if self.use_contact_angle_feat:
            # if second+angle feat are extracted, then features are summed up
            grasp_feat = first_contact_feat + second_contact_angle_feat
        else:
            # first contact features are concatenated with the coordinates of the second point and the angle
            grasp_feat = torch.cat((first_contact_feat, second_contact_angle_feat), dim=-1)

        grasp_scores = self.grasp_classifier(grasp_feat)  # [B x M x 1]

        return (first_generated, first_sampled), predicted_grasps, grasp_scores
