import torch
import torch.nn.functional as F


# pytorch version
# gripperX: torch tensor of size [bs, 3]
# return: torch tensor of size [bs, 3]
def getUniqueGripperX2(gripperX):
    negGripperX = torch.clone(torch.neg(gripperX))
    gripperXwithZgt0Flag = gripperX[:, 2].gt(0)
    gripperXwithZgt0Flag = gripperXwithZgt0Flag[:, None].expand_as(gripperX)
    gripperX = gripperX.where(gripperXwithZgt0Flag, negGripperX)
    gripperX = F.normalize(gripperX, dim=1)
    gripperXOffset = torch.clone(gripperX + 0.001)
    gripperXNotPerpendicularToXoYFlag = torch.abs(torch.mul(gripperX[:, 0], gripperX[:, 1])).gt(1e-6)
    gripperXNotPerpendicularToXoYFlag = gripperXNotPerpendicularToXoYFlag[:, None].expand_as(gripperX)
    gripperX = gripperX.where(gripperXNotPerpendicularToXoYFlag, gripperXOffset)
    gripperX = F.normalize(gripperX, dim=1)
    return gripperX


def getOrientation2(contact1, center, cosAngle):
    gripperX = contact1 - center
    gripperX = getUniqueGripperX2(gripperX)
    # print('gripperX\t', gripperX)

    zAxis = torch.tensor([0, 0, 1], dtype=torch.float).to(center.device)
    zAxis = zAxis.expand_as(gripperX)
    tangentY = zAxis.cross(gripperX)
    tangentY = F.normalize(tangentY, dim=1)
    # print('tangentY\t', tangentY)

    tangentZ = gripperX.cross(tangentY)
    # print('tangentZ\t', tangentZ)
    sinAngle = torch.sqrt(1 - cosAngle * cosAngle)

    gripperZ = cosAngle[:, None].expand_as(tangentY) * tangentY + sinAngle[:, None].expand_as(tangentY) * tangentZ
    gripperZ = F.normalize(gripperZ, dim=1)
    gripperY = gripperZ.cross(gripperX)
    gripperY = F.normalize(gripperY, dim=1)
    return torch.transpose(torch.stack([gripperX, gripperY, gripperZ], dim=1), dim0=-1, dim1=-2)


# pytorch version
# matrix : torch tensor of size [bs, 9]
# return quaternion: torch tensor of size [bs, 4]
def matrix2quaternion2(matrix):
    fourWSquaredMinus1 = matrix[:, 0] + matrix[:, 4] + matrix[:, 8]
    fourXSquaredMinus1 = matrix[:, 0] - matrix[:, 4] - matrix[:, 8]
    fourYSquaredMinus1 = matrix[:, 4] - matrix[:, 0] - matrix[:, 8]
    fourZSquaredMinus1 = matrix[:, 8] - matrix[:, 0] - matrix[:, 4]
    temp = torch.stack([fourWSquaredMinus1, fourXSquaredMinus1, fourYSquaredMinus1, fourZSquaredMinus1], dim=1)
    fourBiggestSquaredMinus1, biggestIndex = torch.max(temp, dim=1)
    biggestVal = torch.sqrt(fourBiggestSquaredMinus1 + 1) * 0.5
    mult = 0.25 / biggestVal
    temp0 = biggestVal
    temp1 = (matrix[:, 7] - matrix[:, 5]) * mult
    temp2 = (matrix[:, 2] - matrix[:, 6]) * mult
    temp3 = (matrix[:, 3] - matrix[:, 1]) * mult
    temp4 = (matrix[:, 7] + matrix[:, 5]) * mult
    temp5 = (matrix[:, 2] + matrix[:, 6]) * mult
    temp6 = (matrix[:, 3] + matrix[:, 1]) * mult

    quaternion = torch.empty(size=[matrix.shape[0], 4], dtype=torch.float)
    quaternionBiggestIndex0 = torch.stack([temp0, temp1, temp2, temp3], dim=1)
    quaternionBiggestIndex1 = torch.stack([temp1, temp0, temp6, temp5], dim=1)
    quaternionBiggestIndex2 = torch.stack([temp2, temp6, temp0, temp4], dim=1)
    quaternionBiggestIndex3 = torch.stack([temp3, temp5, temp4, temp0], dim=1)

    # biggestIndex0Map = torch.ne(biggestIndex, 0)
    # biggestIndex0Map = biggestIndex0Map[:, None].expand_as(quaternion)
    biggestIndex1Map = torch.ne(biggestIndex, 1)
    biggestIndex1Map = biggestIndex1Map[:, None].expand_as(quaternion)
    biggestIndex2Map = torch.ne(biggestIndex, 2)
    biggestIndex2Map = biggestIndex2Map[:, None].expand_as(quaternion)
    biggestIndex3Map = torch.ne(biggestIndex, 3)
    biggestIndex3Map = biggestIndex3Map[:, None].expand_as(quaternion)

    quaternion = quaternionBiggestIndex0
    quaternion = quaternion.where(biggestIndex1Map, quaternionBiggestIndex1)
    quaternion = quaternion.where(biggestIndex2Map, quaternionBiggestIndex2)
    quaternion = quaternion.where(biggestIndex3Map, quaternionBiggestIndex3)
    return quaternion


# original in  GPNet/test.py
def get_7dof_poses(contact1, contact2, angle):
    contact1, contact2, angle = contact1.squeeze(0), contact2.squeeze(0), angle.squeeze(0)

    centers = (contact1 + contact2) / 2
    widths = torch.sqrt(((contact1 - contact2) ** 2).sum(1))
    matrix = getOrientation2(contact1, centers, angle).contiguous().view(-1, 9)
    quaternions = matrix2quaternion2(matrix)

    return centers.unsqueeze(0), widths.view(-1, centers.shape[0], 1), quaternions.unsqueeze(0)


def reparametrize_grasps(grasps, with_width=False, gpnet_scale=False):
    """
    Turn grasp parametrization from (c1, c2, cos_angle) to (center, width [?], quaternion)
    @param grasps: (c1, c2, cos_angle) [B x G x 7]
    @param with_width: whether to include the width in the returned parametrization
    @param gpnet_scale: whether to scale grasp back to GPNet data scale (0.22x0.22x0.22 grid)
    @return: grasps: (center, width [?], quaternion) [B x G x 7/8]
    """

    contact1 = grasps[:, :, :3]
    contact2 = grasps[:, :, 3:6]

    if gpnet_scale:
        contact1 = contact1 * torch.tensor([0.22 / 2, 0.22 / 2, 0.22]).to(contact1.device)
        contact2 = contact2 * torch.tensor([0.22 / 2, 0.22 / 2, 0.22]).to(contact2.device)
    angle = grasps[:, :, 6]

    centers, widths, quaternions = get_7dof_poses(contact1, contact2, angle)

    if with_width:
        rep_grasps = torch.cat((centers, quaternions, widths), dim=-1)
    else:
        rep_grasps = torch.cat((centers, quaternions), dim=-1)

    return rep_grasps


def cal_accuracy(gt_label, pred_score, th=0.5, recall=False, posi_num=None):
    pred_label = (pred_score > th).float()
    correct = (gt_label == pred_label).float().view(-1)
    acc = correct.sum() / correct.size(0)
    if not recall:
        return acc.item()
    else:
        posi_correct = correct * gt_label
        if posi_num is None:
            recall = posi_correct.sum() / gt_label.sum()
        else:
            recall = posi_correct.sum() / (posi_num + 1e-8)
        return acc.item(), recall.item()
