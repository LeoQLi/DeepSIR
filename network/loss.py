import os, logging
import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.math import se3_torch
from network.matchnet import square_distance_V2

_EPS = 1e-16
_logger = logging.getLogger(os.path.basename(__file__))


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    """
    return torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    diffs = all_diffs(a, b)
    if metric == 'sqeuclidean':
        return torch.sum(diffs ** 2, dim=-1)
    elif metric == 'euclidean':
        return torch.sqrt(torch.sum(diffs ** 2, dim=-1) + _EPS)
    elif metric == 'cityblock':
        return torch.sum(torch.abs(diffs), dim=-1)
    else:
        raise NotImplementedError(
            'The following metric is not implemented by `cdist` yet: {}'.format(metric))


def desc_pair_loss(anc_descriptors, pos_descriptors, neg_descriptors, anc_sigmas,
            triple_loss_gamma=0.5, sigma_max=3):
    """
    Positive: For each keypoint A in src scan, find keypoint B in positive scan, with the most similar descriptor, minimize the distance
    Negative: For each keypoint B in src scan, find keypoint B in negative scan, with the most similar descriptor, maximize the distance
    the 'minimize / maximize' is via triplet loss

    :param anc_descriptors: BxCxM
    :param pos_descriptors: BxCxM
    :param neg_descriptors: BxCxM
    :param anc_sigmas: BxM
    :return:
    """
    anc_descriptors_Mx1 = anc_descriptors.unsqueeze(3)  # BxCxMx1
    pos_descriptors_1xM = pos_descriptors.unsqueeze(2)  # BxCx1xM
    neg_descriptors_1xM = neg_descriptors.unsqueeze(2)  # BxCx1xM

    # positive loss
    anc_pos_diff = torch.norm(anc_descriptors_Mx1 - pos_descriptors_1xM, dim=1, keepdim=False)  # BxCxMxM -> BxMxM
    min_anc_pos_diff, _ = torch.min(anc_pos_diff, dim=2, keepdim=False)  # BxM

    # negative loss
    anc_neg_diff = torch.norm(anc_descriptors_Mx1 - neg_descriptors_1xM, dim=1, keepdim=False)  # BxCxMxM -> BxMxM
    min_anc_neg_diff, _ = torch.min(anc_neg_diff, dim=2, keepdim=False)  # BxM

    # triplet loss
    before_clamp_loss = min_anc_pos_diff - min_anc_neg_diff + triple_loss_gamma  # BxM
    active_percentage = torch.mean((before_clamp_loss > 0).float(), dim=1, keepdim=False)  # B

    # 1. without sigmas
    # loss = torch.clamp(before_clamp_loss, min=0)

    # 2. with sigmas, use only the anc_sigmas
    # sigma is the uncertainly, smaller->more important. Turn it into weight by alpha - sigma
    anc_weights = torch.clamp(sigma_max - anc_sigmas, min=0)  # BxM
    # normalize to be mean of 1
    anc_weights_mean = torch.mean(anc_weights, dim=1, keepdim=True)  # Bx1
    anc_weights = (anc_weights / anc_weights_mean).detach()  # BxM
    loss = anc_weights * torch.clamp(before_clamp_loss, min=0)  # BxM

    return loss, active_percentage


def desc_loss(anc_descriptors, pos_descriptors, anc_keypoints=None, pos_keypoints=None, keypoint_diff=None,
            anc_sigmas=None, thres_radius=0.1, triple_loss_gamma=0.5, sigma_max=3,
            anc_pc=None, pos_pc=None):
    """ Triplet loss of 'Learning Compact Geometric Features'
    positive pair: for each keypoint in anc, find keypoints in pos with d < threshold, random sample one
    negative pair: for each keypoint in anc, find keypoitns in pos with the closest but d > threshold.
                    This is different with original CGF. If the detector works perfectly,
                    it may be difficult to find keypoints with threshold < d < 2*threshold

    :param anc_descriptors: BxCxM
    :param pos_descriptors: BxCxM
    :param anc_keypoints: Bx3xM
    :param pos_keypoints: Bx3xM, already transformed to the coordinate of anc
    :param keypoint_diff: BxMxM, keypoint difference matrix
    :param anc_sigmas: BxM, score of the anc_keypoints
    :param thres_radius: CGF Threshold to determine positive match of points
    :param triple_loss_gamma: parameter of triple loss
    :param sigma_max: Threshold of distance between cloesest keypoint pairs,
                        large distance is considered to be mis-matched.
    :param anc_pc: Bx3xM
    :param pos_pc: Bx3xM, already transformed to the coordinate of anc
    :return:
    """
    B, _, M = anc_descriptors.size()
    device = anc_descriptors.device

    if keypoint_diff is None and anc_keypoints is not None:
        anc_keypoints = anc_keypoints.unsqueeze(3)  # Bx3xMx1
        pos_keypoints = pos_keypoints.unsqueeze(2)  # Bx3x1xM
        keypoint_diff = torch.norm(anc_keypoints - pos_keypoints, dim=1, keepdim=False)  # Bx3xMxM -> BxMxM

    anc_descriptors = anc_descriptors.unsqueeze(3)  # BxCxMx1
    pos_descriptors = pos_descriptors.unsqueeze(2)  # BxCx1xM
    descriptor_diff = torch.norm(anc_descriptors - pos_descriptors, dim=1, keepdim=False)  # BxCxMxM -> BxMxM

    ######### 1. positive pair #########
    positive_mask_BMM = keypoint_diff <= thres_radius  # BxMxM
    positive_mask_BM = torch.max(positive_mask_BMM, dim=2, keepdim=False)[0]  # TODO bug *keypoint_diff BxM, to indicate whether there is a match

    ## 1.1 random sample a match
    random_mat = torch.rand((B, M, M), dtype=torch.float32, device=device, requires_grad=False)  # [0, 1)
    random_mat_nearby_mask = positive_mask_BMM.float() * random_mat  # BxMxM
    nearby_idx = torch.max(random_mat_nearby_mask, dim=2, keepdim=True)[1]  # BxMx1
    positive_dist = torch.gather(descriptor_diff, dim=2, index=nearby_idx).squeeze(2)  # BxMx1 -> BxM

    ######### 2. negative pair #########
    ## 2.1 for each keypoint in anc, find keypoitns in pos with the closest but d>threshold.
    augmented_keypoint_diff = keypoint_diff + positive_mask_BMM.float() * 1000  # BxMxM
    far_close_idx = torch.min(augmented_keypoint_diff, dim=2, keepdim=True)[1]  # BxMx1
    far_close_dist = torch.gather(descriptor_diff, dim=2, index=far_close_idx).squeeze(2)  # BxMx1 -> BxM

    ## 2.2 randomly choose a keypoint with d>threshold
    outside_radius_mask = keypoint_diff > thres_radius  # BxMxM
    random_mat_outside = torch.rand((B, M, M), dtype=torch.float32, device=device, requires_grad=False)  # [0, 1)
    random_mat_outside_masked = random_mat_outside * outside_radius_mask.float()  # BxMxM
    outside_idx = torch.max(random_mat_outside_masked, dim=2, keepdim=True)[1]  # BxMx1
    outside_random_dist = torch.gather(descriptor_diff, dim=2, index=outside_idx).squeeze(2)  # BxM

    ## 2.3 assemble a negative_dist by combining far_close_dist & outside_random_dist
    random_mat_selection = torch.rand((B, M), dtype=torch.float32, device=device, requires_grad=False)  # BxM
    selection_mat = (random_mat_selection < 0.5).float()  # BxM
    negative_dist = selection_mat * far_close_dist + (1 - selection_mat) * outside_random_dist  # BxM

    ######### triplet loss #########
    # consider only the matched keypoints, so a re-scale is necessary
    scaling = (M / (torch.sum(positive_mask_BM.float(), dim=1, keepdim=False) + 1)).detach()  # B
    before_clamp_loss = (positive_dist - negative_dist + triple_loss_gamma) * positive_mask_BM.float()  # BxM
    active_percentage = torch.sum((before_clamp_loss > 1e-5).float(), dim=1, keepdim=False) /  \
                        (torch.sum(positive_mask_BM.float(), dim=1, keepdim=False) + 1)  # B
    active_percentage = torch.mean(active_percentage)

    # with sigmas, use only the anc_sigmas
    # sigma is the uncertainly, smaller->more important. Turn it into weight by alpha - sigma
    # anc_weights = torch.clamp(sigma_max - anc_sigmas, min=0)  # BxM
    anc_weights = anc_sigmas       # TODO  不应该加score，特征都应该相似
    # normalize to be mean of 1
    anc_weights_mean = torch.mean(anc_weights, dim=1, keepdim=True)  # Bx1
    anc_weights = (anc_weights / anc_weights_mean).detach()  # BxM

    loss = anc_weights * torch.clamp(before_clamp_loss, min=0) * scaling.unsqueeze(1)  # BxM
    loss = torch.mean(loss)

    # if debug:
        # anc_pc_batch_np = torch.transpose(anc_pc, 1, 2).detach().cpu().numpy()
        # pos_pc_batch_np = torch.transpose(pos_pc, 1, 2).detach().cpu().numpy()
        # anc_keypoints_batch_np = torch.transpose(anc_keypoints, 1, 2).detach().cpu().numpy()
        # pos_keypoints_batch_np = torch.transpose(pos_keypoints, 1, 2).detach().cpu().numpy()
        # for b in range(B):
        #     print('---\nscaling %f' % scaling[b].item())
        #     anc_pc_np = anc_pc_batch_np[b]  # Nx3
        #     pos_pc_np = pos_pc_batch_np[b]  # Nx3
        #     anc_keypoints_np = anc_keypoints_batch_np[b]  # Mx3
        #     pos_keypoints_np = pos_keypoints_batch_np[b]  # Mx3

        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     ax.scatter(pos_pc_np[:, 0].tolist(),
        #             pos_pc_np[:, 1].tolist(),
        #             pos_pc_np[:, 2].tolist(),
        #             s=5, c=[1, 0.8, 0.8])
        #     ax.scatter(anc_pc_np[:, 0].tolist(),
        #             anc_pc_np[:, 1].tolist(),
        #             anc_pc_np[:, 2].tolist(),
        #             s=5, c=[0.8, 0.8, 1])

        #     # plot the first matched keypoint
        #     for m in range(M):
        #         if positive_mask_BM[b, m] == 1:
        #             ax.scatter(anc_keypoints_np[m, 0],
        #                     anc_keypoints_np[m, 1],
        #                     anc_keypoints_np[m, 2],
        #                     s=30, c=[1, 0, 0])
        #             ax.scatter(pos_keypoints_np[nearby_idx[b, m].item(), 0],
        #                     pos_keypoints_np[nearby_idx[b, m].item(), 1],
        #                     pos_keypoints_np[nearby_idx[b, m].item(), 2],
        #                     s=30, c=[0, 0, 1])
        #             ax.scatter(pos_keypoints_np[far_close_idx[b, m].item(), 0],
        #                     pos_keypoints_np[far_close_idx[b, m].item(), 1],
        #                     pos_keypoints_np[far_close_idx[b, m].item(), 2],
        #                     s=30, c=[0, 1, 0])
        #             ax.scatter(pos_keypoints_np[outside_idx[b, m].item(), 0],
        #                     pos_keypoints_np[outside_idx[b, m].item(), 1],
        #                     pos_keypoints_np[outside_idx[b, m].item(), 2],
        #                     s=30, c=[0.4, 0, 0.8])
        #             print('matched dist: %f' % np.linalg.norm(anc_keypoints_np[m] - pos_keypoints_np[nearby_idx[b, m].item()]))
        #             print('far-close dist: %f' % np.linalg.norm(anc_keypoints_np[m] - pos_keypoints_np[far_close_idx[b, m].item()]))
        #             print('outside random dist: %f' % np.linalg.norm(anc_keypoints_np[m] - pos_keypoints_np[outside_idx[b, m].item()]))
        #             break

        #     vis_tools.axisEqual3D(ax)
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     ax.set_zlabel('z')
        #     plt.show()

    return loss, active_percentage


def batch_rotation_error(rots1, rots2, eps=1e-16):
    """ Compute rotation error (in radian): arccos((tr(R_1^T R_2) - 1) / 2)
    rots1: [B, 3, 3] or [B, 9]
    rots1: [B, 3, 3] or [B, 9]
    return: [B]
    """
    assert len(rots1) == len(rots2)
    trace_r1Tr2 = (rots1.reshape(-1, 9) * rots2.reshape(-1, 9)).sum(1)
    side = (trace_r1Tr2 - 1) / 2
    return torch.acos(torch.clamp(side, min=-1 + eps, max=1 - eps))


def batch_translation_error(trans1, trans2):
    r""" Compute translation error
    trans1: [B, 3]
    trans2: [B, 3]
    return: [B]
    """
    assert len(trans1) == len(trans2)
    return torch.norm(trans1 - trans2, p=2, dim=1, keepdim=False)


def pose_error(gt_transforms, pred_transforms, eps=1e-16):
    """ Compute translation error and rotation error (in degree)
    gt: [B, 3, 4]
    pred: [B, 3, 4]
    return: [B]
    """
    assert len(gt_transforms) == len(pred_transforms)
    concatenated = se3_torch.concatenate(se3_torch.inverse(gt_transforms), pred_transforms)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1 + eps, max=1 - eps)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)
    return residual_rotdeg, residual_transmag


def _hash(arr, M=None):
    """ hash = arr[:, 0] * X + arr[:, 1] * X + arr[:, 2] * X + ...
    """
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M**d
        else:
            hash_vec += arr[d] * M**d
    return hash_vec


class ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=0.1, neg_margin=1.4, metric='euclidean', thres_radius=0.1):
        super(ContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.metric = metric
        self.thres_radius = thres_radius

    def forward(self, anchor, positive, dist_pc):
        pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
        dist = cdist(anchor, positive, metric=self.metric)
        dist_pc = np.eye(dist_pc.shape[0]) * 10 + dist_pc.detach().cpu().numpy()
        add_matrix = torch.zeros_like(dist)
        add_matrix[np.where(dist_pc < self.thres_radius)] += 10
        dist = dist + add_matrix
        return self.calculate_loss(dist, pids)


    def calculate_loss(self, dists, pids):
        """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

        Args:
            dists (2D tensor): A square all-to-all distance matrix as given by cdist.
            pids (1D tensor): The identities of the entries in `batch`, shape (B,).
                This can be of any type that can be compared, thus also a string.
            margin: The value of the margin if a number, alternatively the string
                'soft' for using the soft-margin formulation, or `None` for not
                using a margin at all.

        Returns:
            A 1D tensor of shape (B,) containing the loss value for each sample.
        """
        # generate the mask that mask[i, j] reprensent whether i th and j th are from the same identity.
        # torch.equal is to check whether two tensors have the same size and elements
        # torch.eq is to computes element-wise equality
        same_identity_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
        # negative_mask = np.logical_not(same_identity_mask)

        # dists * same_identity_mask get the distance of each valid anchor-positive pair.
        furthest_positive, _ = torch.max(dists * same_identity_mask.float(), dim=1)
        # here we use "dists +  10000*same_identity_mask" to avoid the anchor-positive pair been selected.
        closest_negative, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=1)
        # closest_negative_row, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=0)
        # closest_negative = torch.min(closest_negative_col, closest_negative_row)
        diff = furthest_positive - closest_negative
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]
        loss = torch.max(furthest_positive - self.pos_margin, torch.zeros_like(diff)) + torch.max(self.neg_margin - closest_negative, torch.zeros_like(diff))

        average_negative = (torch.sum(dists, dim=-1) - furthest_positive) / (dists.shape[0] - 1)

        return torch.mean(loss), accuracy, furthest_positive.tolist(), average_negative.tolist(), 0, dists


class ChamferLoss_Single(nn.Module):
    def __init__(self):
        super(ChamferLoss_Single, self).__init__()

    def forward(self, pc_src, pc_dst):
        '''
        :param pc_src: [B, 3, M]
        :param pc_dst: [B, 3, N]
        :return:
        '''
        pc_src = pc_src[:, :, :, None]   # [B, 3, M, 1]
        pc_dst = pc_dst[:, :, None, :]   # [B, 3, 1, N]
        diff = torch.norm(pc_src - pc_dst, dim=1, keepdim=False)  # [B, M, N]

        # pc_src vs selected pc_dst, M
        diff = torch.min(diff, dim=2, keepdim=False)[0]  # BxM

        return torch.mean(diff)


class KeypointOnPCLoss(nn.Module):
    def __init__(self):
        super(KeypointOnPCLoss, self).__init__()

        self.single_side_chamfer = SingleSideChamferLoss()
        self.keypoint_on_surface = PointOnSurfaceLoss()

    def forward(self, keypoint, pc_tgt, sn=None):
        '''
        :param keypoint: Bx3xM
        :param pc_tgt: Bx3xN
        :return:
        '''
        if sn is None:
            loss = self.single_side_chamfer(keypoint, pc_tgt)
        else:
            loss = self.keypoint_on_surface(keypoint, pc_tgt, sn)

        return torch.mean(loss)


class PointOnSurfaceLoss(nn.Module):
    def __init__(self):
        super(PointOnSurfaceLoss, self).__init__()

    def forward(self, kp, pc, sn):
        '''
        :param kp: Bx3xM
        :param pc: Bx3xN
        :param sn: Bx3xN
        :return:
        '''
        B, _, M = kp.size()

        kp = kp[:, :, :, None]   # [B, 3, M, 1]
        pc = pc[:, :, None, :]   # [B, 3, 1, N]
        diff = torch.norm(kp - pc, dim=1, keepdim=False)  # [B, M, N]

        # keypoint vs selected pc, M
        keypoint_pc_min_dist, keypoint_pc_min_I = torch.min(diff, dim=2, keepdim=False)  # BxM
        pc_selected = torch.gather(pc, dim=2, index=keypoint_pc_min_I.unsqueeze(1).expand(B, 3, M))  # Bx3xM
        sn_selected = torch.gather(sn, dim=2, index=keypoint_pc_min_I.unsqueeze(1).expand(B, 3, M))  # Bx3xM

        # keypoint on surface loss
        keypoint_minus_pc = kp - pc_selected  # Bx3xM
        keypoint_minus_pc_norm = torch.norm(keypoint_minus_pc, dim=1, keepdim=True)  # Bx1xM
        keypoint_minus_pc_normalized = keypoint_minus_pc / (keypoint_minus_pc_norm + 1e-7)  # Bx3xM

        sn_selected = sn_selected.permute(0, 2, 1)  # BxMx3
        keypoint_minus_pc_normalized = keypoint_minus_pc_normalized.permute(0, 2, 1)  # BxMx3

        loss = torch.matmul(sn_selected.unsqueeze(2), keypoint_minus_pc_normalized.unsqueeze(3)) ** 2  # BxMx1x3 * BxMx3x1 -> BxMx1x1 -> 1

        return loss


class ChamferLoss(nn.Module):
    def __init__(self,):
        super(ChamferLoss, self).__init__()
        _logger.info('Chamfer Loss.')

    def forward(self, pc_src=None, pc_dst=None, sigma_src=None, sigma_dst=None, diff=None):
        """
        :param pc_src: [B, 3, M], be transformed to the coordinate of dst
        :param pc_dst: [B, 3, N]
        :param sigma_src: [B, M], score of the src_keypoints
        :param sigma_dst: [B, N]
        :return:
        """
        if diff is None and pc_src is not None:
            pc_src = pc_src.unsqueeze(3)       # [B, 3, M, 1]
            pc_dst = pc_dst.unsqueeze(2)       # [B, 3, 1, N]

            # the gradient of norm is set to 0 at zero-input. There is no need to use custom norm anymore.
            diff = torch.norm(pc_src - pc_dst, dim=1, keepdim=False)  # BxMxN

        if sigma_src is None or sigma_dst is None:
            # pc_src vs selected pc_dst, M
            src_dst_min_dist = torch.min(diff, dim=2, keepdim=False)[0]  # BxM
            forward_loss = src_dst_min_dist.mean()

            # pc_dst vs selected pc_src, N
            dst_src_min_dist = torch.min(diff, dim=1, keepdim=False)[0]  # BxN
            backward_loss = dst_src_min_dist.mean()

            # chamfer_pure = forward_loss + backward_loss
            # chamfer_weighted = chamfer_pure
        else:
            # pc_src vs selected pc_dst, M
            src_dst_min_dist, src_dst_I = torch.min(diff, dim=2, keepdim=False)   # BxM, BxM
            selected_sigma_dst = torch.gather(sigma_dst, dim=1, index=src_dst_I)  # BxN -> BxM 最小距离对应点的sigma
            sigma_src_dst = (sigma_src + selected_sigma_dst) / 2
            # forward_loss = (torch.log(sigma_src_dst) + src_dst_min_dist / sigma_src_dst).mean()  # TODO
            forward_loss = (src_dst_min_dist * sigma_src_dst).mean()

            # pc_dst vs selected pc_src, N
            dst_src_min_dist, dst_src_I = torch.min(diff, dim=1, keepdim=False)  # BxN, BxN
            selected_sigma_src = torch.gather(sigma_src, dim=1, index=dst_src_I)  # BxM -> BxN
            sigma_dst_src = (sigma_dst + selected_sigma_src) / 2
            # backward_loss = (torch.log(sigma_dst_src) + dst_src_min_dist / sigma_dst_src).mean()
            backward_loss = (dst_src_min_dist * sigma_dst_src).mean()

            # # loss that do not involve in optimization
            # chamfer_pure = (src_dst_min_dist.mean() + dst_src_min_dist.mean()).detach()
            # weight_src_dst = sigma_src_dst / torch.mean(sigma_src_dst)
            # weight_dst_src = sigma_dst_src / torch.mean(sigma_dst_src)
            # chamfer_weighted = ((weight_src_dst * src_dst_min_dist).mean() +
            #                     (weight_dst_src * dst_src_min_dist).mean()).detach()

        # return forward_loss + backward_loss, chamfer_pure, chamfer_weighted, diff
        return forward_loss + backward_loss


class CircleLoss(nn.Module):
    def __init__(self, m=0.1, log_scale=10, thres_radius=0.0):
        """ Computes the circle loss proposed in https://arxiv.org/abs/2002.10857.
            pos_margin, neg_margin (float): the margin for contrastive loss
            O_p = 1+m, O_n = -m, ∆_p = 1-m, ∆_n = m
        """
        super(CircleLoss, self).__init__()
        _logger.info('Circle Loss.')

        self.log_scale    = log_scale  # gamma
        self.pos_margin   = 0.1
        self.neg_margin   = 1.4
        self.pos_optimal  = 0.1
        self.neg_optimal  = 1.4
        self.thres_radius = thres_radius
        assert thres_radius > 0

    def forward(self, anc_feat, pos_feat, anc_pc=None, pos_pc=None, anc_score=None, pos_score=None, dist_pc=None):
        """ Note that the implementation is slightly different:
                use distance instead of cosine similarity && set the optimal and margin to be the same.
            :param anc_feat: [B, C, N1], feature of anchor points
            :param pos_feat: [B, C, N2], feature of positive points
            :param anc_pc: [B, 3, N1]
            :param pos_pc: [B, 3, N2], already transformed to the coordinate of anc
            :param dist_pc: [B, N1, N2], point difference matrix
            :param anc_score: [B, N1], score of the anc_keypoints
            :param pos_score: [B, N2], score of the pos_keypoints
        """
        eps = 1e5
        # normalization with sum to be 1
        anc_score = anc_score / torch.sum(anc_score, dim=1, keepdim=True)  # [B, N]
        # pos_score = pos_score / torch.sum(pos_score, dim=1, keepdim=True)  # [B, N]

        if dist_pc is None and anc_pc is not None:
            anc_pc = anc_pc.unsqueeze(3)                                   # [B, 3, N1, 1]
            pos_pc = pos_pc.unsqueeze(2)                                   # [B, 3, 1, N2]
            dist_pc = torch.norm(anc_pc - pos_pc, dim=1, keepdim=False)    # [B, N1, N2]
            # dist_pc = torch.sqrt(square_distance_V2(anc_pc, pos_pc) + _EPS)

        # faster way
        # anc_feat = anc_feat.unsqueeze(3)                                   # [B, C, N1, 1]
        # pos_feat = pos_feat.unsqueeze(2)                                   # [B, C, 1, N2]
        # dist_feat = torch.norm(anc_feat - pos_feat, dim=1, keepdim=False)  # [B, N1, N2]
        # TODO slower but low memory footprint
        dist_feat = torch.sqrt(square_distance_V2(anc_feat, pos_feat) + _EPS)

        #===============================================================

        false_negative = dist_pc < self.thres_radius                                    # [B, N1, N2]
        dist_min = torch.min(dist_pc * false_negative.float(), dim=2, keepdim=True)[0]  # [B, N1, 1]
        pos_mask = torch.eq(dist_pc, dist_min.repeat(1, 1, dist_pc.shape[-1]))          # [B, N1, N2]
        neg_mask = torch.logical_not(pos_mask | false_negative)                         # [B, N1, N2]

        # add eps to eliminate the effect of positve and false negative fairs
        pos = dist_feat - eps * neg_mask.float()                                 # [B, N1, N2]  neg都会很小，pos不变
        pos_weight = (pos - self.pos_optimal).detach()                           # [B, N1, N2]
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)         # [B, N1, N2]  >0
        pos_weighted = self.log_scale * (pos - self.pos_margin) * pos_weight
        lse_positive = torch.logsumexp(pos_weighted, dim=-1, keepdim=False)      # [B, N1]  neg趋于0

        neg = dist_feat + eps * (~neg_mask).float()                              # [B, N1, N2]  非neg都会很大，neg不变
        neg_weight =  (self.neg_optimal - neg).detach()                          # [B, N1, N2]
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)         # [B, N1, N2]  >0
        neg_weighted = self.log_scale * (self.neg_margin - neg) * neg_weight     # [B, N1, N2]
        lse_negative_row = torch.logsumexp(neg_weighted, dim=-1, keepdim=False)  # [B, N1]
        lse_negative_col = torch.logsumexp(neg_weighted, dim=-2, keepdim=False)  # [B, N2]

        ### triplet loss + contrastive loss
        # lse_positive = lse_positive * anc_score
        loss_col = F.softplus(lse_positive + lse_negative_row) / self.log_scale   # [B, N1]
        loss_row = F.softplus(lse_positive + lse_negative_col) / self.log_scale   # [B, N1]
        loss_feat = loss_col + loss_row                                           # [B, N1]
        loss_feat = torch.mean(loss_feat)

        #===============================================================

        ### compute accuracy
        # dist_feat * pos_mask get the distance of each valid anchor-positive pair.
        furthest_positive = torch.max(dist_feat * pos_mask.float(), dim=-1)[0]        # [B, N1]

        # here we use "dist_feat +  10000*pos_mask" to avoid the anchor-positive pair been selected.
        closest_negative = torch.min(dist_feat + eps * pos_mask.float(), dim=-1)[0]   # [B, N1]
        # closest_negative_row = torch.min(dist_feat + eps * pos_mask.float(), dim=0)[0]
        # closest_negative = torch.min(closest_negative_col, closest_negative_row)

        diff = furthest_positive - closest_negative          # [B, N1]
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[1]  # [B, N1]  特征能够正确匹配的比例using the nearest neighbor search

        # average_negative = (torch.sum(dist_feat, dim=-1) - furthest_positive) / (dist_feat.shape[1] - 1)   # [B, N1] #TODO ???

        #===============================================================

        # loss_det = diff * (anc_score + pos_score)       # TODO [B, N1]  diff<0, score应该大，diff>0，score应该小
        loss_det = diff * anc_score
        loss_det  = torch.mean(loss_det)

        if False:
            # dist_pc = cdist(sel_P_src, sel_P_src)
            # dist_feat = cdist(anchor, positive, metric='euclidean')   # (B1, B2)

            ### 正负样本的mask
            # build false negative
            false_negative = dist_pc < self.thres_radius    #  [B, B] [B, N, N]

            pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)  # [B,]
            # 对角线上的为真对应，这里不对，他有作对应，我没有对应起来，只能通过点距离找最近的
            pos_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))  # [B, B] identity matrix
            neg_mask = torch.logical_not(pos_mask | false_negative)  # [B, B]  不在对角线上或dist_keypts >= thres_radius

            # add eps to eliminate the effect of positve and false negative fairs
            pos = dist_feat - eps * neg_mask.float()            # [B, B]  neg都会很小，pos不变
            pos_weight = (pos - self.pos_optimal).detach()
            pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)  # [B, B]  >0
            lse_positive = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight, dim=-1)  # [B,]

            neg = dist_feat + eps * (~neg_mask).float()        # [B, B]  非neg都会很大，neg不变
            neg_weight =  (self.neg_optimal - neg).detach()
            neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)  # [B, B]  >0
            lse_negative_row = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight, dim=-1)  # [B,]
            lse_negative_col = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight, dim=-2)  # [B,]

            # triplet loss + contrastive loss
            loss_col = F.softplus(lse_positive + lse_negative_row) / self.log_scale
            loss_row = F.softplus(lse_positive + lse_negative_col) / self.log_scale
            loss = loss_col + loss_row

            ### compute accuracy
            # pos 里最大的，neg里最小的
            # dist_feat * pos_mask get the distance of each valid anchor-positive pair.
            furthest_positive = torch.max(dist_feat * pos_mask.float(), dim=1)[0]                # [B,]

            # here we use "dist_feat +  10000*pos_mask" to avoid the anchor-positive pair been selected. ???
            closest_negative = torch.min(dist_feat + eps * pos_mask.float(), dim=1)[0]           # [B,]
            # closest_negative_row = torch.min(dist_feat + eps * pos_mask.float(), dim=0)[0]
            # closest_negative = torch.min(closest_negative_col, closest_negative_row)

            average_negative = (torch.sum(dist_feat, dim=-1) - furthest_positive) / (dist_feat.shape[0] - 1)   # [B,]
            diff = furthest_positive - closest_negative              # [B,]
            accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]

        # return loss, accuracy, furthest_positive.tolist(), average_negative.tolist(), pos_mask, dist_feat
        return loss_feat, loss_det, dist_pc, accuracy

########################################################################################

class ChamferDistLoss(nn.Module):
    def __init__(self):
        super(ChamferDistLoss, self).__init__()

        self.chamfer_loss = ChamferLoss()
        self.chamfer_loss_s = ChamferLoss_Single()

    def forward(self, data: Dict):
        """
        data: dict, point clouds [B, 3, N]
        """
        pt_src = data['pt_src'].permute(0, 2, 1).contiguous()                # [B, 3, N]
        pt_ref = data['pt_ref'].permute(0, 2, 1).contiguous()                # [B, 3, N]
        points_src_raw = data['points_src'].permute(0, 2, 1).contiguous()    # [B, 3, N]
        points_ref_raw = data['points_ref'].permute(0, 2, 1).contiguous()    # [B, 3, N]

        loss_cham = self.chamfer_loss(pc_src=pt_src, pc_dst=pt_ref)
        src_dst_min_dist = self.chamfer_loss_s(pt_src, points_src_raw)
        ref_dst_min_dist = self.chamfer_loss_s(pt_ref, points_ref_raw)

        loss = loss_cham + src_dst_min_dist + ref_dst_min_dist
        return loss


class DetDesLoss(nn.Module):
    """ Compute loss of detection and description
    """
    def __init__(self, args):
        super(DetDesLoss, self).__init__()
        _logger.info('Detection and Description Loss.')

        self.thres_radius        = args.thres_radius     # changed in data_loader
        self.det_loss_weight     = args.det_loss_weight
        self.chamfer_loss_weight = args.chamfer_loss_weight

        self.chamfer_loss   = ChamferLoss()
        self.chamfer_loss_s = ChamferLoss_Single()
        self.circle_loss    = CircleLoss(m=0.1, log_scale=10, thres_radius=self.thres_radius)

    def forward(self, data: Dict):
        feat_src     = data['feat_src']        # [B, C, N]
        feat_ref     = data['feat_ref']        # [B, C, N]
        pt_src       = data['pt_src']          # [B, 3, N]
        pt_ref       = data['pt_ref']          # [B, 3, N]
        score_src    = data['score_src']       # [B, N]
        score_ref    = data['score_ref']       # [B, N]
        transform_gt = data['transform_gt']    # [B, 3, 4]
        loss_feat = loss_det = loss_cham = 0.0

        # transform the source cloud using the GT pose
        pt_src = se3_torch.transform_V2(transform_gt, pt_src)   # [B, 3, N]

        ### circle loss of features
        loss_feat, loss_det, dist_pc, acc = self.circle_loss(feat_ref, feat_src,
                                                            pt_ref, pt_src,
                                                            score_ref, score_src)

        ### chamfer loss of points
        # loss_cham = self.chamfer_loss(sigma_src=score_src, sigma_dst=score_ref, diff=dist_pc)

        # if 'points_src' in data and 'points_ref' in data:
        #     points_src_raw = data['points_src']    # [B, 3, N]
        #     points_ref_raw = data['points_ref']    # [B, 3, N]

        #     src_dst_min_dist = self.chamfer_loss_s(pt_src, points_src_raw)
        #     ref_dst_min_dist = self.chamfer_loss_s(pt_ref, points_ref_raw)

        #     loss_cham = loss_cham + src_dst_min_dist + ref_dst_min_dist

        # loss_dic = {}
        # loss_dic['loss_feat'] = loss_feat
        # loss_dic['loss_det']  = loss_det * self.det_loss_weight
        # loss_dic['loss_cham'] = loss_cham * self.chamfer_loss_weight

        return loss_feat + loss_det * self.det_loss_weight + loss_cham * self.chamfer_loss_weight, acc


class ScanAlignmentLoss(nn.Module):
    """ Compute loss of scan alignment
    """
    def __init__(self, args):
        """
        loss_type: Registration loss type, either 'mae' (Mean absolute error) or 'mse'
        wt_inlier_loss: Weight to encourage inliers
        """
        super(ScanAlignmentLoss, self).__init__()
        _logger.info('Scan alignment Loss.')

        self.loss_type       = args.loss_type
        self.wt_ptDist_loss  = args.wt_ptDist_loss
        self.wt_inlier_loss  = args.wt_inlier_loss
        self.wt_pose_loss    = args.wt_pose_loss
        self.discount_factor = args.loss_discount_factor
        assert self.loss_type in ['mae', 'mse']

    def find_correct_correspondence(self, pos_pairs, pred_pairs, hash_seed=None, len_batch=None):
        """ whether all elements of pred are included in pos
        """
        assert len(pos_pairs) == len(pred_pairs)  # [B, N, 2]
        if hash_seed is None:
            assert len(len_batch) == len(pos_pairs)  # [B, 2]

        corrects = []
        for i in range(len(pos_pairs)):
            pos_pair, pred_pair = pos_pairs[i], pred_pairs[i]
            if isinstance(pos_pair, torch.Tensor):
                pos_pair = pos_pair.numpy()    # [N', 2]   N' > N
            if isinstance(pred_pair, torch.Tensor):
                pred_pair = pred_pair.numpy()  # [N, 2]

            if hash_seed is None:
                N0, N1 = len_batch[i]
                _hash_seed = max(N0, N1)
            else:
                _hash_seed = hash_seed       # N

            pos_keys = _hash(pos_pair, _hash_seed)    # [N]
            pred_keys = _hash(pred_pair, _hash_seed)  # [N]

            corrects.append(np.isin(pred_keys, pos_keys, assume_unique=False)) # [[N], [N], ...]

        return np.stack(corrects, axis=0)

    def forward(self, data: Dict, reduction=None):
        """
        data: Current mini-batch data
        reduction: Either 'mean' or 'none'. Use 'none' to accumulate losses outside
                    (useful for accumulating losses for entire validation dataset)
        Returns:
            loss_dic: Dict containing various fields. Total loss to be optimized is in loss_dic['total']
        """
        loss_dic = {}
        points_src     = data['pt_src']          # [B, N, 3]
        perm_matrices  = data['perm_matrices']   # [B, J, K] list of iter
        transform_pred = data['transform_pred']  # [B, 3, 4] list of iter
        transform_gt   = data['transform_gt']    # [B, 3, 4]
        num_iter = len(transform_pred)
        assert reduction in ['mean', 'none']

        ### MSE/L1 point distance loss
        if self.wt_ptDist_loss > 0:
            # transform the source cloud using the GT pose
            gt_src_transformed = se3_torch.transform(transform_gt, points_src)

            # MSE loss to the groundtruth (does not take into account possible symmetries)
            if self.loss_type == 'mse':
                criterion = nn.MSELoss(reduction=reduction)
                for i in range(num_iter):
                    pred_src_transformed = se3_torch.transform(transform_pred[i], points_src)
                    dist = criterion(pred_src_transformed, gt_src_transformed)
                    if reduction == 'mean':
                        loss_dic['mse_{}'.format(i)] = dist
                    elif reduction == 'none':
                        loss_dic['mse_{}'.format(i)] = torch.mean(dist, dim=[-1, -2])

            # L1 loss to the groundtruth (does not take into account possible symmetries)
            elif self.loss_type == 'mae':
                criterion = nn.L1Loss(reduction=reduction)
                for i in range(num_iter):
                    pred_src_transformed = se3_torch.transform(transform_pred[i], points_src)
                    dist = criterion(pred_src_transformed, gt_src_transformed)
                    if reduction == 'mean':
                        loss_dic['mae_{}'.format(i)] = dist
                    elif reduction == 'none':
                        loss_dic['mae_{}'.format(i)] = torch.mean(dist, dim=[-1, -2])
        else:
            for i in range(num_iter):
                if reduction == 'mean':
                    loss_dic['{}_{}'.format(self.loss_type, i)] = torch.tensor(0).cuda()
                elif reduction == 'none':
                    loss_dic['{}_{}'.format(self.loss_type, i)] = torch.zeros(len(points_src)).cuda()

        ### Penalize outliers
        # if self.wt_inlier_loss > 0:
            # for i in range(num_iter):
            #     ref_outliers_strength = (1.0 - torch.sum(perm_matrices[i], dim=1)) * self.wt_inlier_loss
            #     src_outliers_strength = (1.0 - torch.sum(perm_matrices[i], dim=2)) * self.wt_inlier_loss
            #     if reduction == 'mean':
            #         loss_dic['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + \
            #                                         torch.mean(src_outliers_strength)
            #     elif reduction == 'none':
            #         loss_dic['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength, dim=1) + \
            #                                         torch.mean(src_outliers_strength, dim=1)

        ### Correspondence confidence loss
        if self.wt_inlier_loss > 0 and 'pred_pairs' in data and 'matches' in data:
            pos_pairs  = data['matches']          # [B, N', 2] N' >= N, list of batch, in CPU
            pred_pairs = data['pred_pairs']       # [B, N, 2] list of iter, in CPU
            assert pred_pairs[0].shape[1] == perm_matrices[0].shape[1]

            criterion = nn.BCEWithLogitsLoss(reduction=reduction)
            for i in range(num_iter):
                is_correct = self.find_correct_correspondence(pos_pairs, pred_pairs[i], hash_seed=points_src.shape[1])
                gt_labels = torch.from_numpy(is_correct).float()   # [B, N]

                pred_logits = perm_matrices[i].cpu()
                dist = (criterion(pred_logits, gt_labels) * self.wt_inlier_loss).cuda()
                if reduction == 'mean':
                    loss_dic['outlier_{}'.format(i)] = dist
                elif reduction == 'none':
                    loss_dic['outlier_{}'.format(i)] = torch.mean(dist, dim=1)

        ### Rotation and translation error
        if self.wt_pose_loss > 0:
            trans_scale = 1.0
            for i in range(num_iter):
                # err_r, err_t = self.pose_error(transform_gt, transform_pred[i])
                err_r = batch_rotation_error(transform_gt[:, :3, :3], transform_pred[i][:, :3, :3])   # [B]
                err_t = batch_translation_error(transform_gt[:, :3, 3], transform_pred[i][:, :3, 3])  # [B]

                if reduction == 'mean':
                    loss_dic['poseError_{}'.format(i)] = (torch.mean(err_r) +  \
                                                          torch.mean(err_t) * trans_scale) * self.wt_pose_loss
                elif reduction == 'none':
                    loss_dic['poseError_{}'.format(i)] = (err_r + err_t * trans_scale) * self.wt_pose_loss

        ### Early iterations will be discounted
        total_losses = []
        for k in loss_dic:
            discount = self.discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
            total_losses.append(loss_dic[k] * discount)
        loss_dic['total'] = torch.sum(torch.stack(total_losses), dim=0)

        return loss_dic


class SemanticLoss(nn.Module):
    def __init__(self):
        super(SemanticLoss, self).__init__()
        _logger.info('Semantic Loss.')
        self.num_classes = 19
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}

        self.ignored_labels = np.sort([0])
        self.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        self.class_weights = self.get_class_weights('SemanticKITTI')

        self.reset()

    def reset(self):
        self.gt_classes = [0 for _ in range(self.num_classes)]
        self.positive_classes = [0 for _ in range(self.num_classes)]
        self.true_positive_classes = [0 for _ in range(self.num_classes)]
        self.val_total_correct = 0
        self.val_total_seen = 0

    @staticmethod
    def get_class_weights(dataset_name):
        # pre-calculate the number of points in each category
        num_per_class = []
        if dataset_name is 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name is 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                     dtype=np.int32)
        elif dataset_name is 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                      9833174, 129609852, 4506626, 1168181])
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)

    @staticmethod
    def compute_acc(logits, labels):
        logits = logits.max(dim=1)[1]
        acc = (logits == labels).sum().float() / float(labels.shape[0])
        return acc

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
        # one_hot_labels = F.one_hot(labels, self.num_classes)

        # criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        # output_loss = criterion(logits, labels)
        # output_loss = output_loss.mean()
        output_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='mean')
        return output_loss

    def compute_loss(self, logits, labels):
        """
        logits: [B, num_class, N]
        labels: [B, N]
        """
        logits = logits.transpose(1, 2).reshape(-1, self.num_classes)   # [B*N, num_class]
        labels = labels.reshape(-1)

        # Boolean mask of points that should be ignored
        ignored_bool = (labels == 0)
        for ign_label in self.ignored_label_inds:
            ignored_bool = ignored_bool | (labels == ign_label)  # [B*N]

        # Collect logits and labels that are not ignored
        valid_idx = (ignored_bool == 0)
        valid_logits = logits[valid_idx, :]        # [(B*N)', num_class]
        valid_labels_init = labels[valid_idx]      # [(B*N)']

        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0, self.num_classes).long().cuda()
        inserted_value = torch.zeros((1,)).long().cuda()
        for ign_label in self.ignored_label_inds:
            reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], dim=0)

        valid_labels = torch.gather(reducing_list, dim=0, index=valid_labels_init)

        self.add_data(valid_logits, valid_labels)

        loss = self.get_loss(valid_logits, valid_labels, self.class_weights)
        acc = self.compute_acc(valid_logits, valid_labels)
        return loss, acc

    def add_data(self, logits, labels):
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        correct = np.sum(pred_valid == labels_valid)
        self.val_total_correct += correct
        self.val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def semantic_metric(self):
        iou_list = []
        for n in range(0, self.num_classes, 1):
            temp = float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
            if temp != 0:
                iou = self.true_positive_classes[n] / temp
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.num_classes)
        mean_acc = self.val_total_correct / float(self.val_total_seen)

        self.reset()
        return mean_iou, iou_list, mean_acc

    def forward(self, endpoints):
        loss_src, acc_src = self.compute_loss(endpoints['logits_src'], endpoints['labels_src'])
        loss_ref, acc_ref = self.compute_loss(endpoints['logits_ref'], endpoints['labels_ref'])
        loss = loss_src + loss_ref
        acc = acc_src + acc_ref

        # logits = torch.cat((endpoints['logits_src'], endpoints['logits_ref']), dim=-1)
        # labels = torch.cat((endpoints['labels_src'], endpoints['labels_ref']), dim=-1)
        # loss, acc = self.compute_loss(logits, labels)
        return loss, acc


