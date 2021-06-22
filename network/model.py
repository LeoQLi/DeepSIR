import os, sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from common.math import se3_torch
from network.RandLANet import RandLA, FC, MLP
from network.matchnet import compute_affinity, sinkhorn  #, ParameterPredictionNet, AttentionalGNN
from network.matchnet import match_features_V2, feat_dist
from network.loss import DetDesLoss, ScanAlignmentLoss, SemanticLoss
from network.tools import gather_neighbour_V2, gather_neighbour_V3

_EPS = 1e-16  # To prevent division by zero
_logger = logging.getLogger()


def compute_rigid_transform_2(src, tgt, weights):
    """ Compute rigid transforms between two point sets

    Args:
        src (torch.Tensor): (B, M, 3) source points
        tgt (torch.Tensor): (B, N, 3) target points
        weights (torch.Tensor): (B, M, 1)

    Returns:
        Transform T (B, 3, 4) to get from src to tgt, i.e. T*src = tgt
    """
    # compute weighted coordinates
    invalid_gradient = False
    weights_norm = weights / (torch.sum(torch.abs(weights), dim=1, keepdim=True) + _EPS)  # [B, M, 1]

    centroid_src = torch.sum(src * weights_norm, dim=1)       # [B, 3] <= [B, M, 3] * [B, M, 1]
    centroid_tgt = torch.sum(tgt * weights_norm, dim=1)       # [B, 3] <= [B, M, 3] * [B, M, 1]
    src_centered = src - centroid_src[:, None, :]             # [B, M, 3]
    tgt_centered = tgt - centroid_tgt[:, None, :]             # [B, M, 3]
    cov = src_centered.transpose(-2, -1).contiguous() @ (tgt_centered * weights_norm)   # [B, 3, 3]=[B, 3, M]x[B, M, 3]

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    try:
        # Use CPU for small arrays
        u, s, v = torch.svd(cov.cpu().double(), some=False, compute_uv=True)  # [B, 3, 3], [B, 3], [B, 3, 3]

        rot_mat_pos = v @ u.transpose(-1, -2).contiguous()        # [B, 3, 3]
        v_neg = v.clone()                                         # [B, 3, 3]
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2).contiguous()    # [B, 3, 3]
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg).float()  # [B, 3, 3]
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_src[:, :, None].cpu() + centroid_tgt[:, :, None].cpu()  # [B, 3, 1]
        transform = torch.cat((rot_mat, translation), dim=2).to(src.device)               # [B, 3, 4]

    # torch.svd may have convergence issues for GPU and CPU (nan value)
    except Exception as error:
        _logger.error(error)
        transform = se3_torch.identity(len(src)).to(src.device)
        invalid_gradient = True

    return transform, invalid_gradient

def compute_rigid_transform(src, tgt, weights):
    """ Compute rigid transforms between two point sets

    Args:
        src (torch.Tensor): (B, M, 3) source points
        tgt (torch.Tensor): (B, N, 3) target points
        weights (torch.Tensor): (B, M, N)

    Returns:
        Transform T (B, 3, 4) to get from src to tgt, i.e. T*src = tgt
    """
    # compute weighted coordinates
    invalid_gradient = False
    weights_sum = torch.sum(weights, dim=2, keepdim=True)     # [B, M, 1]
    weights_norm = weights_sum / (torch.sum(weights_sum, dim=1, keepdim=True) + _EPS)  # [B, M, 1]

    # compute new target points
    tgt = weights @ tgt / (weights_sum + _EPS)                # [B, M, 3] = [B, M, N]x[B, N, 3]/[B, M, 1]

    centroid_src = torch.sum(src * weights_norm, dim=1)       # [B, 3] = [B, M, 3] * [B, M, 1]
    centroid_tgt = torch.sum(tgt * weights_norm, dim=1)       # [B, 3] = [B, M, 3] * [B, M, 1]
    src_centered = src - centroid_src[:, None, :]             # [B, M, 3]
    tgt_centered = tgt - centroid_tgt[:, None, :]             # [B, M, 3]
    cov = src_centered.transpose(-2, -1).contiguous() @ (tgt_centered * weights_norm)   # [B, 3, 3]=[B, 3, M]x[B, M, 3]

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    try:
        # Use CPU for small arrays
        u, s, v = torch.svd(cov.cpu().double(), some=False, compute_uv=True)     # [B, 3, 3], [B, 3], [B, 3, 3]

        rot_mat_pos = v @ u.transpose(-1, -2).contiguous()        # [B, 3, 3]
        v_neg = v.clone()                                         # [B, 3, 3]
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2).contiguous()    # [B, 3, 3]
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)  # [B, 3, 3]
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_src[:, :, None].cpu() + centroid_tgt[:, :, None].cpu()  # [B, 3, 1]
        transform = torch.cat((rot_mat, translation), dim=2).float().to(src.device)               # [B, 3, 4]

    # torch.svd may have convergence issues for GPU and CPU (nan value)
    except Exception as error:
        _logger.error(error)
        transform = se3_torch.identity(len(src)).to(src.device)
        invalid_gradient = True

    return transform, invalid_gradient


class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipeline = args.pipeline
        self.num_sub  = args.num_sub
        self.num_knn  = args.num_knn
        self.d_out    = args.out_feat_dim
        self.clip_weight_thresh = args.clip_weight_thresh
        self.compute_score = False
        self.return_flag   = False
        self.sub_selection = False

        assert self.pipeline in ['align', 'feat', 'label']

        self.feat_extractor = RandLA(args)

        if self.pipeline != 'label':
            self.freeze_model()
            self.compute_score = True
            self.sub_selection = True if self.num_sub > 0 else False

            ### semantic score weights
            # {0: 'unlabeled'} in the input data is ignored
            # labels = { 0: 'car',       1: 'bicycle',       2: 'motorcycle',   3: 'truck',  4: 'other-vehicle',
            #            5: 'person',    6: 'bicyclist',     7: 'motorcyclist', 8: 'road',   9: 'parking',
            #            10: 'sidewalk', 11: 'other-ground', 12: 'building',    13: 'fence', 14: 'vegetation',
            #            15: 'trunk',    16: 'terrain',      17: 'pole',        18: 'traffic-sign'}
            self.label_weights = [3,  1,  1,  3,  2,
                                  0,  0,  0,  6,  5,
                                  6,  4,  7,  7,  6,
                                  8,  4,  9,  9]
            self.label_weights = torch.tensor(self.label_weights).float()

            # num_layers = 4
            # num_heads  = 4
            # layer_names = ['self', 'cross'] * num_layers
            # self.atten = AttentionalGNN(feature_dim=self.d_out, layer_names=layer_names, num_heads=num_heads)

            # =====================================================
            d_feat = [self.d_out, self.d_out, 128, self.d_out]
            self.mlp_feat = MLP(d_feat, do_bn=True, full=False)

            d_att = [4, 32, 64, 128, 256, self.d_out]
            # d_att = [4, 32, 64, 128, self.d_out]
            # d_att = [4, 32, 64, 128, 256, 128, self.d_out]
            # d_att = [4, 32, 64, 128, 256, self.d_out]
            # d_att = [68, 128, 256, 512, 265, self.d_out]
            # d_att = [68, 128, 256, 128, self.d_out]
            self.mlp_att = MLP(d_att, do_bn=True, full=False)

            d_proj = [self.d_out, self.d_out]
            self.mlp_proj = MLP(d_proj, do_bn=True, full=False)
            # =====================================================

        if self.pipeline == 'label':
            self.forward_fun = self.forward_pair
            self.loss_label_fun = SemanticLoss()

        elif self.pipeline == 'feat':
            self.forward_fun   = self.forward_pair
            self.loss_feat_fun = DetDesLoss(args)

        elif self.pipeline == 'align':
            self.freeze_model_2()
            self.return_flag = True
            self.forward_fun = self.forward_align_4
            # self.add_slack   = not args.no_slack
            # self.num_sk_iter = args.num_sk_iter

            # self.weights_net = ParameterPredictionNet(weights_dim=[0])
            # self.weights_net = ParameterPredictionNetConstant(weights_dim=[0])

            new_args = copy.copy(args)
            new_args.feat_len = 6
            self.inlier_model = RandLA(new_args, num_classes=1)

            self.loss_align_fun = ScanAlignmentLoss(args)

    def freeze_model(self):
        for p in self.feat_extractor.parameters():
            p.requires_grad = False

    def freeze_model_2(self):
        for p in self.mlp_feat.parameters():
            p.requires_grad = False
        for p in self.mlp_att.parameters():
            p.requires_grad = False
        for p in self.mlp_proj.parameters():
            p.requires_grad = False

    def aggregation(self, xyz_src, xyz_ref, feat_src, feat_ref,
                            label_src=None, label_ref=None, score_src=None, score_ref=None):
        """
        xyz_src: [B, 3, J]
        xyz_ref: [B, 3, K]
        feat: [B, C, N]
        score: [B, N]
        label: [B, 1, N]
        """
        feat_src = self.mlp_feat(feat_src)
        feat_ref = self.mlp_feat(feat_ref)

        # xyz_src_g = torch.cat((xyz_src, label_src.float()), dim=1)      # [B, 4, N]
        # xyz_ref_g = torch.cat((xyz_ref, label_ref.float()), dim=1)      # [B, 4, N]
        xyz_src_g = torch.cat((xyz_src, score_src[:, None, :]), dim=1)    # [B, 4, N]
        xyz_ref_g = torch.cat((xyz_ref, score_ref[:, None, :]), dim=1)    # [B, 4, N]

        xyz_src_g, xyz_ref_g = self.mlp_att(xyz_src_g), self.mlp_att(xyz_ref_g)   # [B, C, N]

        feat_src = feat_src + xyz_src_g                                # [B, C, N]
        feat_ref = feat_ref + xyz_ref_g                                # [B, C, N]

        feat_src, feat_ref = self.mlp_proj(feat_src), self.mlp_proj(feat_ref)

        feat_src = F.normalize(feat_src, p=2, dim=1)
        feat_ref = F.normalize(feat_ref, p=2, dim=1)
        return feat_src, feat_ref

    def aggregation_x(self, xyz_src, xyz_ref, feat_src, feat_ref,
                            label_src=None, label_ref=None, score_src=None, score_ref=None):
        """
        xyz_src: [B, 3, J]
        xyz_ref: [B, 3, K]
        feat: [B, C, N]
        label: [B, 1, N]
        """
        # xyz_src = xyz_src - torch.mean(xyz_src, dim=-1, keepdim=True)   # [B, 3, N]
        # xyz_ref = xyz_ref - torch.mean(xyz_ref, dim=-1, keepdim=True)   # [B, 3, N]

        score_src = score_src[:, None, :]     # [B, 1, N]
        score_ref = score_ref[:, None, :]
        # feat_src = self.mlp_feat(feat_src)
        # feat_ref = self.mlp_feat(feat_ref)

        flag = 0
        if flag == 0:
            # xyz_src_g = torch.cat((xyz_src, label_src.float(), score_src), dim=1)      # [B, 5, N]
            # xyz_ref_g = torch.cat((xyz_ref, label_ref.float(), score_ref), dim=1)      # [B, 5, N]
            # xyz_src_g = torch.cat((xyz_src, label_src.float()), dim=1)      # [B, 4, N]
            # xyz_ref_g = torch.cat((xyz_ref, label_ref.float()), dim=1)      # [B, 4, N]
            xyz_src_g = torch.cat((xyz_src, score_src), dim=1)      # [B, 4, N]
            xyz_ref_g = torch.cat((xyz_ref, score_ref), dim=1)      # [B, 4, N]

            xyz_src_g, xyz_ref_g = self.mlp_att(xyz_src_g), self.mlp_att(xyz_ref_g)   # [B, C, N]

            # xyz_src_g, xyz_ref_g = self.atten(xyz_src_g, xyz_ref_g)

            feat_src = feat_src + xyz_src_g                                # [B, C, N]
            feat_ref = feat_ref + xyz_ref_g                                # [B, C, N]

            feat_src, feat_ref = self.mlp_proj(feat_src), self.mlp_proj(feat_ref)

        elif flag == 1:
            xyz_src_g = torch.cat((xyz_src, label_src.float()), dim=1)      # [B, 4, N]
            xyz_ref_g = torch.cat((xyz_ref, label_ref.float()), dim=1)      # [B, 4, N]

            feat_src = feat_src + self.mlp_att(xyz_src_g)                   # [B, C, N]
            feat_ref = feat_ref + self.mlp_att(xyz_ref_g)                   # [B, C, N]

            # feat_src, feat_ref = self.atten(feat_src, feat_ref)            # [B, C, N]

            feat_src, feat_ref = self.mlp_proj(feat_src), self.mlp_proj(feat_ref)

        elif flag == 2:
            # xyz_src_g = torch.cat((xyz_src, label_src.float(), score_src, feat_src), dim=1)  # [B, 69, N]
            # xyz_ref_g = torch.cat((xyz_ref, label_ref.float(), score_ref, feat_ref), dim=1)  # [B, 69, N]
            xyz_src_g = torch.cat((xyz_src, label_src.float(), feat_src), dim=1)             # [B, 68, N]
            xyz_ref_g = torch.cat((xyz_ref, label_ref.float(), feat_ref), dim=1)             # [B, 68, N]
            # xyz_src_g = torch.cat((xyz_src, score_src, feat_src), dim=1)                       # [B, 68, N]
            # xyz_ref_g = torch.cat((xyz_ref, score_ref, feat_ref), dim=1)                       # [B, 68, N]

            feat_src = self.mlp_att(xyz_src_g)                   # [B, C, N]
            feat_ref = self.mlp_att(xyz_ref_g)                   # [B, C, N]

        feat_src = F.normalize(feat_src, p=2, dim=1)
        feat_ref = F.normalize(feat_ref, p=2, dim=1)
        return feat_src, feat_ref

    def forward(self, data, opt=None):
        return self.forward_fun(data, opt)

    def forward_align_1(self, data, opt=None):
        """ Forward pass

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, C)
                    'points_ref': Reference points (B, K, C)
            num_reg_iter (int): Number of iterations.

        Returns:
            R_t: Transform to apply to source points such that they align to reference
            endpoints: (src, ref) point clouds [B, N, 3]
        """
        points_src     = data['points_src']             # xyz+others [B, N, C]
        src_xyz_multi  = data['points_src_xyz']         # xyz coordinate [B, n0+n1+..., 3]
        src_neigh_idx  = data['points_src_neigh_idx']
        src_sub_idx    = data['points_src_sub_idx']
        src_interp_idx = data['points_src_interp_idx']

        feat_src_0, xyz_src, label_src, score_src,  \
        feat_ref_0, xyz_ref, label_ref, score_ref = self.forward_pair(data)

        endpoints = {}
        endpoints['pt_src'] = xyz_src.permute(0, 2, 1).contiguous()   # [B, N, 3], un-transformed points
        endpoints['pt_ref'] = xyz_ref.permute(0, 2, 1).contiguous()

        # match_matrix_0 = torch.zeros((xyz_src.shape[0], xyz_src.shape[2], xyz_ref.shape[2])).to(xyz_src.device)
        # match_matrix_0 = None

        # self.clip_weight_thresh = 0.05
        transforms = []
        all_matrices = []
        all_beta, all_alpha = [], []
        invalid_gradient = False
        for i in range(num_reg_iter):
            feat_src, feat_ref = self.aggregation(xyz_src, xyz_ref, feat_src_0, feat_ref_0,
                                                label_src, label_ref, score_src, score_ref)

            ########################### Matching ##########################
            # Coarse correspondences: compute feature distance between src and ref
            # match_matrix = feat_dist(feat_src, feat_ref)
            match_matrix = match_features_V2(feat_src, feat_ref)                # [B, J, K]

            beta, alpha = self.weights_net(xyz_src, xyz_ref)                    # [B], [B]
            match_matrix = compute_affinity(beta, match_matrix, alpha=alpha)    # [B, J, K]

            match_matrix = sinkhorn(match_matrix, n_iters=self.num_sk_iter, slack=self.add_slack)  # [B, J, K]
            match_matrix = torch.exp(match_matrix)                              # [B, J, K]

            # if self.clip_weight_thresh > 0:
            #     match_matrix[match_matrix < self.clip_weight_thresh] = 0

            if torch.isnan(match_matrix).any():
                _logger.error('Nan value in match_matrix!')
                _logger.info('Sum of affinity: {}'.format(affinity.sum().item()))
            ########################### Matching ##########################

            ######################## Transformation ########################
            # Compute R_t and transform source points
            # TODO cycle consistency: T(src->ref) and T(ref->src)
            xyz_src = xyz_src.permute(0, 2, 1).contiguous()                     # [B, N, 3]
            xyz_ref = xyz_ref.permute(0, 2, 1).contiguous()
            R_t, cur_invalid_gradient = compute_rigid_transform(xyz_src, xyz_ref, weights=match_matrix)  # [B, 3, 4]

            xyz_src = se3_torch.transform(R_t.detach(), xyz_src)
            xyz_src = xyz_src.permute(0, 2, 1).contiguous()                     # [B, 3, N]
            xyz_ref = xyz_ref.permute(0, 2, 1).contiguous()
            ######################## Transformation ########################

            # the last one is the final transformation
            transforms.append(R_t) if i == 0 else transforms.append(se3_torch.concatenate(R_t, transforms[-1]))
            all_matrices.append(match_matrix)
            all_beta.append(beta)
            all_alpha.append(alpha)
            invalid_gradient = invalid_gradient or cur_invalid_gradient

        endpoints['beta']  = torch.mean(torch.stack(all_beta, axis=0))
        endpoints['alpha'] = torch.mean(torch.stack(all_alpha, axis=0))
        endpoints['perm_matrices'] = all_matrices                               # [B, J, K]
        endpoints['invalid_gradient'] = invalid_gradient
        # endpoints['feat_src'] = feat_src
        # endpoints['feat_ref'] = feat_ref
        return transforms, endpoints

    def forward_align_2(self, data, opt=None):
        """ Forward pass

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, C)
                    'points_ref': Reference points (B, K, C)
            num_reg_iter (int): Number of iterations.

        Returns:
            R_t: Transform to apply to source points such that they align to reference
            endpoints: (src, ref) point clouds [B, N, 3]
        """
        feat_src, xyz_src, label_src, \
        feat_ref, xyz_ref, label_ref = self.forward_pair(data)

        endpoints = {}
        endpoints['pt_src'] = xyz_src.permute(0, 2, 1).contiguous()   # [B, N, 3], un-transformed points
        endpoints['pt_ref'] = xyz_ref.permute(0, 2, 1).contiguous()

        score_src = score_src[:, None, :]   # [B, 1, N]
        score_ref = score_ref[:, None, :]
        feat_src, feat_ref = self.aggregation(xyz_src, xyz_ref, feat_src, feat_ref,
                                                        label_src, label_ref)

        match_matrix = match_features_V2(feat_src, feat_ref)                    # [B, J, K]

        transforms = []
        all_matrices = []
        all_beta, all_alpha = [], []
        invalid_gradient = False
        for i in range(num_reg_iter):
            beta, alpha = self.weights_net(xyz_src, xyz_ref)                    # [B], [B]
            match_matrix = compute_affinity(beta, match_matrix, alpha=alpha)    # [B, J, K]

            # compute weighted coordinates
            match_matrix = sinkhorn(match_matrix, n_iters=self.num_sk_iter, slack=self.add_slack)  # [B, J, K]
            match_matrix = torch.exp(match_matrix)                              # [B, J, K]

            if torch.isnan(match_matrix).any():
                _logger.error('Nan value in match_matrix!')
                _logger.info('Sum of affinity: {}'.format(affinity.sum().item()))

            ######################## Transformation ########################
            # Compute R_t and transform source points
            xyz_src = xyz_src.permute(0, 2, 1).contiguous()                     # [B, N, 3]
            xyz_ref = xyz_ref.permute(0, 2, 1).contiguous()
            R_t, cur_invalid_gradient = compute_rigid_transform(xyz_src, xyz_ref, weights=match_matrix)  # [B, 3, 4]

            xyz_src = se3_torch.transform(R_t.detach(), xyz_src)
            xyz_src = xyz_src.permute(0, 2, 1).contiguous()                     # [B, 3, N]
            xyz_ref = xyz_ref.permute(0, 2, 1).contiguous()
            ######################## Transformation ########################

            # the last one is the final transformation
            transforms.append(R_t) if i == 0 else transforms.append(se3_torch.concatenate(R_t, transforms[-1]))
            all_matrices.append(match_matrix)
            all_beta.append(beta)
            all_alpha.append(alpha)
            invalid_gradient = invalid_gradient or cur_invalid_gradient

        endpoints['beta']   = torch.mean(torch.stack(all_beta, axis=0))
        endpoints['alpha']  = torch.mean(torch.stack(all_alpha, axis=0))
        endpoints['perm_matrices'] = all_matrices                               # [B, J, K]
        endpoints['invalid_gradient'] = invalid_gradient
        return transforms, endpoints

    def forward_align_3(self, data, opt=None):
        """ Forward pass

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, C)
                    'points_ref': Reference points (B, K, C)
            num_reg_iter (int): Number of iterations.

        Returns:
            R_t: Transform to apply to source points such that they align to reference
            endpoints: (src, ref) point clouds [B, N, 3]
        """
        transforms = []
        all_matrices = []
        all_beta, all_alpha = [], []
        invalid_gradient = False
        for i in range(num_reg_iter):
            feat_src, xyz_src, label_src, \
            feat_ref, xyz_ref, label_ref = self.forward_pair(data)

            feat_src = F.normalize(feat_src, p=2, dim=1)
            feat_ref = F.normalize(feat_ref, p=2, dim=1)

            ########################### Matching ##########################
            # compute feature distance between src and ref
            # match_matrix = feat_dist(feat_src, feat_ref)
            match_matrix = match_features_V2(feat_src, feat_ref)                # [B, J, K]

            beta, alpha = self.weights_net(xyz_src, xyz_ref)                    # [B], [B]
            match_matrix = compute_affinity(beta, match_matrix, alpha=alpha)    # [B, J, K]

            # compute weighted coordinates
            match_matrix = sinkhorn(match_matrix, n_iters=self.num_sk_iter, slack=self.add_slack)  # [B, J, K]
            match_matrix = torch.exp(match_matrix)                              # [B, J, K]

            if torch.isnan(match_matrix).any():
                _logger.error('Nan value in match_matrix!')
                _logger.info('Sum of affinity: {}'.format(affinity.sum().item()))
            ########################### Matching ##########################

            ######################## Transformation ########################
            # Compute R_t and transform source points
            # TODO cycle consistency: T(src->ref) and T(ref->src)
            xyz_src = xyz_src.permute(0, 2, 1).contiguous()                     # [B, N, 3]
            xyz_ref = xyz_ref.permute(0, 2, 1).contiguous()
            R_t, cur_invalid_gradient = compute_rigid_transform(xyz_src, xyz_ref, weights=match_matrix)  # [B, 3, 4]

            data['points_src'][:, :, :3] = se3_torch.transform(R_t.detach(), data['points_src'][:, :, :3])
            data['points_src_xyz'] = se3_torch.transform(R_t.detach(), data['points_src_xyz'])
            ######################## Transformation ########################

            # the last one is the final transformation
            transforms.append(R_t) if i == 0 else transforms.append(se3_torch.concatenate(R_t, transforms[-1]))
            all_matrices.append(match_matrix)
            all_beta.append(beta)
            all_alpha.append(alpha)
            invalid_gradient = invalid_gradient or cur_invalid_gradient

        endpoints = {}
        # the generated xyz_src is transformed during the iteraction, [B, N, 3]
        endpoints['pt_src'] = se3_torch.transform(se3_torch.inverse(transforms[-2].detach()), xyz_src)
        endpoints['pt_ref'] = xyz_ref
        endpoints['beta']   = torch.mean(torch.stack(all_beta, axis=0))
        endpoints['alpha']  = torch.mean(torch.stack(all_alpha, axis=0))
        endpoints['perm_matrices'] = all_matrices                               # [B, J, K]
        endpoints['invalid_gradient'] = invalid_gradient
        return transforms, endpoints

    def forward_align_4(self, data, opt=None):
        """ Forward pass

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, C)
                    'points_ref': Reference points (B, K, C)
            opt: forward options

        Returns:
            R_t: Transform to apply to source points such that they align to reference
            endpoints: (src, ref) point clouds [B, N, 3]
        """
        num_reg_iter, clip_weight = opt
        points_src     = data['points_src']             # xyz+others [B, N, C]
        src_xyz_multi  = data['points_src_xyz']         # xyz coordinate [B, n0+n1+..., 3]
        src_neigh_idx  = data['points_src_neigh_idx']
        src_sub_idx    = data['points_src_sub_idx']
        src_interp_idx = data['points_src_interp_idx']

        feat_src_0, xyz_src, label_src, score_src,  \
        feat_ref_0, xyz_ref, label_ref, score_ref = self.forward_pair(data)

        endpoints = {}
        endpoints['pt_src'] = xyz_src.permute(0, 2, 1).contiguous()   # [B, N, 3], un-transformed points
        endpoints['pt_ref'] = xyz_ref.permute(0, 2, 1).contiguous()

        transforms = []
        all_matrices = []
        all_pred_pairs = []
        invalid_gradient = False
        for iter in range(num_reg_iter):
            feat_src, feat_ref = self.aggregation(xyz_src, xyz_ref, feat_src_0, feat_ref_0,
                                                label_src, label_ref, score_src, score_ref)

            ########################### Matching ##########################
            ### Coarse correspondences: compute feature distance between src and ref
            # use too much memory if the feature matrix is large
            with torch.no_grad():  # TODO
                stride = 6000
                N = feat_src.shape[2]
                C = int(np.ceil(N / stride))
                indexs = []
                for n in range(int(np.ceil(N / stride))):
                    # match_matrix = feat_dist(feat_src[:, :, n * stride:(n + 1) * stride], feat_ref)
                    match_matrix = match_features_V2(feat_src[:, :, n * stride:(n + 1) * stride], feat_ref)   # [B, stride, K]
                    index = match_matrix.min(dim=2, keepdim=False)[1]   # [B, stride]
                    indexs.append(index)
                indexs = torch.cat(indexs, dim=1)                       # [B, J], J <= K
                assert indexs.shape[1] == N

            xyz_ref_new = gather_neighbour_V3(xyz_ref, indexs)          # [B, 3, J]

            ### Compute the matching probability
            cat_xyz = torch.cat((xyz_src, xyz_ref_new), dim=1).permute(0, 2, 1).contiguous()      # [B, 6, J]
            _, _, logit = self.inlier_model(cat_xyz, src_xyz_multi, src_neigh_idx, src_sub_idx, src_interp_idx)  # [B, 1, J]
            logit = logit.squeeze(dim=1)
            weights = logit.sigmoid()[:, :, None]  # [B, J, 1]
            ########################### Matching ##########################

            ### Truncate weights too low. For training, inplace modification is prohibited for backward
            # if clip_weight and self.clip_weight_thresh > 0:
            #     weights[weights < self.clip_weight_thresh] = 0

            ######################## Transformation ########################
            # Compute R_t and transform source points
            xyz_src = xyz_src.permute(0, 2, 1).contiguous()            # [B, N, 3]
            xyz_ref_new = xyz_ref_new.permute(0, 2, 1).contiguous()    # [B, N, 3]
            R_t, cur_invalid_gradient = compute_rigid_transform_2(xyz_src, xyz_ref_new, weights=weights)  # [B, 3, 4]

            xyz_src = se3_torch.transform(R_t.detach(), xyz_src)
            xyz_src = xyz_src.permute(0, 2, 1).contiguous()          # [B, 3, N]
            ######################## Transformation ########################

            # the last one is the final transformation
            transforms.append(R_t) if iter == 0 else transforms.append(se3_torch.concatenate(R_t, transforms[-1]))
            all_matrices.append(logit)
            invalid_gradient = invalid_gradient or cur_invalid_gradient

            corres_idx0 = torch.arange(indexs.shape[1])[None, :].expand(indexs.shape[0], indexs.shape[1]).int()[:, :, None]  # [B, J, 1]
            corres_idx1 = indexs.int().cpu()[:, :, None]   # [B, J, 1]
            all_pred_pairs.append(torch.cat([corres_idx0, corres_idx1], dim=2))  # [B, J, 2]

        endpoints['perm_matrices'] = all_matrices             # [B, J]
        endpoints['pred_pairs'] = all_pred_pairs              # [B, J, 2]
        endpoints['invalid_gradient'] = invalid_gradient
        endpoints['pt_ref_new'] = xyz_ref_new
        return transforms, endpoints

    def forward_pair(self, data, opt=None):
        """ Forward pass

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, C)
                    'points_ref': Reference points (B, K, C)

        Returns:
            pt: [B, 3, N], feat: [B, C, N], score: [B, N], logits: [B, num_class, N]
        """
        points_src     = data['points_src']             # xyz+others [B, N, C]
        src_xyz_multi  = data['points_src_xyz']         # xyz coordinate [B, n0+n1+..., 3]
        src_neigh_idx  = data['points_src_neigh_idx']
        src_sub_idx    = data['points_src_sub_idx']
        src_interp_idx = data['points_src_interp_idx']

        points_ref     = data['points_ref']
        ref_xyz_multi  = data['points_ref_xyz']
        ref_neigh_idx  = data['points_ref_neigh_idx']
        ref_sub_idx    = data['points_ref_sub_idx']
        ref_interp_idx = data['points_ref_interp_idx']

        feat_src, xyz_src, logits_src = self.feat_extractor(points_src, src_xyz_multi, src_neigh_idx,
                                                            src_sub_idx, src_interp_idx)
        feat_ref, xyz_ref, logits_ref = self.feat_extractor(points_ref, ref_xyz_multi, ref_neigh_idx,
                                                            ref_sub_idx, ref_interp_idx)

        if self.compute_score:
            prob_src, label_src = torch.max(logits_src, dim=1, keepdim=True)   # [B, 1, N]
            prob_ref, label_ref = torch.max(logits_ref, dim=1, keepdim=True)   # [B, 1, N]

            feat_src, xyz_src, label_src, score_src = self.feat_score(
                    feat_src, xyz_src, prob_src, label_src, src_neigh_idx, self.num_sub)
            feat_ref, xyz_ref, label_ref, score_ref = self.feat_score(
                    feat_ref, xyz_ref, prob_ref, label_ref, ref_neigh_idx, self.num_sub)

            if self.return_flag:
                return feat_src, xyz_src, label_src, score_src,  \
                       feat_ref, xyz_ref, label_ref, score_ref

            feat_src, feat_ref = self.aggregation(xyz_src, xyz_ref, feat_src, feat_ref,
                                                            label_src, label_ref, score_src, score_ref)

        feat_src = F.normalize(feat_src, p=2, dim=1)
        feat_ref = F.normalize(feat_ref, p=2, dim=1)

        endpoints = {}
        endpoints['pt_src']     = xyz_src      # [B, 3, N]
        endpoints['pt_ref']     = xyz_ref
        endpoints['feat_src']   = feat_src     # [B, C, N]
        endpoints['feat_ref']   = feat_ref
        endpoints['logits_src'] = logits_src   # [B, num_class, N]
        endpoints['logits_ref'] = logits_ref
        if self.compute_score:
            endpoints['score_src']  = score_src    # [B, N]
            endpoints['score_ref']  = score_ref
        return None, endpoints

    def feat_score(self, feat, xyz, prob, label, neigh_idx, num_sub=0):
        """
        :param xyz: [B, 3, N]
        :param feat: [B, C, N]
        :param prob: [B, 1, N]
        :param label: [B, 1, N]
        :param neigh_idx: [B, N1+N2+..., nsample]
        :return:
        """
        num_points = xyz.shape[2]
        neigh_idx = neigh_idx[:, 0:num_points, :]
        score = self.score_fun(feat, xyz, prob, label, neigh_idx)  # [B, N]

        # used during feat test or align
        if self.sub_selection:
            assert 0 < num_sub <= num_points

            # neigh_idx = neigh_idx[:, 0:self.num_points, :]
            # score_stati = self.score_fun(feat, xyz, label, neigh_idx)   # [B, N]
            # score_stati = self.score_fun(score, xyz, label, neigh_idx)  # [B, N]
            # score = score * score_stati

            assert xyz.shape[-1] == score.shape[-1]
            # find the top-k score points and its feature
            score, index = torch.topk(score, k=num_sub, dim=-1, largest=True)   # [B, k]

            # select the points and theirs features according to index
            xyz = gather_neighbour_V3(xyz, index)       # [B, 3, k]
            feat = gather_neighbour_V3(feat, index)     # [B, C, k]
            label = gather_neighbour_V3(label, index)   # [B, 1, k]

        return feat, xyz, label, score

    def score_fun(self, feat, xyz, prob, label, neigh_idx):
        """
        :param xyz: [B, 3, N]
        :param feat: [B, C, N]
        :param prob: [B, 1, N]
        :param label: [B, 1, N]
        :param neigh_idx: [B, N, nsample]
        :return: score: [B, k]
        """
        batch = feat.shape[0]
        k_neighbors = 16
        assert k_neighbors <= self.num_knn
        assert neigh_idx.shape[1] == xyz.shape[-1]
        neigh_idx = neigh_idx[:, :, :k_neighbors]  # [B, N, k]
        local_max_score = aggregation_score = depth_wise_max_score = label_score = 1.0

        ### 0. normalize the feature to avoid overflow
        # the max channel value of all points in a batch
        max_per_sample = torch.max(feat.view(batch, -1), dim=1, keepdims=True)[0]   # [B, 1]
        feat_norm = feat / (max_per_sample.view(batch, 1, 1) + _EPS)                # [B, C, N]

        ############################ Score ############################
        ### 1. the local max score (saliency score)
        neighbor_feat = gather_neighbour_V2(feat_norm, neigh_idx)          # [B, C, N, k]
        neighbor_feat = torch.mean(neighbor_feat, dim=3, keepdims=False)   # [B, C, N]
        local_max_score = F.softplus(feat_norm - neighbor_feat)            # [B, C, N]

        ### 2. aggregation score to remove isolated points
        ball_r = 2.0
        neighbor_xyz = gather_neighbour_V2(xyz, neigh_idx)                 # [B, 3, N, k]
        relative_xyz = neighbor_xyz - xyz.unsqueeze(-1)                    # [B, 3, N, k]
        relative_xyz = torch.norm(relative_xyz, dim=1, keepdim=True)       # [B, 1, N, k]
        relative_xyz = torch.mean(relative_xyz, dim=-1, keepdims=False)    # [B, 1, N]
        aggregation_score = (relative_xyz < ball_r).float()                # [B, 1, N]  (0 or 1)

        ### 3. the channel-wise max score
        depth_wise_max = torch.max(feat_norm, dim=1, keepdims=True)[0]     # [B, 1, N]
        depth_wise_max_score = feat_norm / (depth_wise_max + _EPS)         # [B, C, N]

        ### 4. semantic score
        label = label.view(-1).long()                                      # [B*N]
        label_score = self.label_weights[label].to(xyz.device)             # [B*N]
        label_score = label_score.view(batch, 1, xyz.shape[-1])            # [B, 1, N]
        label_max = torch.max(label_score, dim=-1, keepdims=True)[0]       # [B, 1, 1]
        label_score = label_score / (label_max + _EPS)                     # [B, 1, N]  in range(0, 1)

        prob_max = torch.max(prob, dim=-1, keepdims=True)[0]               # [B, 1, 1]
        prob = prob / (prob_max + _EPS)                                    # [B, 1, N]  in range(0, 1)
        label_score = label_score * torch.gt(prob, 0.2)                    # scores with prob > 0.2

        ### 5. total score
        score = local_max_score * aggregation_score * depth_wise_max_score * label_score

        # use the max score among channel to be the score of a single point
        score = torch.max(score, dim=1, keepdims=False)[0]                 # [B, N]
        ############################ Score ############################
        return score    # [B, N]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device('cuda:0')

    batch = 2
    N = 10
    label_weights = [3,  1,  1,  3,  2,
                            0,  0,  0,  6,  5,
                            5,  4,  8,  7,  6,
                            8,  4,  9,  9]
    label_weights = torch.tensor(label_weights).float().cuda()

    label = np.random.randint(0, 19, (batch, N))
    label = torch.from_numpy(label).cuda()
    print(label)

    label = label.view(-1).long()                                      # [B*N]
    label_score = label_weights[label]                            # [B*N]
    label_score = label_score.view(batch, 1, N)            # [B, 1, N]
    print(label_score)

    label_max = torch.max(label_score, dim=-1, keepdims=True)[0]       # [B, 1, 1]
    label_score = label_score / (label_max + _EPS)                     # [B, 1, N]
    print(label_score)


    # net = MLP([3, 32, 64])
    # nn.init.constant_(net[-1].bias, 0.0)
    # net = Network(None, 0)
    # print(net)

    # B = 2
    # num_sk_iter = 2
    # add_slack = True
    # feat_src = torch.randn(B, 10, 32)
    # feat_ref = torch.randn(B, 12, 32)
    # xyz_src = torch.randn(B, 10, 3)
    # xyz_ref = torch.randn(B, 12, 3)
    # beta = torch.randn(B)
    # alpha = torch.randn(B)

    # pt_fea = torch.randn((B, 5, 1), dtype=torch.float).to(pytorch_device)
    # pt = torch.randn((B, 5, 10), dtype=torch.float).to(pytorch_device)
    # # pt_fea1 = [pt_fea[i, :, :] for i in range(B)]
    # # pt_fea1 = torch.cat(pt_fea1, dim=0)
    # # pt_fea2 = pt_fea.view(-1, 10)
    # # bo = torch.equal(pt_fea1, pt_fea2)
    # topk_mask = topk(pt_fea, 3)
    # b = topk_mask.to(torch.float) * pt_fea
    # idx = b.nonzero()
    # c = pt_fea[idx[:, 0], idx[:, 1], :].view(B, 3, -1)
    # print(pt_fea)
    # print(b)
    # print(c)


