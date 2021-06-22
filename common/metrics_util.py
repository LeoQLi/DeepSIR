import logging
import numpy as np
from typing import Dict, List
import torch

from common.math import se3_torch
from common.math.so3 import dcm2euler
from common.torch_utils import to_numpy

_logger = logging.getLogger()


def rte_rre(T_pred, T_gt, rte_thresh, rre_thresh, eps=1e-16):
    """ Transformation Criteria
    T_pred, T_gt: [3/4, 4]
    rte_thresh, rre_thresh: float
    """
    if T_pred is None:
        return np.array([0, np.inf, np.inf])

    rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
    rre = np.arccos(np.clip((np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2,
                            -1 + eps, 1 - eps)) * 180 / np.pi
    return np.array([rte < rte_thresh and rre < rre_thresh, rte, rre])


def compute_metrics(data: Dict, pred_transforms, rte_thresh, rre_thresh, eps=1e-16) -> Dict:
    """ Compute metrics required in the paper
    """
    def square_distance(src, dst):
        # [B, N, 3] - [B, M, 3] => [B, N, M]
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    # TODO src and ref may not in raw
    gt_transforms = data['transform_gt']            # [B, 3, 4]
    points_src = data['points_src'][:, :2048, :3]        # [B, N, 3]
    points_ref = data['points_ref'][:, :2048, :3]        # [B, N, 3]
    if 'points_raw' in data:
        points_raw = data['points_raw'][..., :3]    # [B, N, 3]
    else:
        points_src_gt = se3_torch.transform(gt_transforms, points_src)
        points_raw = torch.cat([points_src_gt, points_ref], dim=1)   # [B, 2N, 3]

    with torch.no_grad():
        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3_torch.concatenate(se3_torch.inverse(gt_transforms), pred_transforms)  # [B, 3, 4]
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1),
                                    min=-1 + eps, max=1 - eps)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)   # [B, 3] -> [B]

        success = (residual_transmag < rte_thresh) * (residual_rotdeg < rre_thresh)  # [B]

        # Modified Chamfer distance
        src_transformed = se3_torch.transform(pred_transforms, points_src)
        ref_clean = points_raw
        inter__transforms = se3_torch.concatenate(pred_transforms, se3_torch.inverse(gt_transforms))
        src_clean = se3_torch.transform(inter__transforms, points_raw)    # should no transformation

        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_numpy(t_mse),
            't_mae': to_numpy(t_mae),
            'err_r_deg': to_numpy(residual_rotdeg),
            'err_t': to_numpy(residual_transmag),
            'succ': to_numpy(success),
            'chamfer_dist': to_numpy(chamfer_dist) }
    return metrics


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def print_metrics(summary_metrics: Dict, losses_by_iteration: List = None, title: str = 'Metrics'):
    """ Prints out formated metrics to logger
    """
    _logger.info('-' * (len(title) + 3))
    _logger.info(title + ':')

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
        _logger.info('Losses by iteration: {}'.format(losses_all_str))

    _logger.info('DCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae']))

    _logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))

    _logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))

    _logger.info('Chamfer error: {:.7f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']))

    _logger.info('Successful rate: {:.3f}'.format(
        summary_metrics['succ']))