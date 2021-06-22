""" Evaluate model. Also contains functionality to compute evaluation metrics given transforms

Example Usages:
    1. Evaluate model
        python eval.py --noise_type crop --resume [path-to-model.pth]

    2. Evaluate precomputed transforms (.npy file containing np.array of size (B, 3, 4) or (B, n_iter, 3, 4))
        python eval.py --noise_type crop --transform_file [path-to-transforms.npy]
"""
import os
import sys
import json
import pickle
import time
import re
import numpy as np
import pandas as pd
import open3d as o3d  # Need to import before torch
from tqdm import tqdm
from scipy import sparse
from datetime import datetime
from collections import defaultdict
import torch
import torch.optim as optim

from arguments import eval_arguments
from common.math import se3_torch
from common.misc import prepare_logger
from common.torch_utils import dict_all_to_device, to_numpy
from common.metrics_util import compute_metrics, summarize_metrics, print_metrics, rte_rre
from network.model import Network
from network.matchnet import match_features
from network.DGR import Transformation, ortho2rotation, registration_ransac_based_on_feature_matching
# from dataloader.data_base import pointcloud_to_spheres
############################################### Arguments and logging
parser = eval_arguments()
_args = parser.parse_args()

if _args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    _device = torch.device('cpu')
##############################################
from dataloader.datasets import get_test_datasets_V2
from dataloader.data_base import make_open3d_point_cloud

_EPS = 1e-16
if _args.dataset_type == 'KITTI':
    RTE_THRESH = 0.6  # m
    RRE_THRESH = 5    # deg
elif _args.dataset_type == '3DMatch':
    RTE_THRESH = 0.3  # m
    RRE_THRESH = 15   # deg

BATCH_SIZE  = 1
NUM_WORKERS = 4  # 0


def visualize_pair(xyz0, xyz1, T, voxel_size):
    pcd0 = pointcloud_to_spheres(xyz0,
                                 voxel_size,
                                 np.array([0, 0, 1]),
                                 sphere_size=0.6)
    pcd1 = pointcloud_to_spheres(xyz1,
                                 voxel_size,
                                 np.array([0, 1, 0]),
                                 sphere_size=0.6)
    pcd0.transform(T)
    o3d.visualization.draw_geometries([pcd0, pcd1])


def save_data(path, data, delimiter=' '):
    if path.endswith('bin'):
        data = data.astype(np.float32)
        with open(path, 'wb') as f:
            data.tofile(f, sep='', format='%f')
    elif path.endswith('txt'):
        with open(path, 'w') as f:
            np.savetxt(f, data, delimiter=delimiter)
            # np.savetxt(f, data, fmt='%.8f', delimiter=delimiter)
    elif path.endswith('npy'):
        with open(path, 'wb') as f:
            np.save(f, data)
    else:
        print('Unknown file type: %s' % path)
        exit()


def print_stats(stats):
    succ_rate, rte, rre, avg_time, _ = stats.mean(axis=0)
    _logger.info('All result mean:')
    _logger.info('Time: {:.3f}, RTE all: {:.3f}, RRE all: {:.3f}, Success: {:.3f} %'.format(avg_time, rte, rre, succ_rate * 100))

    sel_stats = stats[stats[:, 0] > 0]
    if len(sel_stats) > 0:
        succ_rate, rte, rre, avg_time, _ = sel_stats.mean(axis=0)
        _logger.info('Success result mean:')
        _logger.info('Time: {:.3f}, RTE all: {:.3f}, RRE all: {:.3f}'.format(avg_time, rte, rre))


class HighDimSmoothL1Loss:
    def __init__(self, weights, quantization_size=1, eps=np.finfo(np.float32).eps):
        """
        weights: [B, N, 1]
        """
        self.eps = eps
        self.quantization_size = quantization_size
        self.weights = weights
        if self.weights is not None:
            self.w1 = weights.sum()

    def __call__(self, X, Y, delta=1.0):
        """
        X: [B, N, 3]
        Y: [B, N, 3]
        """
        sq_dist = torch.sum(((X - Y) / self.quantization_size)**2, dim=2, keepdim=True)  # [B, N, 1]

        # sq_dist = match_features(X, Y) / self.quantization_size**2      # [B, N, N]
        # sq_dist = sq_dist.min(dim=2, keepdim=True)[0]                   # [B, N, 1]

        use_sq_half = 0.5 * (sq_dist < delta).float()

        loss = (0.5 - use_sq_half) * (torch.sqrt(sq_dist + self.eps) - 0.5 * delta**2) + use_sq_half * sq_dist  # [B, N, 1]
        # loss = 2 * (0.5 - use_sq_half) * (torch.sqrt(sq_dist + self.eps) - 0.5 * delta**2) + use_sq_half * sq_dist  # [B, N, 1]

        if self.weights is None:
            return loss.mean()
        else:
            return (loss * self.weights).sum() / self.w1

class PtLoss:
    def __init__(self, weights=None):
        self.weights = weights

    def __call__(self, xyz_src, xyz_ref):
        # [B, C, N1, 1] - [B, C, 1, N2] => [B, N1, N2]
        # xyz_src = xyz_src.permute(0, 2, 1).contiguous()                     # [B, 3, N]
        # xyz_ref = xyz_ref.permute(0, 2, 1).contiguous()
        # dist_pc = torch.norm(xyz_ref[:, :, :, None] - xyz_src[:, :, None, :], dim=1, keepdim=False)

        # [B, N1, 1, C] - [B, 1, N2, C] => [B, N1, N2]
        # dist_pc = torch.norm(xyz_ref[:, :, None, :] - xyz_src[:, None, :, :], dim=-1, keepdim=False)
        dist_pc = match_features(xyz_src, xyz_ref)
        # dist_pc = torch.sqrt(dist_pc + _EPS)

        loss = torch.min(dist_pc, dim=2, keepdim=True)[0]     # [B, N1, 1]

        if self.weights is None:
            return loss.mean()
        else:
            weights_sum = torch.sum(self.weights, dim=2, keepdim=True)     # [B, M, 1]
            weights_norm = weights_sum / (torch.sum(weights_sum, dim=1, keepdim=True) + _EPS)  # [B, M, 1]
            loss = (loss * weights_norm).sum()
            # loss = (loss * weights_sum).sum() / (torch.sum(weights_sum, dim=1, keepdim=True) + _EPS)
            return loss

def transformation_finetune(xyz_src, xyz_ref, pose, weights=None, quantization_size=1, max_iter=1000,
                            break_threshold_ratio=1e-4, max_break_count=20):
    """
    xyz: [1, N, 3]
    pose: [1, 3/4, 4]
    weights: [1, M, 1]

    pose_optimized: [1, 3, 4]
    """
    # with torch.enable_grad():
    # if weights is not None:
    #     weights.requires_grad = False

    loss_fn = HighDimSmoothL1Loss(weights, quantization_size)

    R = pose[0, :3, :3]
    t = pose[0, :3, 3]
    transformation = Transformation(R, t).to(xyz_src.device)

    optimizer = optim.Adam(transformation.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    loss_prev = loss_fn(transformation(xyz_src), xyz_ref).item()

    break_counter = 0
    for i in range(max_iter):
        loss = loss_fn(transformation(xyz_src), xyz_ref)
        if loss.item() < 1e-7: break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(i, scheduler.get_lr(), loss.item())

        if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
            break_counter += 1
            if break_counter >= max_break_count: break

        loss_prev = loss.item()

    pose_optimized = torch.eye(3, 4)
    rot6d = transformation.rot6d.detach()
    pose_optimized[:3, :3] = ortho2rotation(rot6d)[0]
    pose_optimized[:, 3] = transformation.trans[0].detach()

    opt_result = {'iterations': i, 'loss': loss.item(), 'break_count': break_counter}
    return pose_optimized[None, :, :], opt_result

def pose_optimization(test_data, endpoints, pose_in, voxel_size=0.3):
    """ Finetune to optimise the transformation
    pose_in: [1, 3, 4]

    :return pose_optimized: [1, 3, 4]
    """
    use_tune = False
    use_icp  = False

    pose_optimized = pose_in
    corres_dist = voxel_size * 2
    if use_tune:
        weights = endpoints['perm_matrices'][-1].detach().sigmoid()[:, :, None]  # [B, J, 1]
        # w = to_numpy(weights[0, :, 0])
        # print(w.mean(), w.max(), w.min())
        # weights[weights < 0.01] = _EPS
        # weights = None

        pred_pairs = endpoints['pred_pairs'][-1].long()
        # pred_pairs = test_data['matches'][0][None, :, :]
        # xyz_src = test_data['points_src'][:, pred_pairs[0, :, 0], :3]   # [B, J, 3]
        # xyz_src = test_data['points_src'][:, :, :3]
        # xyz_ref = test_data['points_ref'][:, pred_pairs[0, :, 1], :3]
        # weights = weights[:, pred_pairs[0, :, 0], :]

        xyz_src = endpoints['pt_src'][:, pred_pairs[0, :, 0], :3]   # [B, J, 3]
        xyz_ref = endpoints['pt_ref'][:, pred_pairs[0, :, 1], :3]

        # xyz_src = endpoints['pt_src'][:, :, :3]
        # xyz_ref = endpoints['pt_ref_new'][:, :, :3]

        assert xyz_src.shape[1] == xyz_ref.shape[1]
        pose_optimized, opt_result = transformation_finetune(xyz_src,
                                                             xyz_ref,
                                                             pose=pose_optimized,
                                                             weights=weights,
                                                             quantization_size=corres_dist)

    if use_icp:
        T = np.identity(4)
        T[:3, :] = to_numpy(pose_optimized[0, :3, :])

        # src = make_open3d_point_cloud(to_numpy(endpoints['pt_src'][0, :, :3]))
        # tgt = make_open3d_point_cloud(to_numpy(endpoints['pt_ref'][0, :, :3]))
        src = make_open3d_point_cloud(to_numpy(test_data['points_src'][0, :, :3]))
        tgt = make_open3d_point_cloud(to_numpy(test_data['points_ref'][0, :, :3]))

        T = o3d.registration.registration_icp(src, tgt, corres_dist, T,
                o3d.registration.TransformationEstimationPointToPoint()).transformation

        # T = registration_ransac_based_on_feature_matching(src, tgt,
        #                                                 endpoints['feat_src'][0].cpu().numpy(),
        #                                                 endpoints['feat_ref'][0].cpu().numpy(),
        #                                                 corres_dist,
        #                                                 num_iterations=10000)
        pose_optimized = torch.from_numpy(T[None, :3, :])

    return pose_optimized.float().cuda()


def save_eval_align(pred_transforms, endpoints, metrics, summary_metrics, save_path):
    """Saves out the computed transforms
    """
    # Save transforms
    np.save(os.path.join(save_path, 'pred_transforms.npy'), pred_transforms)

    # Save endpoints if any
    for k in endpoints:
        if isinstance(endpoints[k], np.ndarray):
            np.save(os.path.join(save_path, '{}.npy'.format(k)), endpoints[k])
        else:
            with open(os.path.join(save_path, '{}.pickle'.format(k)), 'wb') as fid:
                pickle.dump(endpoints[k], fid)

    ### Save metrics
    # Write each iteration to a different worksheet.
    writer = pd.ExcelWriter(os.path.join(save_path, 'metrics.xlsx'))
    for i_iter in range(len(metrics)):
        metrics[i_iter]['r_rmse'] = np.sqrt(metrics[i_iter]['r_mse'])
        metrics[i_iter]['t_rmse'] = np.sqrt(metrics[i_iter]['t_mse'])
        metrics[i_iter].pop('r_mse')
        metrics[i_iter].pop('t_mse')
        metrics_df = pd.DataFrame.from_dict(metrics[i_iter])
        metrics_df.to_excel(writer, sheet_name='Iter_{}'.format(i_iter+1))
    writer.close()

    # Save summary metrics
    summary_metrics_float = {k: float(summary_metrics[k]) for k in summary_metrics}
    with open(os.path.join(save_path, 'summary_metrics.json'), 'w') as json_out:
        json.dump(summary_metrics_float, json_out)

    # save for futher analyze
    # metrics[-1]['err_r_deg']
    # metrics[-1]['err_t']
    # endpoints['scene']

    _logger.info('Saved evaluation results to {}'.format(save_path))


def evaluate_align(pred_transforms, data_loader):
    """ Evaluates the computed transforms against the groundtruth

    Args:
        pred_transforms: Predicted transforms (B, [iter], 3/4, 4)
        data_loader: Loader for dataset.

    Returns:
        Computed metrics (List of dicts), and summary metrics (only for last iter)
    """
    _logger.info('Evaluating transforms...')
    num_processed, num_total = 0, len(pred_transforms)

    if pred_transforms.ndim == 4:
        pred_transforms = torch.from_numpy(pred_transforms).to(_device)
    else:
        assert pred_transforms.ndim == 3 and \
               (pred_transforms.shape[1:] == (4, 4) or pred_transforms.shape[1:] == (3, 4))
        pred_transforms = torch.from_numpy(pred_transforms[:, None, :, :]).to(_device)

    metrics_for_iter = [defaultdict(list) for _ in range(pred_transforms.shape[1])]

    for data in tqdm(data_loader, leave=False):
        # needed for Chamfer distance
        data['points_src'] = data['points_src'][:, :1024, :]
        data['points_ref'] = data['points_ref'][:, :1024, :]

        dict_all_to_device(data, _device)

        batch_size = 0
        for i_iter in range(pred_transforms.shape[1]):
            batch_size = data['points_src'].shape[0]

            cur_pred_transforms = pred_transforms[num_processed:(num_processed + batch_size), i_iter, :, :]

            metrics = compute_metrics(data, cur_pred_transforms, RTE_THRESH, RRE_THRESH)
            for k in metrics:
                metrics_for_iter[i_iter][k].append(metrics[k])

        num_processed += batch_size

    for i_iter in range(len(metrics_for_iter)):
        metrics_for_iter[i_iter] = {k: np.concatenate(metrics_for_iter[i_iter][k], axis=0)
                                    for k in metrics_for_iter[i_iter]}
        summary_metrics = summarize_metrics(metrics_for_iter[i_iter])
        print_metrics(summary_metrics, title='Evaluation result (iter {})'.format(i_iter))

    return metrics_for_iter, summary_metrics


def inference_align(data_loader, model, stats_path):
    """Runs inference over entire dataset

    Args:
        data_loader (torch.utils.data.DataLoader): Dataset loader
        model (model.nn.Module): Network model to evaluate

    Returns:
        pred_transforms_all: predicted transforms (B, n_iter, 3, 4) where B is total number of instances
        endpoints_out (Dict): Network endpoints
    """
    save_matrix = True
    _logger.info('Starting transformation inference...')
    model.eval()

    total_time = 0.0
    total_rotation = []
    total_translation = []
    endpoints_out = defaultdict(list)
    pred_transforms_all = []

    method_names = ['Ours']
    stats = np.zeros((len(method_names), len(data_loader), 5))  # bool succ, rte, rre, time, drive id
    stats_id = 0
    methods_id = 0

    opt_tuple = (_args.num_reg_iter, True)
    for test_data in tqdm(data_loader):
        with torch.no_grad():
            ### compute statistics of test data
            rot_trace = test_data['transform_gt'][:, 0, 0] + test_data['transform_gt'][:, 1, 1] + \
                        test_data['transform_gt'][:, 2, 2]
            rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1),
                                    min=-1 + _EPS, max=1 - _EPS)) * 180.0 / np.pi
            total_rotation.append(np.abs(to_numpy(rotdeg)))
            transmag = test_data['transform_gt'][:, :, 3].norm(dim=-1)
            total_translation.append(to_numpy(transmag))

            ### run
            dict_all_to_device(test_data, _device)

            time_before = time.time()
            pred_transforms, endpoints = model(test_data, opt_tuple)
            time_after = time.time() - time_before
            total_time += time_after

        assert pred_transforms[-1].shape[0] == 1  # batch size must be 1

        pose_in = pred_transforms[-1].detach()
        pose_optimized = pose_optimization(test_data, endpoints, pose_in)
        pred_transforms.append(pose_optimized)

        # list of array [B, 3, 4] => array of [B, n_iter, 3, 4]
        if isinstance(pred_transforms[-1], torch.Tensor):
            pred_transforms = to_numpy(torch.stack(pred_transforms, dim=1))
        else:
            pred_transforms = np.stack(pred_transforms, axis=1)
        pred_transforms_all.append(pred_transforms)

        ### Saves match matrix. We only save the top matches to save storage/time.
        # However, this still takes quite a bit of time to save. Comment out if not needed.
        # if 'perm_matrices' in endpoints and save_matrix:
            # perm_matrices = to_numpy(torch.stack(endpoints['perm_matrices'], dim=1))
            # thresh = np.percentile(perm_matrices, 99.9, axis=[2, 3])  # Only retain top 0.1% of entries
            # below_thresh_mask = perm_matrices < thresh[:, :, None, None]
            # perm_matrices[below_thresh_mask] = 0.0

            # for i_data in range(perm_matrices.shape[0]):
            #     sparse_perm_matrices = []
            #     for i_iter in range(perm_matrices.shape[1]):
            #         sparse_perm_matrices.append(sparse.coo_matrix(perm_matrices[i_data, i_iter, :, :]))
            #     endpoints_out['perm_matrices'].append(sparse_perm_matrices)

        ### compute statistics using the last pred
        T_gt = to_numpy(test_data['transform_gt'])
        T_pred = to_numpy(pose_optimized)
        bs = len(T_gt)
        for i in range(bs):
            cur_id = stats_id + i
            stats[methods_id, cur_id, :3] = rte_rre(T_pred[i, :, :], T_gt[i, :, :], RTE_THRESH, RRE_THRESH)
            stats[methods_id, cur_id, 3] = time_after
            stats[methods_id, cur_id, 4] = test_data['others'][i]['seq'] # {'others': [[{...}, {...}, ...][...]]}
            # _logger.info(f"{method_names[i]}: failed") if stats[methods_id, cur_id, 0] == 0
        stats_id += bs

        del test_data, pred_transforms, endpoints

    assert len(data_loader) == stats_id

    _logger.info('Total inference time: {:.3f}s'.format(total_time))

    total_rotation = np.concatenate(total_rotation, axis=0)
    _logger.info('Rotation range in test data: {:.3f}(avg), {:.3f}(max)'.format(np.mean(total_rotation), np.max(total_rotation)))

    _logger.info(f'Saving the stats to: {stats_path}')
    np.savez(stats_path, stats=stats, names=method_names)
    print_stats(stats[methods_id])

    pred_transforms_all = np.concatenate(pred_transforms_all, axis=0)
    return pred_transforms_all, endpoints_out


def inference_feat(data_loader, model, save_path):
    """Runs inference over entire dataset

    Args:
        data_loader (torch.utils.data.DataLoader): Dataset loader
        model (model.nn.Module): Network model to evaluate

    """
    _logger.info('Starting feat inference...')
    model.eval()
    total_time = 0.0

    count = 0
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for test_data in tqdm(data_loader):
            dict_all_to_device(test_data, _device)

            time_before = time.time()
            _, endpoints = model(test_data)
            total_time += time.time() - time_before

            if count % 10 == 0:
                test_data['points_src'][:, :, :3] = se3_torch.transform(test_data['transform_gt'], test_data['points_src'][:, :, :3])

                endpoints['pt_ref'] = endpoints['pt_ref'].permute(0, 2, 1).contiguous()  # [B, N, 3]
                endpoints['pt_src'] = endpoints['pt_src'].permute(0, 2, 1).contiguous()
                endpoints['pt_src'] = se3_torch.transform(test_data['transform_gt'], endpoints['pt_src'])

                for k in ['ref', 'src']:
                    pt = torch.cat([endpoints['pt_%s' % k], endpoints['score_%s' % k].unsqueeze(2)], dim=2)

                    # save a sample of a batch
                    raw_pt = test_data['points_%s' % k][:, :, :3]
                    save_data(os.path.join(save_path, '%06d_%s_raw.txt' % (count, k)),
                                raw_pt[0, :, :].detach().cpu().numpy())
                    save_data(os.path.join(save_path, '%06d_%s_pt.txt' % (count, k)),
                                pt[0, :, :].detach().cpu().numpy())

                    # pcd1 = o3d.geometry.PointCloud()
                    # pcd1.points = o3d.utility.Vector3dVector(pt1)
                    # o3d.io.write_point_cloud(os.path.join(save_path, '%06d_1_%s.pcd' % (i, k)), pcd1)

            count += len(endpoints['pt_ref'])
    _logger.info('Total inference time: {:.3f}s'.format(total_time))


def inference_label(data_loader, model, save_path):
    """Runs inference over entire dataset

    Args:
        data_loader (torch.utils.data.DataLoader): Dataset loader
        model (model.nn.Module): Network model to evaluate
    """
    _logger.info('Starting feat inference...')
    model.eval()
    total_time = 0.0

    count = 0
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for test_data in tqdm(data_loader):
            dict_all_to_device(test_data, _device)

            time_before = time.time()
            _, endpoints = model(test_data)
            total_time += time.time() - time_before

            endpoints['labels_src'] = test_data['labels_src']
            endpoints['labels_ref'] = test_data['labels_ref']
            test_loss, test_acc = model.loss_label_fun(endpoints)

            if count % 10 == 0:
                # test_data['points_src'][:, :, :3] = se3_torch.transform(test_data['transform_gt'], test_data['points_src'][:, :, :3])

                endpoints['pt_src'] = endpoints['pt_src'].permute(0, 2, 1).contiguous()  # [B, N, 3]
                endpoints['pt_ref'] = endpoints['pt_ref'].permute(0, 2, 1).contiguous()
                endpoints['pt_src'] = se3_torch.transform(test_data['transform_gt'], endpoints['pt_src'])

                for k in ['ref', 'src']:
                    pt = endpoints['pt_%s' % k]
                    # score = endpoints['score_%s' % k][:, :, None]

                    pred_labels = torch.argmax(endpoints['logits_%s' % k], dim=1)
                    pred_labels = pred_labels[:, :, None] + 1  # 0-'unlabeled' is not included

                    # pt = test_data['points_%s' % k][:, :, :3]
                    # gt_labels = test_data['labels_%s' % k][:, :, None]

                    # pt_l = torch.cat([pt, pred_labels.float(), gt_labels.float()], dim=2)
                    pt_l = torch.cat([pt, pred_labels.float()], dim=2)
                    # pt_l = torch.cat([pt, pred_labels.float(), score], dim=2)
                    # pt_l = torch.cat([pt, score], dim=2)

                    save_data(os.path.join(save_path, '%06d_%s.txt' % (count, k)),
                                pt_l[0, :, :].detach().cpu().numpy())

            count += len(endpoints['pt_ref'])
    _logger.info('Total inference time: {:.3f}s'.format(total_time))

    mean_iou, iou_list, mean_acc = model.loss_label_fun.semantic_metric()

    _logger.info('Validation accuracy: {:.3f}'.format(mean_acc))
    _logger.info('Mean IoU: {:.1f}'.format(mean_iou * 100))
    s = 'IoU: '
    for iou_tmp in iou_list:
        s += '{:5.2f}|'.format(100 * iou_tmp)
    _logger.info(s)



if __name__ == '__main__':
    datetime_str = None
    if _args.resume is not None:
        assert os.path.exists(_args.resume)
        pattern = re.compile(r'\d{6}_\d{6}')
        patterns = pattern.findall(_args.resume)
        datetime_str = patterns[0] if len(patterns) > 0 else None
    if datetime_str is None:
        datetime_str = datetime.now().strftime("%y%m%d_%H%M%S")

    if _args.resume is not None:
        epoch = (_args.resume).split('/')[-1]
        pattern = re.compile(r'\d+')
        patterns = pattern.findall(epoch)
        epoch = patterns[0] if len(patterns) > 0 else '000'
    else:
        epoch = '000'

    folder = 'eval_' + _args.pipeline + '_' + _args.dataset_type + '_' + datetime_str
    save_path = os.path.join(_args.eval_save_path, folder, epoch)
    _logger, _log_path = prepare_logger(_args, log_path=save_path)

    stats_path = os.path.join(save_path, f'stats_{_args.dataset_type}_{datetime_str}_{epoch}')

    test_dataset = get_test_datasets_V2(_args)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=BATCH_SIZE,
                                            collate_fn=test_dataset.collate_fn,
                                            shuffle=False,
                                            num_workers=NUM_WORKERS)

    if _args.transform_file is not None:
        _logger.info('Loading from precomputed transforms: {}'.format(_args.transform_file))
        pred_transforms = np.load(_args.transform_file)
        endpoints = {}
    else:
        assert _args.resume is not None
        _logger.info('building model...')
        model = Network(_args)
        model.to(_device)
        # for k, v in model.state_dict().items():
        #     print(k)

        model.load_state_dict(torch.load(_args.resume)['state_dict'])
        _logger.info('loaded pre-train model from: {}'.format(_args.resume))

        if _args.pipeline == 'feat':
            inference_feat(test_loader, model, save_path)
            sys.exit(0)

        elif _args.pipeline == 'label':
            inference_label(test_loader, model, save_path)
            sys.exit(0)

        pred_transforms, endpoints = inference_align(test_loader, model, stats_path)

    # Compute evaluation matrices
    eval_metrics, summary_metrics = evaluate_align(pred_transforms, data_loader=test_loader)

    save_eval_align(pred_transforms, endpoints, eval_metrics, summary_metrics, save_path)
    _logger.info('Finished')

