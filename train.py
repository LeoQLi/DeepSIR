#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, List
from tensorboardX import SummaryWriter
from collections import defaultdict
from matplotlib.pyplot import cm as colormap

from arguments import train_arguments
from common.misc import prepare_logger
from common.math import se3_torch
from common.metrics_util import compute_metrics, summarize_metrics, print_metrics
from common.torch_utils import dict_all_to_device, TorchDebugger, CheckPointManager
##############################################
# Set up arguments and logging
parser = train_arguments()
_args = parser.parse_args()
_logger, _log_path = prepare_logger(_args)

if _args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    _device = torch.device('cpu')
##############################################
from dataloader.datasets import get_train_datasets_V2
from network.model import Network


def update_learning_rate(optimizer, ratio=0.95, lr_clip=1e-4):
    lr0 = optimizer.param_groups[0]['lr']
    lr = lr0 * ratio
    if lr < lr_clip:
        lr = lr_clip

    if lr0 != lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    _logger.info('Update learning rate: %f => %f' % (lr0, lr))


def save_summaries(writer, pred_trans: List=None, endpoints: Dict=None, losses: Dict=None, metrics: Dict=None, step=0):
    """ Save tensorboard summaries of scan alignment
    pred_trans: computed transformation, list [[B, 3, 4], ...(iter)]
    endpoints:  src, ref point clouds [B, N, 3]
                perm_matrices [B, J, K]
                alpha, beta
    """
    BLUE = [0, 61, 124]
    ORANGE = [239, 124, 0]
    subset = [0] if endpoints['pt_src'].shape[0] == 1 else [0, 1]   # only save the first one or two sample
    num_sub = _args.num_sub if 0 < _args.num_sub < 1024 else 1024        # only save the first 1024 points of each sample

    points_src = endpoints['pt_src'][subset, :num_sub, :3]   # [B, N, 3]
    points_ref = endpoints['pt_ref'][subset, :num_sub, :3]   # [B, N, 3]

    with torch.no_grad():
        # Save point cloud at iter0, iter1 and after last iter
        concat_cloud_input = torch.cat((points_src, points_ref), dim=1)
        colors = torch.from_numpy(np.concatenate([np.tile(ORANGE, (*points_src.shape[0:2], 1)),
                                                np.tile(BLUE, (*points_ref.shape[0:2], 1))], axis=1) )

        writer.add_mesh('iter_0(no-trans)', vertices=concat_cloud_input, colors=colors, global_step=step)

        iters_to_save = [0, len(pred_trans)-1] if (pred_trans is not None and len(pred_trans) > 1) else [0]

        if pred_trans is not None:
            for i_iter in iters_to_save:
                src_transformed_first = se3_torch.transform(pred_trans[i_iter][subset, ...], points_src)
                concat_cloud_first = torch.cat((src_transformed_first, points_ref), dim=1)
                writer.add_mesh('iter_{}'.format(i_iter+1), vertices=concat_cloud_first,
                                    colors=colors, global_step=step)

        # if 'perm_matrices' in endpoints:
        #     color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('coolwarm'))
        #     for i_iter in iters_to_save:
        #         perm_matrix = endpoints['perm_matrices'][i_iter][subset, :num_sub, :num_sub]
        #         ref_weights = torch.sum(perm_matrix, dim=1)
        #         ref_colors = color_mapper.to_rgba(ref_weights.detach().cpu().numpy())[..., :3]
        #         writer.add_mesh('ref_weights_{}'.format(i_iter), vertices=points_ref,
        #                         colors=torch.from_numpy(ref_colors) * 255, global_step=step)

        #     for i_iter in range(len(endpoints['perm_matrices'])):
        #         src_weights = torch.sum(endpoints['perm_matrices'][i_iter], dim=2)
        #         ref_weights = torch.sum(endpoints['perm_matrices'][i_iter], dim=1)
        #         writer.add_histogram('src_weights_{}'.format(i_iter), src_weights, global_step=step)
        #         writer.add_histogram('ref_weights_{}'.format(i_iter), ref_weights, global_step=step)

        # if 'alpha' in endpoints and 'beta' in endpoints:
        #     writer.add_scalar('alpha', endpoints['alpha'], step)
        #     writer.add_scalar('beta', endpoints['beta'], step)

        # Write losses and metrics
        if losses is not None:
            for l in losses:
                writer.add_scalar('losses/{}'.format(l), losses[l], step)
        if metrics is not None:
            for m in metrics:
                writer.add_scalar('metrics/{}'.format(m), metrics[m], step)

        writer.flush()


def validate_align(data_loader, model, summary_writer, step):
    """ Perform a single validation run, and saves results into tensorboard summaries
    """
    print('\n')
    _logger.info('************** Validation ***************')

    ###### Validate with all data
    total_time = 0.0
    opt_tuple = (_args.num_reg_iter, True)
    with torch.no_grad():
        all_val_losses = defaultdict(list)
        all_val_metrics_np = defaultdict(list)

        for val_data in tqdm(data_loader, total=len(data_loader), leave=True, ncols=100, position=0):
            dict_all_to_device(val_data, _device)
            time_before = time.time()
            pred_transforms, endpoints = model(val_data, opt_tuple)
            total_time += time.time() - time_before

            ### loss
            endpoints['matches'] = val_data['matches']
            endpoints['transform_gt'] = val_data['transform_gt']
            endpoints['transform_pred'] = pred_transforms
            val_losses = model.loss_align_fun(endpoints, reduction='none')

            ### metric
            data_dict = {'transform_gt': val_data['transform_gt'],
                           'points_src': endpoints['pt_src'],
                           'points_ref': endpoints['pt_ref'] }
            # use the last output (R|t) to compute error
            val_metrics = compute_metrics(data_dict, pred_transforms[-1], _args.rte_thresh, _args.rre_thresh)

            for k in val_losses:
                all_val_losses[k].append(val_losses[k])
            for k in val_metrics:
                all_val_metrics_np[k].append(val_metrics[k])

        #################################################################
        #     hit_ratio_meter.update(is_correct.sum().item() / len(is_correct))
        #     regist_rre_meter.update(rot_error.squeeze())
        #     regist_rte_meter.update(trans_error.squeeze())
        #     # Compute success
        #     success = (trans_error < self.config.success_rte_thresh) * (
        #         rot_error < self.config.success_rre_thresh) * valid_mask
        #     regist_succ_meter.update(success.float())

        #     target = torch.from_numpy(is_correct).squeeze()
        #     neg_target = (~target).to(torch.bool)
        #     pred = weights > 0.5  # TODO thresh
        #     pred_on_pos, pred_on_neg = pred[target], pred[neg_target]
        #     tp += pred_on_pos.sum().item()
        #     fp += pred_on_neg.sum().item()
        #     tn += (~pred_on_neg).sum().item()
        #     fn += (~pred_on_pos).sum().item()
        # precision = tp / (tp + fp + eps)
        # recall = tp / (tp + fn + eps)
        # f1 = 2 * (precision * recall) / (precision + recall + eps)
        # tpr = tp / (tp + fn + eps)
        # tnr = tn / (tn + fp + eps)
        # balanced_accuracy = (tpr + tnr) / 2

        # logging.info(' '.join([
        #     f"Hit Ratio: {hit_ratio_meter.avg:.4f}, Precision: {precision}, Recall: {recall}, F1: {f1}, ",
        #     f"TPR: {tpr}, TNR: {tnr}, BAcc: {balanced_accuracy}, ",
        #     f"RTE: {regist_rte_meter.avg:.3e}, RRE: {regist_rre_meter.avg:.3e}",
        #     f"Succ rate: {regist_succ_meter.avg:3e}",
        # ]))
        ######################################################################

        _logger.info('Total validate time: {:.3f}s'.format(total_time))
        all_val_losses = {k: torch.cat(all_val_losses[k]) for k in all_val_losses}
        mean_val_losses = {k: torch.mean(all_val_losses[k]) for k in all_val_losses}
        all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}

        ###### Print validation results and save summary
        summary_metrics = summarize_metrics(all_val_metrics_np)
        losses_by_iteration = None
        if '{}_0'.format(_args.loss_type) in mean_val_losses:
            losses_by_iteration = torch.stack([mean_val_losses['{}_{}'.format(_args.loss_type, i)]
                                                    for i in range(_args.num_reg_iter)] ).cpu().numpy()
        print_metrics(summary_metrics, losses_by_iteration, 'Validation results')


        ##### Select random and worst data instances (batch) and save to summary
        # ================================================================
        rand_idx = random.randint(0, all_val_losses['total'].shape[0] - 1)
        worst_idx = torch.argmax(all_val_losses['{}_{}'.format(_args.loss_type, _args.num_reg_iter - 1)]).cpu().item()
        indices_to_rerun = [rand_idx, worst_idx]

        del val_data, endpoints, pred_transforms, data_dict, val_metrics, all_val_losses    # TODO

        # Re-validate with the chosen ones: the rand and the worst
        data_to_rerun = []
        for i in indices_to_rerun:
            data_to_rerun.append(data_loader.dataset[i])
        # form a batch with 2 samples
        data_to_rerun = data_loader.collate_fn(data_to_rerun)
        dict_all_to_device(data_to_rerun, _device)
        pred_transforms, endpoints = model(data_to_rerun, opt_tuple)

        save_summaries(summary_writer, pred_trans=pred_transforms, endpoints=endpoints,
                        losses=mean_val_losses, metrics=summary_metrics, step=step)

        # error = summary_metrics['chamfer_dist']
        # error = summary_metrics['err_t_mean'] * 4.0 + summary_metrics['err_r_deg_mean']
        succ_rate = summary_metrics['succ']
    return succ_rate


def validate_feat(data_loader, model, summary_writer, step):
    """ Perform a single validation run, and saves results into tensorboard summaries
    """
    print('\n')
    _logger.info('************** Validation ***************')

    total_time = 0.0
    all_val_loss = []
    all_val_acc = []
    with torch.no_grad():
        for val_data in tqdm(data_loader, total=len(data_loader), leave=True, ncols=100, position=0):
            dict_all_to_device(val_data, _device)

            time_before = time.time()
            _, endpoints = model(val_data)
            total_time += time.time() - time_before

            ### compute losses
            endpoints['transform_gt'] = val_data['transform_gt']
            # endpoints['points_src'] = val_data['points_src'][:, :, :3].permute(0, 2, 1).contiguous()  # [B, 3, N]
            # endpoints['points_ref'] = val_data['points_ref'][:, :, :3].permute(0, 2, 1).contiguous()  # [B, 3, N]

            val_loss, val_acc = model.loss_feat_fun(endpoints)

            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

        loss = torch.mean(torch.stack(all_val_loss, dim=0))
        acc = torch.mean(torch.stack(all_val_acc, dim=0))

    _logger.info('Total validate time: {:.3f}s'.format(total_time))
    _logger.info('Validation loss: {:.3f}'.format(loss.item()))
    _logger.info('Validation acc: {:.3f}'.format(acc.item()))

    return loss.item(), acc.item()


def validate_label(data_loader, model, summary_writer, step):
    """ Perform a single validation run, and saves results into tensorboard summaries
    """
    print('\n')
    _logger.info('************** Validation ***************')

    total_time = 0.0
    all_val_loss = []
    all_val_acc = []
    with torch.no_grad():
        for val_data in tqdm(data_loader, total=len(data_loader), leave=True, ncols=100, position=0):
            dict_all_to_device(val_data, _device)

            time_before = time.time()
            _, endpoints = model(val_data)
            total_time += time.time() - time_before

            ### compute losses
            # endpoints['transform_gt'] = val_data['transform_gt']
            endpoints['labels_src'] = val_data['labels_src']
            endpoints['labels_ref'] = val_data['labels_ref']
            val_loss, val_acc = model.loss_label_fun(endpoints)

            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

        loss = torch.mean(torch.stack(all_val_loss, dim=0))
        # acc = torch.mean(torch.stack(all_val_acc, dim=0))

    _logger.info('Total validate time: {:.3f}s'.format(total_time))
    _logger.info('Validation loss: {:.3f}'.format(loss.item()))

    mean_iou, iou_list, mean_acc = model.loss_label_fun.semantic_metric()

    _logger.info('Validation accuracy: {:.3f}'.format(mean_acc))
    _logger.info('Mean IoU: {:.1f}'.format(mean_iou * 100))
    s = 'IoU: '
    for iou_tmp in iou_list:
        s += '{:5.2f}|'.format(100 * iou_tmp)
    _logger.info(s)

    return loss.item(), mean_iou, mean_acc


def main(_args):
    # train_set, val_set = get_train_datasets(_args)
    train_set, val_set = get_train_datasets_V2(_args)

    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=_args.batch_size,
                                                collate_fn=train_set.collate_fn,
                                                shuffle=True, # drop_last=True,
                                                num_workers=_args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set,
                                                batch_size=_args.batch_size,
                                                collate_fn=val_set.collate_fn,
                                                shuffle=False,
                                                num_workers=_args.num_workers)

    _logger.info('Building model ...')
    my_model = Network(_args)
    my_model.to(_device)

    saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'model'), keep_checkpoint_every_n_hours=1.0)
    optimizer = optim.Adam(my_model.parameters(), lr=_args.lr)

    if _args.resume is not None:
        if not os.path.exists(_args.resume):
            _logger.error('Model path error!')
            sys.exit(0)
        _logger.info('Loading model from: %s'%_args.resume)

        if _args.load_model_all:
            _logger.info('### Loading all parameters of the provided model ###')
            global_step = saver.load(_args.resume, my_model, optimizer)
            # my_model.load_state_dict(torch.load(_args.resume)['state_dict'], strict=False)

            _logger.info('Set learning rate to: %f' % optimizer.param_groups[0]['lr'])

            # if 'optimizer' in pretrained_dict:
            #     optimizer.load_state_dict(pretrained_dict['optimizer'])
            #     _logger.info('Loaded optimizer parameters')

            # if 'lr' in pretrained_dict:
            #     lr = pretrained_dict['lr']
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            #     _logger.info('Set learning rate to: %f' % lr)
        else:
            # only load existing model parameters
            pretrained_dict = torch.load(_args.resume, map_location=torch.device('cpu'))
            model_dict = my_model.state_dict()
            # filter out unnecessary keys
            load_dict = {k: v for k, v in pretrained_dict['state_dict'].items()  \
                            if (k in model_dict) and (v.size() == model_dict[k].size())}
            # overwrite entries in the existing state dict
            model_dict.update(load_dict)
            my_model.load_state_dict(model_dict)

            for k, v in load_dict.items():
                _logger.info(k)
            _logger.info('Number of loaded model dict from pretrained model: %d\n' % len(load_dict))

    if _args.debug:
        _logger.info('\n*************************************\n')
        _logger.info(my_model)
        _logger.info('\n*************************************\n')
        for param in my_model.parameters():
            if param.requires_grad:
                _logger.info(param.shape)
        _logger.info('\n*************************************\n')
        _logger.info("Model size: %i" % sum(param.numel() for param in my_model.parameters() if param.requires_grad))
        _logger.info('\n*************************************\n')

    # summary writer
    train_writer = SummaryWriter(os.path.join(_log_path, 'train'), flush_secs=10)
    val_writer = SummaryWriter(os.path.join(_log_path, 'val'), flush_secs=10)

    # only for debugging as the different tests will slow down your program execution
    torch.autograd.set_detect_anomaly(_args.debug)
    my_model.train()

    steps_per_epoch = len(train_loader)
    if _args.summary_every < 0:
        _args.summary_every = abs(_args.summary_every) * steps_per_epoch
    if _args.validate_every < 0:
        _args.validate_every = abs(_args.validate_every) * steps_per_epoch

    epoch = 0
    global_step = 0
    opt_tuple = (_args.num_train_reg_iter, False)
    while True:
        _logger.info('Begin epoch {} (steps {} - {})'.format(epoch, global_step, global_step + len(train_loader)))
        pbar = tqdm(total=len(train_loader), ncols=100, position=0)

        for train_data in train_loader:
            # zero the parameter gradients
            optimizer.zero_grad()
            global_step += 1

            ################ training ################
            dict_all_to_device(train_data, _device)
            pred_transforms, endpoints = my_model(train_data, opt_tuple)

            ################ Compute loss ################
            endpoints['matches'] = train_data['matches']
            endpoints['transform_gt'] = train_data['transform_gt']
            endpoints['transform_pred'] = pred_transforms
            if _args.pipeline == 'align':
                # the total loss during training
                loss = my_model.loss_align_fun(endpoints, reduction='mean')['total']

                # if _args.feat_loss_weight > 0.0:
                #     loss = loss + torch.sum(torch.stack(endpoints['feat_loss']), dim=0) * _args.feat_loss_weight

                if not np.isfinite(loss.item()):
                    logging.info('Abort! Loss is infinite.')
                    continue

            elif _args.pipeline == 'feat':
                # endpoints['points_src'] = train_data['points_src'][:, :, :3].permute(0, 2, 1).contiguous()  # [B, 3, N]
                # endpoints['points_ref'] = train_data['points_ref'][:, :, :3].permute(0, 2, 1).contiguous()  # [B, 3, N]
                loss, _ = my_model.loss_feat_fun(endpoints)

            elif _args.pipeline == 'label':
                endpoints['labels_src'] = train_data['labels_src']
                endpoints['labels_ref'] = train_data['labels_ref']
                loss, _ = my_model.loss_label_fun(endpoints)

            if _args.debug:
                with TorchDebugger():
                    loss.backward()
            else:
                loss.backward()

            ################ Optimize ################
            if _args.pipeline == 'align':
                # Only update the parameters if there were no problems in the forward pass (mostly SVD)
                # Check if any of the gradients is NaN
                backprop_flag = False
                for name, param in my_model.named_parameters():
                    if param.grad is not None and torch.any(torch.isnan(param.grad)):
                        optimizer.zero_grad()
                        _logger.info('Gradients include NaN values. Parameters will not be updated.')
                        backprop_flag = True
                        break
                if not (backprop_flag or endpoints['invalid_gradient']):
                    optimizer.step()
            else:
                optimizer.step()

            ################ Save logs ################
            if global_step % _args.summary_every == 0 and global_step > 0 and _args.pipeline == 'align':
                save_summaries(train_writer, pred_trans=pred_transforms, endpoints=endpoints,
                               losses={'total': loss}, step=global_step)

            pbar.set_description('Loss:{:.3g}'.format(loss))
            pbar.update(1)

            del train_data, endpoints, pred_transforms, loss    # TODO
            # torch.cuda.empty_cache()

            ################ validation ################
            if global_step % _args.validate_every == 0 and global_step > 0:
                my_model.eval()

                if _args.pipeline == 'align':
                    val_acc = validate_align(val_loader, my_model, val_writer, global_step)
                    # val_acc = -1 * val_error

                elif _args.pipeline == 'feat':
                    val_error, _ = validate_feat(val_loader, my_model, val_writer, global_step)
                    val_acc = -1 * val_error

                elif _args.pipeline == 'label':
                    val_error, mean_iou, mean_acc = validate_label(val_loader, my_model, val_writer, global_step)
                    val_acc = mean_iou

                save_step = epoch if _args.validate_every > steps_per_epoch else global_step
                saver.save(my_model, optimizer, step=save_step, score=val_acc)
                _logger.info('Summary information saved to: %s\n' % _log_path)

                my_model.train()

        pbar.close()
        epoch += 1

        # decrease the learning rate
        if epoch % _args.lr_decay_epoch == 0 and epoch > 0:
            update_learning_rate(optimizer, ratio=_args.lr_decay_ratio)



if __name__ == '__main__':
    # Prepares the folder for saving files, logs
    code_dir = os.path.join(_log_path, 'code')
    os.makedirs(code_dir, exist_ok=True)
    os.system('cp %s %s' % ('*.py', code_dir))
    os.system('cp -r %s %s' % ('dataloader', code_dir))
    os.system('cp -r %s %s' % ('common', code_dir))
    os.system('cp -r %s %s' % ('network', code_dir))

    _logger.info("Welcome! Happy training :-)")
    _logger.info('PID: %s', str(os.getpid()))

    main(_args)
