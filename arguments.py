"""Common arguments for train and evaluation"""
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

def net_arguments():
    """Arguments used for both training and testing"""
    parser = argparse.ArgumentParser(add_help=False)

    # Logging
    parser.add_argument('--logdir', type=str, default='./logs',
                        help='Directory to store logs, summaries, checkpoints.')
    parser.add_argument('--dev', action='store_true',
                        help='If true, will ignore logdir and log to ../logdev instead')
    parser.add_argument('--name', type=str, default=None,
                        help='Prefix to add to logging directory')
    parser.add_argument('--debug', action='store_true',
                        help='If set, will enable autograd anomaly detection')
    # settings for input data_loader
    parser.add_argument('--dataset_path', type=str, default='../data/',
                        help='path to the processed dataset.')
    parser.add_argument('--dataset_type', default='KITTI', choices=['3DMatch', 'Modelnet', 'Oxford', 'KITTI'],
                        help='Which dataset to use.')
    parser.add_argument('--feat_len', type=int, default=4,
                        help='3 (xyz), 4 (xyz+reflectance)')
    parser.add_argument('--pipeline', type=str, default='align', choices=['feat', 'align', 'label'],
                        help='feat: detection and description. align: scan registration')
    parser.add_argument('--use_ppf', type=str2bool, default=False)
    parser.add_argument('--voxel_size', type=float, default=0.3,
                        help='The voxel size to sub-sample point clouds.')
    parser.add_argument('--positive_pair_radius_multiplier', type=float, default=3.0,
                        help='The radius multiplier to find positive pairs.')
    # data transformation
    parser.add_argument('--rot_mag', default=45.0, type=float,
                        help='Maximum magnitude of rotation perturbation (in degrees)')
    parser.add_argument('--xy_rot_scale', type=float, default=0.1,
                        help='Rotation discount in XY axis (non-upward direction)')
    parser.add_argument('--trans_mag', default=2.0, type=float,
                        help='Maximum magnitude of translation perturbation')
    # feature learning
    parser.add_argument('--thres_radius', type=float, default=-1.0,
                        help='Threshold to determine positive point match in loss')
    parser.add_argument('--det_loss_weight', type=float, default=1.0,
                        help='The weight of detection loss')
    parser.add_argument('--chamfer_loss_weight', type=float, default=0.0,
                        help='The weight of chamfer loss')
    parser.add_argument('--feat_loss_weight', type=float, default=0.0,
                        help='The weight of feature alignment loss')
    # Alignment
    parser.add_argument('--loss_type', type=str, choices=['mse', 'mae'], default='mae',
                        help='The point cloud pairs distance loss')
    parser.add_argument('--wt_ptDist_loss', type=float, default=1.0,
                        help='Weight point cloud pairs distance loss')
    parser.add_argument('--wt_inlier_loss', type=float, default=1.0,
                        help='Weight to encourage inliers')
    parser.add_argument('--wt_pose_loss', type=float, default=0.0,
                        help='Weight for rotation and translation error')
    parser.add_argument('--clip_weight_thresh', type=float, default=0.0,
                        help='Truncate the low matching probability')
    parser.add_argument('--loss_discount_factor', type=float, default=0.5,
                        help='Discount factor to compute the loss')
    parser.add_argument('--no_slack', action='store_true',
                        help='If set, will not have a slack column')
    parser.add_argument('--num_sk_iter', type=int, default=5,
                        help='Number of inner iterations used in sinkhorn normalization')
    parser.add_argument('--num_train_reg_iter', type=int, default=2,
                        help='Number of outer iterations used for registration (for training)')
    parser.add_argument('--num_reg_iter', type=int, default=5,
                        help='Number of outer iterations used for registration (for evaluation)')
    # Net settings
    parser.add_argument('--num_points', type=int, default=18000,
                        help='the number of points in point-cloud')
    parser.add_argument('--num_sub', type=int, default=-1,
                        help='top k points used for matching, in integer')
    parser.add_argument('--num_knn', type=int, default=16,
                        help='number of k nearest points')
    parser.add_argument('--sub_sampling_ratio', default=[4,4,4,4],
                        help='Sampling Ratio of random sampling at each layer')
    parser.add_argument('--d_out', default=[16,64,128,256],
                        help='feature dimension of each layer')
    parser.add_argument('--out_feat_dim', type=int, default=64,
                        help='Feature dimension (to compute distances on). Other numbers will be scaled accordingly')
    # Training parameters
    parser.add_argument('-gpu', '--gpu', type=int, default=1,
                        help='GPU to use, ignored if no GPU is present. Set to negative to use CPU.')
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
                        help='batch size for training and validation.')
    parser.add_argument('-nv', '--num_val', type=int, default=-1,
                        help='Number of data for validation, <=0 for all data')
    parser.add_argument('--resume', type=str, default='./logs/201012_162910/ckpt/model_21.pth',
                        help='Pretrained network to load from. Optional for train, required for inference.')
    parser.add_argument('--load_model_all', action='store_true',
                        help='If set, will load all parameters from the pretrained model.')
    return parser


def train_arguments():
    """Used only for training"""
    parser = argparse.ArgumentParser(parents=[net_arguments()])

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate during training')
    parser.add_argument('--lr_decay_epoch', type=int, default=4,
                        help='Frequency of learning rate decay')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.98,
                        help='Ratio of learning rate decay')
    parser.add_argument('-su', '--summary_every', type=int, default=3000,
                        help='Frequency of saving summary (number of steps if positive, number of epochs if negative)')
    parser.add_argument('-v', '--validate_every', type=int, default=-2,
                        help='Frequency of evaluation (number of steps if positive, number of epochs if negative).'
                            'Also saves checkpoints at the same interval')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data_loader loader')
    parser.add_argument('--rte_thresh', type=float, default=0.6,
                        help='Success if the RTE below this (m)')
    parser.add_argument('--rre_thresh', type=float, default=5,
                        help='Success if the RTE below this (degree)')

    parser.description = 'Train'
    return parser


def eval_arguments():
    """Used during evaluation"""
    parser = argparse.ArgumentParser(parents=[net_arguments()])

    # Provided transforms
    parser.add_argument('--transform_file', type=str,
                        help='If provided, will use transforms from this provided pickle file')
    # Save out evaluation data_loader for further analysis
    parser.add_argument('--eval_save_path', type=str, default='./out/',
                        help='Output data_loader to save evaluation results')

    parser.description = 'Evaluation'
    return parser
