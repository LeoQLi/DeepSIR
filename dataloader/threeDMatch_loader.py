import os, sys
import copy
import pickle
import logging
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from dataloader.data_base import apply_transform, make_open3d_point_cloud, DataBase

_logger = logging.getLogger('3DMatch_Dataset')


def read_trajectory(filename, dim=4):
    class CameraPose:
        def __init__(self, meta, mat):
            self.metadata = meta
            self.pose = mat

        def __str__(self):
            return 'metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
                "pose : " + "\n" + np.array_str(self.pose)

    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(dim, dim))
            for i in range(dim):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


class ThreeDMatch(DataBase):
    def __init__(self, args, split='train'):
        super(ThreeDMatch, self).__init__(args)
        assert split in ['train', 'val', 'test']
        self.split = split

        self.root_path = os.path.join(args.dataset_path, '3dmatch_train_val')
        self.test_path = os.path.join(args.dataset_path, 'test')
        self.files = []
        self.overlap_thres = 0.3
        self.voxel_size = 0.03
        args.thres_radius = self.voxel_size * args.positive_pair_radius_multiplier

        self.num_val = args.num_val
        self.num_points = args.num_points
        self.random_rotation = True
        self.rotation_range = 90
        self.ramdom_jitter = True
        self.jitter_scale = 0.005
        self.random_scale = True
        self.max_scale = 1.2
        self.min_scale = 0.8

        if self.split == 'val':
            self.random_rotation = True
            self.ramdom_jitter = False
            self.random_scale = False
        elif self.split == 'test':
            self.random_rotation = False
            self.ramdom_jitter = False
            self.random_scale = False

        assert os.path.exists(self.root_path), (f'Invalid path: {self.root_path}')
        assert os.path.exists(self.test_path), (f'Invalid path: {self.test_path}')
        if self.split == 'train' or self.split == 'val':
            self.load_data()
            self.prepare_files()
        else:
            self.prepare_test()

        if self.num_val > 0 and self.split == 'val':
            self.files = self.files[:self.num_val]

        _logger.info(f'Found {len(self.files)} {self.split} instances')

    def __len__(self):
        return len(self.files)

    def load_data(self):
        assert self.split in ['train', 'val']
        pts_filename = os.path.join(self.root_path, f'3DMatch_{self.split}_0.030_points.pkl')
        keypts_filename = os.path.join(self.root_path, f'3DMatch_{self.split}_0.030_keypts.pkl')
        overlap_filename = os.path.join(self.root_path, f'3DMatch_{self.split}_0.030_overlap.pkl')

        if os.path.exists(pts_filename) and os.path.exists(keypts_filename) and os.path.exists(overlap_filename):
            _logger.info('Loading data files...')
            with open(pts_filename, 'rb') as f:
                data = pickle.load(f)
                self.points = [*data.values()]
                self.ids_list = [*data.keys()]

            # with open(keypts_filename, 'rb') as f:
            #     self.corr_ids = pickle.load(f)

            with open(overlap_filename, 'rb') as f:
                self.overlap_ratios = pickle.load(f)

        else:
            _logger.error(f"File not found: {pts_filename}")
            raise FileNotFoundError

    def prepare_files(self):
        for idpair in self.overlap_ratios.keys():
            src_idx, ref_idx = idpair.split("@")

            if self.overlap_ratios[idpair] > self.overlap_thres:
                self.files.append((src_idx, ref_idx))

    def prepare_test(self, scene_id=None, return_ply_names=False):
        DATA_FILES = {
            # 'train': './dataloader/split/train_3dmatch.txt',
            # 'val': './dataloader/split/val_3dmatch.txt',
            'test': './dataloader/split/test_3dmatch.txt' }
        self.return_ply_names = return_ply_names

        subset_names = open(DATA_FILES[self.split]).read().split()
        if scene_id is not None:
            subset_names = [subset_names[scene_id]]

        for sname in subset_names:
            traj_file = os.path.join(self.test_path, sname + '-evaluation/gt.log')
            assert os.path.exists(traj_file)
            traj = read_trajectory(traj_file)

            for ctraj in traj:
                i = ctraj.metadata[0]
                j = ctraj.metadata[1]
                T_gt = ctraj.pose
                self.files.append((sname, i, j, T_gt))

    def get_data(self, index):
        if self.split == 'train' or self.split == 'val':
            src_idx, ref_idx = self.files[index]
            src_i = self.ids_list.index(src_idx)
            ref_i = self.ids_list.index(ref_idx)
            src_pcd = self.points[src_i].astype(np.float32)
            ref_pcd = self.points[ref_i].astype(np.float32)

            ##### voxel_down_sample
            pcd0 = make_open3d_point_cloud(src_pcd)
            pcd1 = make_open3d_point_cloud(ref_pcd)
            pcd0 = pcd0.voxel_down_sample(self.voxel_size)
            pcd1 = pcd1.voxel_down_sample(self.voxel_size)
            sel_src = np.array(pcd0.points)
            sel_ref = np.array(pcd1.points)

            T_gt = np.identity(4)

            sname = src_idx.split('/')[0]
            i = int(ref_idx.split('_')[-1])
            j = int(src_idx.split('_')[-1])
        else:
            sname, i, j, T_gt = self.files[index]
            ref_idx = sname + f'/cloud_bin_{i}.ply'
            src_idx = sname + f'/cloud_bin_{j}.ply'

            if self.return_ply_names:
                return sname, ref_idx, src_idx, T_gt

            pcd0 = o3d.io.read_point_cloud(os.path.join(self.test_path, ref_idx))
            pcd1 = o3d.io.read_point_cloud(os.path.join(self.test_path, src_idx))

            ##### voxel_down_sample
            pcd0 = pcd0.voxel_down_sample(self.voxel_size)
            pcd1 = pcd1.voxel_down_sample(self.voxel_size)
            sel_ref = np.array(pcd0.points)
            sel_src = np.array(pcd1.points)

        extra_package = {'seq': sname, 'id_ref': i, 'id_src': j}

        return sel_src, sel_ref, T_gt, extra_package

    def __getitem__(self, index):
        """ Generates one sample of data
        """
        if index in self.cache:
            xyz0, xyz1, pose, extra_package = self.cache[index]
        else:
            xyz0, xyz1, pose, extra_package = self.get_data(index)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (xyz0, xyz1, pose, extra_package)

        xyz0, xyz1, pose = self.apply_augment(xyz0, xyz1, pose)

        data = {'points_src': xyz0,
                'points_ref': xyz1,
                'transform_gt': pose[:3, :],
                'others': extra_package}
        return data



if __name__=='__main__':
    import torch
    from common.math import se3_torch

    class args:
        dataset_path = '../data/3DMatch/'
        num_knn = 16
        sub_sampling_ratio = [4,4,4]
        num_points = 10000
        num_val = -1
        positive_pair_radius_multiplier = 3.0

    split = 'train'
    dataset = ThreeDMatch(args, split=split)

    save_path = f'./out/3dmatch_{split}_0/'
    os.makedirs(save_path, exist_ok=True)

    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=2,
                                                collate_fn=dataset.collate_fn,
                                                shuffle=False,
                                                num_workers=2)
    for i, data in enumerate(dataset_loader):
        print(i)
        if i % 20 == 0:
            for k in ['ori', 'trans']:
                if k == 'trans':
                    data['points_src'] = se3_torch.transform(data['transform_gt'], data['points_src'])

                pt1 = data['points_src'][0].numpy()
                pt2 = data['points_ref'][0].numpy()
                T   = data['transform_gt'][0].numpy()
                print(i, len(data), len(data['points_src_xyz'][0]), len(data['points_ref_xyz'][0]))

                # pt1 = apply_transform(pt1, T)

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(pt1)
                o3d.io.write_point_cloud(os.path.join(save_path, '%06d_1_%s.pcd' % (i, k)), pcd1)

                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(pt2)
                o3d.io.write_point_cloud(os.path.join(save_path, '%06d_2_%s.pcd' % (i, k)), pcd2)
