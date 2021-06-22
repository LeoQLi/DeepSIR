import os, sys
import logging
import pickle
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from common.math.se3 import xyzquat2mat
from dataloader.data_base import apply_transform, make_open3d_point_cloud, DataBase, process_point_cloud
import dataloader.transformation as Transforms

_logger = logging.getLogger('Oxford')


class Oxford(DataBase):
    def __init__(self, args, split=None):
        super(Oxford, self).__init__(args)
        assert split in ['train', 'val', 'test']
        self.split = split
        self.p_crop = 0.6
        self.cache_size = 5000

        self.root_path = args.dataset_path
        self.feat_len  = 3
        self.num_val   = args.num_val
        self.voxel_size = 0.3
        args.thres_radius = self.voxel_size * args.positive_pair_radius_multiplier

        self.icp_cache = {}
        self.icp_path = os.path.join(args.dataset_path, 'icp_opti_pose')
        os.makedirs(self.icp_path, exist_ok=True)

        self.num_points = args.num_points
        self.random_rotation = True
        self.rotation_range = 60
        self.ramdom_jitter = True
        self.jitter_scale = 0.05
        self.random_scale = True
        self.max_scale = 1.2
        self.min_scale = 0.8
        self.permutation = True

        if self.split == 'val' or self.split == 'test':
            self.random_rotation = False
            self.ramdom_jitter = False
            self.random_scale = False
            self.permutation = False

        self.train_dir = 'train_np_nofilter'
        self.test_dir  = 'test_models_20k_np_nofilter'
        assert os.path.exists(self.root_path), (f'Invalid path: {self.root_path}')
        if self.split == 'train':
            self.files = self.make_train_dataset()
        else:
            self.files = self.make_test_dataset()

        if self.num_val > 0 and split == 'val':
            self.files = self.files[:self.num_val]

        _logger.info(f'Found {len(self.files)} {self.split} instances')

    def make_train_dataset(self):
        """
        return dataset: list, [{'file', 'pos_list', 'nonneg_list'}]
        """
        with open(os.path.join(self.root_path, self.train_dir, 'train_relative.txt'), 'r') as f:
            lines_list = f.readlines()

        dataset = []
        for i, line_str in enumerate(lines_list):
            # convert each line to a dict
            line_splitted_list = line_str.split('|')
            try:
                assert len(line_splitted_list) == 3
            except Exception:
                _logger.info(f'Invalid line {i}: {line_splitted_list}')
                continue

            file_name = line_splitted_list[0].strip()
            positive_lines = list(map(int, line_splitted_list[1].split()))
            non_negative_lines = list(map(int, line_splitted_list[2].split()))

            data = {'file': file_name, 'pos_list': positive_lines, 'nonneg_list': non_negative_lines}
            dataset.append(data)

        return dataset

    def make_test_dataset(self):
        """
        return dataset: list, [['anc_idx', 'pos_idx', 'neg_idx', 'q', 't']]
                        (q, t) => Rt, anc = R*pos + t
        """
        with open(os.path.join(self.root_path, self.test_dir, 'groundtruths.pkl'), 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        """ Return the total number of samples
        """
        return len(self.files)

    def pose_refine(self, xyz0, xyz1, t0, t1, M, voxel_size=0.1):
        pcd0 = make_open3d_point_cloud(xyz0[:, :3])
        pcd1 = make_open3d_point_cloud(xyz1[:, :3])
        pcd0 = pcd0.voxel_down_sample(voxel_size)
        pcd1 = pcd1.voxel_down_sample(voxel_size)

        ##### compute the GT pose between two samples
        key = '%d_%d' % (t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.icp_cache:
            if not os.path.exists(filename):
                # work on the downsampled points for speedup
                pcd0.transform(M)
                # xyz0_t = apply_transform(xyz0, M)
                # pcd0 = make_open3d_point_cloud(xyz0_t)
                # pcd1 = make_open3d_point_cloud(xyz1)
                reg = o3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    o3d.registration.ICPConvergenceCriteria(max_iteration=200))
                M2 = M @ reg.transformation

                # the perfect matched point clouds
                # pcd0.transform(reg.transformation)
                # o3d.draw_geometries([pcd0, pcd1])

                # write the pose to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)

            self.icp_cache[key] = M2
        else:
            M2 = self.icp_cache[key]

        return M2

    def get_data(self, index):
        """ Read data from npy file and pre-process the data
            The raw data include 7 channel: [x, y, z, nx, ny, nz, curvature]
        """
        if self.split == 'train':
            anc_idx = self.files[index]['file']
            xyz = np.load(os.path.join(self.root_path, self.train_dir, anc_idx))
            xyz = xyz[:, :self.feat_len]

            pos_idx = anc_idx
            # pos_idx = self.files[index]['pos_list']
            # xyz1 = np.load(os.path.join(self.root_path, self.train_dir, anc_idx.split('/') + '/%d.npy' % pos_idx))

            xyz0 = Transforms.RandomCrop.crop(xyz, self.p_crop)
            xyz1 = Transforms.RandomCrop.crop(xyz, self.p_crop)

            pose_mat = np.identity(4)
        else:
            pos_idx = self.files[index]['pos_idx']
            xyz0 = np.load(os.path.join(self.root_path, self.test_dir, '%d.npy' % pos_idx))
            anc_idx = self.files[index]['anc_idx']
            xyz1 = np.load(os.path.join(self.root_path, self.test_dir, '%d.npy' % anc_idx))

            xyz0 = xyz0[:, :self.feat_len]
            xyz1 = xyz1[:, :self.feat_len]

            t_xyz = self.files[index]['t']
            quat = self.files[index]['q']  # [qw qx qy qz]
            xyz_quat = np.concatenate([t_xyz, quat], axis=0)
            pose_mat = xyzquat2mat(xyz_quat)

        xyz0 = process_point_cloud(xyz0, r_min=0.0, r_max=50.0, z_min=-3.0, z_max=20.0, grid_size=0.0)
        xyz1 = process_point_cloud(xyz1, r_min=0.0, r_max=50.0, z_min=-3.0, z_max=20.0, grid_size=0.0)

        ##### voxel_down_sample
        pcd0 = make_open3d_point_cloud(xyz0[:, :3])
        pcd1 = make_open3d_point_cloud(xyz1[:, :3])
        pcd0 = pcd0.voxel_down_sample(self.voxel_size)
        pcd1 = pcd1.voxel_down_sample(self.voxel_size)
        xyz0 = np.array(pcd0.points)
        xyz1 = np.array(pcd1.points)

        # pose_mat = self.pose_refine(xyz0, xyz1, pos_idx, anc_idx, pose_mat)

        extra_package = {'seq': None, 'id_src': pos_idx, 'id_ref': anc_idx}

        return xyz0, xyz1, pose_mat, extra_package

    def __getitem__(self, index):
        """ Generates one sample of data
        """
        if index in self.cache:
            xyz0, xyz1, pose, extra_package = self.cache[index]
        else:
            xyz0, xyz1, pose, extra_package = self.get_data(index)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (xyz0, xyz1, pose, extra_package)

        xyz0, xyz1, pose = self.apply_augment_V2(xyz0, xyz1, pose)

        data = {'points_src': xyz0,
                'points_ref': xyz1,
                'transform_gt': pose[:3, :],
                'others': extra_package}
        return data



if __name__=='__main__':
    import time
    import torch
    import torchvision

    from common.math import se3_torch
    from dataloader.datasets import get_transforms

    class args:
        num_knn = 16
        sub_sampling_ratio = [4,4,4]
        dataset_path = '../data/Oxford/'
        feat_len = 3
        num_val = -1
        noise_type = 'crop'  # ['clean', 'jitter', 'crop']
        crop_partial = [0.7, 0.7]
        num_points = 10000
        positive_pair_radius_multiplier = 3.0
        rot_mag = 50
        trans_mag = 2.0
        xy_rot_scale = 0.1

    # train_transforms, _ = get_transforms(noise_type=args.noise_type,
    #                                     rot_mag=args.rot_mag, trans_mag=args.trans_mag,
    #                                     num_points=args.num_points, xy_rot_scale=args.xy_rot_scale,
    #                                     partial_p_keep=args.crop_partial)
    # _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
    # train_transforms = torchvision.transforms.Compose(train_transforms)

    # dataset = Oxford(args, split='train', transform=train_transforms)


    split = 'train'
    dataset = Oxford(args, split=split)

    save_path = f'./out/Oxford_{split}_0/'
    os.makedirs(save_path, exist_ok=True)

    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=2,
                                                collate_fn=dataset.collate_fn,
                                                shuffle=False,
                                                num_workers=2)
    total_time = 0
    time_before = time.time()
    for i, data in enumerate(dataset_loader):
        print(i)
        if i % 10 == 0 and i > 0:
            for k in ['ori', 'trans']:
                if k == 'trans':
                    data['points_src'] = se3_torch.transform(data['transform_gt'], data['points_src'])

                # pt, norm = se3_torch.transform(data['transform_gt'],
                #             data['points_src'][:, :, :3], data['points_src'][:, :, 3:6])
                # data['points_src'] = torch.cat((pt, norm), dim=-1)

                pt1 = data['points_src'][0].numpy()
                pt2 = data['points_ref'][0].numpy()
                T   = data['transform_gt'][0].numpy()
                print(i, len(data), len(data['points_src_xyz'][0]), len(data['points_ref_xyz'][0]))

                # pt1 = apply_transform(pt1, T)

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(pt1[:, :3])
                # pcd1.colors = o3d.utility.Vector3dVector(pt1[:, 3:6])
                o3d.io.write_point_cloud(os.path.join(save_path, '%06d_1_%s.pcd' % (i, k)), pcd1)

                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(pt2[:, :3])
                # pcd2.colors = o3d.utility.Vector3dVector(pt2[:, 3:6])
                o3d.io.write_point_cloud(os.path.join(save_path, '%06d_2_%s.pcd' % (i, k)), pcd2)

        total_time += time.time() - time_before

    print('Total validate time: {:.3f}s'.format(total_time))