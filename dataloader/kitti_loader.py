import os, sys
import glob
import copy
import yaml
import logging
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from dataloader.data_base import apply_transform, make_open3d_point_cloud, \
                                process_point_cloud, get_matching_indices, DataBase

_logger = logging.getLogger('KITTI_Dataset')


class KITTIPair(DataBase):
    def __init__(self, args, split='train'):
        super(KITTIPair, self).__init__(args)
        assert split in ['train', 'val', 'test']

        self.DATA_FILES = {
            'train': 'dataloader/split/train_kitti.txt',
            'val': 'dataloader/split/val_kitti.txt',
            'test': 'dataloader/split/test_kitti.txt'}
        self.IS_ODOMETRY = True
        self.MIN_TIME_DIFF = 2     # TODO
        self.MAX_TIME_DIFF = 3
        self.MIN_DIST = 10

        self.split = split
        self.with_label = False
        self.num_val = args.num_val
        self.pipeline = args.pipeline

        self.voxel_size = args.voxel_size
        self.matching_search_voxel_size = self.voxel_size * args.positive_pair_radius_multiplier
        args.thres_radius = self.matching_search_voxel_size

        self.root_path = os.path.join(args.dataset_path, 'dataset')
        self.icp_path = os.path.join(args.dataset_path, 'icp_opti_pose')
        os.makedirs(self.icp_path, exist_ok=True)

        self.files = []
        self.pose_cache = {}
        self.icp_cache = {}

        self.feat_len = args.feat_len
        self.num_points = args.num_points
        self.random_rotation = True
        # self.rotation_range = 45
        self.ramdom_jitter = True
        # self.jitter_scale = 0.05
        self.random_scale = False
        self.max_scale = 1.2
        self.min_scale = 0.8
        self.permutation = True

        if self.split == 'val':
            self.random_rotation = False
            self.ramdom_jitter = False
            self.random_scale = False
            self.permutation = True
        elif self.split == 'test':
            self.random_rotation = False
            self.ramdom_jitter = False
            self.random_scale = False
            self.permutation = False

        assert os.path.exists(self.root_path), (f'Invalid path: {self.root_path}')
        if self.split == 'train':
            self.prepare_kitti()
        else:
            self.prepare_kitti_test()

        if self.num_val > 0 and split == 'val':
            self.files = self.files[:self.num_val]

        _logger.info(f'Found {len(self.files)} {self.split} instances')

    def prepare_kitti(self):
        subset_names = open(self.DATA_FILES[self.split]).read().split()

        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)
            # the speed of drive_id 1 is very high
            if (drive_id == 1) and (self.MAX_TIME_DIFF - 1) > self.MIN_TIME_DIFF:
                max_time_diff = self.MAX_TIME_DIFF - 1
            else:
                max_time_diff = self.MAX_TIME_DIFF

            for start_time in inames:
                for time_diff in range(self.MIN_TIME_DIFF, max_time_diff):
                    pair_time = time_diff + start_time
                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))

    def prepare_kitti_test(self):
        subset_names = open(self.DATA_FILES[self.split]).read().split()

        # Generate KITTI pairs within N meter distance
        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)

            # load all poses
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])

            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = pdist.sum(-1)
            more_than_10 = pdist > self.MIN_DIST ** 2

            curr_time = inames[0]
            while curr_time in inames:
                # Find the min index
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        # pair (8, 15, 58) is wrong
        if self.split == 'test':
            self.files.remove((8, 15, 58))

    def __len__(self):
        return len(self.files)

    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root_path + '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root_path + '/' + self.date +
                               '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)

        assert len(fnames) > 0, f"Make sure that the path {self.root_path} has drive id: {drive_id}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        return inames

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root_path + '/poses/%02d.txt' % drive
            if data_path not in self.pose_cache:
                self.pose_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.pose_cache[data_path]
            else:
                return self.pose_cache[data_path][indices]
        else:
            data_path = self.root_path + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
            odometry = []
            if indices is None:
                fnames = glob.glob(self.root_path + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
                indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            for index in indices:
                filename = os.path.join(data_path, '%010d%s' % (index, ext))
                if filename not in self.pose_cache:
                    self.pose_cache[filename] = np.genfromtxt(filename)
                    odometry.append(self.pose_cache[filename])

            odometry = np.array(odometry)
            return odometry

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0
        else:
            lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

            R = 6378137  # Earth's radius in metres

            # convert to metres
            lat, lon = np.deg2rad(lat), np.deg2rad(lon)
            mx = R * lon * np.cos(lat)
            my = R * lat

            times = odometry.T[-1]
            return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root_path + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        else:
            fname = self.root_path + \
                    '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)

    def rot3d(self, axis, angle):
        ei = np.ones(3, dtype='bool')
        ei[axis] = 0
        i = np.nonzero(ei)[0]
        m = np.eye(3)
        c, s = np.cos(angle), np.sin(angle)
        m[i[0], i[0]] = c
        m[i[0], i[1]] = -s
        m[i[1], i[0]] = s
        m[i[1], i[1]] = c
        return m

    def pos_transform(self, pos):
        x, y, z, rx, ry, rz, _ = pos[0]
        RT = np.eye(4)
        RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
        RT[:3, 3] = [x, y, z]
        return RT

    def load_label(self, label_path, drive, N=0):
        pass

    def pose_refine(self, xyz0, xyz1, drive, t0, t1, voxel_size=0.05):
        ##### voxel_down_sample
        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)
        pcd0 = pcd0.voxel_down_sample(voxel_size)
        pcd1 = pcd1.voxel_down_sample(voxel_size)

        ##### compute the GT pose between two samples
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.icp_cache:
            if not os.path.exists(filename):
                cur_odometry = self.get_video_odometry(drive, indices=[t0, t1])
                positions = [self.odometry_to_positions(odometry) for odometry in cur_odometry]

                # the provided pose information
                if self.IS_ODOMETRY:
                    M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                         @ np.linalg.inv(self.velo2cam)).T
                else:
                    M = self.get_position_transform(positions[0], positions[1], invert=True).T

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

    def get_matches(self, xyz0, xyz1, trans):
        """ Compute the correspondences using voxelized and untransformed points
        """
        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)
        matches = get_matching_indices(pcd0, pcd1, trans, self.matching_search_voxel_size, K=None)
        if len(matches) < 1000:
            _logger.warning(f"Insufficient matches in {drive}, {t0}, {t1}, {len(matches)}/{len(sel_xyz0)}")
        matches = np.array(matches)     # [N', 2], N' > N
        return matches

    def get_data(self, idx):
        """
        return xyz: [N, 5], [x, y, z, reflectance, (label)]
        """
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        ### load points: XYZ and reflectance, [N, 4]
        xyz0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyz1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = np.concatenate((xyz0, np.zeros((len(xyz0), 2))), axis=1)   # [N, 6]
        xyz1 = np.concatenate((xyz1, np.zeros((len(xyz1), 2))), axis=1)   # [N, 6]

        ### load semantic label: XYZ, reflectance, label, [N, 5]
        if self.with_label:
            labels0 = self.load_label(fname0, drive, len(xyz0))
            labels1 = self.load_label(fname1, drive, len(xyz1))
            xyz0[:, 4] = np.squeeze(labels0)
            xyz1[:, 4] = np.squeeze(labels1)

        ### filter some points
        xyz0 = process_point_cloud(xyz0, r_min=3.0, r_max=60.0, z_min=-3.0, z_max=10.0, grid_size=0.0)
        xyz1 = process_point_cloud(xyz1, r_min=3.0, r_max=60.0, z_min=-3.0, z_max=10.0, grid_size=0.0)

        ### permute point order
        if self.permutation:
            xyz0 = np.random.permutation(xyz0)
            xyz1 = np.random.permutation(xyz1)

        ### load and optimize the gt pose
        gt_T = self.pose_refine(xyz0[:, :3], xyz1[:, :3], drive, t0, t1)

        ### voxel based down-sample
        pcd0 = make_open3d_point_cloud(xyz0[:, :3], color=xyz0[:, 3:6])
        pcd1 = make_open3d_point_cloud(xyz1[:, :3], color=xyz1[:, 3:6])
        pcd0 = pcd0.voxel_down_sample(self.voxel_size)
        pcd1 = pcd1.voxel_down_sample(self.voxel_size)
        # new sub-points with shape: [N, 5]
        sel_xyz0 = np.concatenate((np.array(pcd0.points, dtype=np.float32),
                                   np.array(pcd0.colors, dtype=np.float32)[:, :2]), axis=1)
        sel_xyz1 = np.concatenate((np.array(pcd1.points, dtype=np.float32),
                                   np.array(pcd1.colors, dtype=np.float32)[:, :2]), axis=1)

        extra_package = {'seq': drive, 'id_src': t0, 'id_ref': t1}
        return sel_xyz0, sel_xyz1, gt_T, extra_package

    def __getitem__(self, idx):
        pass


class SemanticKITTIPair(KITTIPair):
    def __init__(self, args, split='train'):
        super(SemanticKITTIPair, self).__init__(args, split)
        self.with_label = True

        # load Semantic_KITTI class info
        self.filelist = "./dataloader/semantic-kitti.yaml"
        with open(self.filelist, 'r') as f:
            semkittiyaml = yaml.safe_load(f)
        self.learning_map = semkittiyaml['learning_map']
        a = np.vectorize(self.learning_map.__getitem__)

        self.label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            self.label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    def load_label(self, label_path, drive, N=0):
        if drive > 10:
            sem_label = np.expand_dims(np.zeros(N, dtype=int), axis=1)
        else:
            label_path = label_path.replace('velodyne', 'labels')[:-3] + 'label'
            sem_label = np.fromfile(label_path, dtype=np.int32).reshape((-1, 1))
            # semantic label in lower half, instance id in upper half, delete high 16 digits binary
            sem_label = sem_label & 0xFFFF
            sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label).astype(np.uint8)
        return sem_label

    def get_unique_label(self):
        unique_label = np.asarray(sorted(list(self.label_name.keys())))[1:] - 1
        unique_label_str = [self.label_name[x] for x in unique_label + 1]
        return unique_label, unique_label_str

    def __getitem__(self, idx):
        if idx in self.cache:
            xyz0, xyz1, gt_T, extra_package = self.cache[idx]
        else:
            xyz0, xyz1, gt_T, extra_package = self.get_data(idx)
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (xyz0, xyz1, gt_T, extra_package)

        fixed = False if (self.split == 'train' or self.split == 'val') else True
        xyz0, xyz1, trans = self.apply_augment_V2(xyz0, xyz1, gt_T, fixed)

        matches = []
        if self.pipeline == 'align' and (self.split == 'train' or self.split == 'val'):
            matches = self.get_matches(xyz0[:, :3], xyz1[:, :3], trans)

        data = {'points_src': xyz0[:, :self.feat_len],      # [x, y, z, reflectance]
                'points_ref': xyz1[:, :self.feat_len],
                'labels_src': xyz0[:, 4].astype(np.int32),
                'labels_ref': xyz1[:, 4].astype(np.int32),
                'transform_gt': trans[:3, :],
                'matches': matches,
                'others': extra_package}
        return data



if __name__=='__main__':
    import time
    import torch
    from common.math import se3_torch

    def save_data(path, data, delimiter=' '):
        if path.endswith('bin'):
            data = data.astype(np.float32)
            with open(path, 'wb') as f:
                data.tofile(f, sep='', format='%f')
        elif path.endswith('txt'):
            with open(path, 'w') as f:
                np.savetxt(f, data, fmt='%.3f', delimiter=delimiter)
        elif path.endswith('npy'):
            with open(path, 'wb') as f:
                np.save(f, data)
        else:
            print('Unknown file type: %s' % path)
            exit()

    class args:
        pipeline = 'feat'
        feat_len = 4
        num_knn = 16
        sub_sampling_ratio = [4,4,4]
        dataset_path = '../data/KITTI/'
        voxel_size = 0.3
        num_val = -1
        num_points = -1
        positive_pair_radius_multiplier = 3.0
        rot_mag = 50
        trans_mag = 2.0
        xy_rot_scale = 0.1

    split = 'test'
    # dataset = KITTIPair(args, split=split)
    dataset = SemanticKITTIPair(args, split=split)

    num = []
    i = 0
    for i in len(dataset):
        print(i)
        data = dataset[i]
        num.append(len(data['points_src']))
        num.append(len(data['points_ref']))

    num = np.array(num)

    print(num.mean())
    print(num.min())
    print(num.max())
    sys.exit(0)

    save_path = f'./out/kitti_{split}_2/'
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
        if i % 10 == 0:
            bs = 0
            for k in ['ori', 'trans']:
                if k == 'trans':
                    data['points_src'] = se3_torch.transform(data['transform_gt'], data['points_src'])

                pt1 = data['points_src'][bs].numpy()
                pt2 = data['points_ref'][bs].numpy()
                print(i, len(data), len(data['points_src_xyz'][bs]), len(data['points_ref_xyz'][bs]))

                # T   = data['transform_gt'][bs].numpy()
                # pt1 = apply_transform(pt1, T)

                # pcd1 = o3d.geometry.PointCloud()
                # pcd1.points = o3d.utility.Vector3dVector(pt1)
                # o3d.io.write_point_cloud(os.path.join(save_path, '%06d_1_%s.pcd' % (i, k)), pcd1)

                # pcd2 = o3d.geometry.PointCloud()
                # pcd2.points = o3d.utility.Vector3dVector(pt2)
                # o3d.io.write_point_cloud(os.path.join(save_path, '%06d_2_%s.pcd' % (i, k)), pcd2)


                l1 = data['labels_src'][bs].numpy()[:, None]
                l2 = data['labels_ref'][bs].numpy()[:, None]
                save_data(os.path.join(save_path, '%06d_1_%s.txt' % (i, k)), np.concatenate((pt1, l1), axis=1) )
                save_data(os.path.join(save_path, '%06d_2_%s.txt' % (i, k)), np.concatenate((pt2, l2), axis=1) )


        total_time += time.time() - time_before

    print('Total validate time: {:.3f}s'.format(total_time))

