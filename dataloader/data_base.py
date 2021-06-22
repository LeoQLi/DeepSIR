import os, sys
import random
import copy
import logging
import numpy as np
import open3d as o3d
from collections import defaultdict
from scipy.linalg import expm, norm
# from sklearn.neighbors import KDTree

import torch
from torch.utils.data import Dataset
import torch_points_kernels as Util

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import dataloader.transformation as Transforms


class DataBase(Dataset):
    def __init__(self, args):
        """ Initialization
        """
        super(DataBase, self).__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.num_knn = args.num_knn                        # Number of neighbours in knn search
        self.sub_sampling_ratio = args.sub_sampling_ratio  # sampling ratio of random sampling at each layer
        self.num_layers = len(self.sub_sampling_ratio)

        self.cache = {}
        self.cache_size = 8000

        self.num_points = None
        self.random_rotation = None
        self.rotation_range = None
        self.ramdom_jitter = None
        self.jitter_scale = None
        self.random_scale = None
        self.max_scale = None
        self.min_scale = None

        rot_mag = args.rot_mag
        trans_mag = args.trans_mag
        xy_rot_scale = args.xy_rot_scale
        self.transform_3d = Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, xy_rot_scale=xy_rot_scale)
        self.transform_z = Transforms.RandomRotatorZ(60)
        self.xyz_jitter = Transforms.RandomJitter(scale=0.01, clip=0.05)

        # Random initialisation for weights
        # self.possibility = []
        # self.min_possibility = []
        # if mode == 'test':
        #     for points in in_dataset:
        #         self.possibility += [np.random.rand(points.shape[0]) * 1e-3]   # possibility of the whole point cloud
        #         self.min_possibility += [float(np.min(self.possibility[-1]))]  # minimum value of the last point cloud
        # self._logger.info('The number of input points is set to: {}'.format(num_points))

    def pt_crop(self, data):
        """ Get the k nearest neighbor points of a seed point, fix the total number of clouds
            :param data: dict
        """
        for k in ['points_src', 'points_ref']:
            num_pts = len(data[k])
            # Check if the number of points in the selected cloud is less than the predefined num_points
            if num_pts < self.num_points:   # pad points
                num_to_pad = self.num_points - num_pts
                pad_id = np.random.choice(num_pts, size=num_to_pad, replace=True)
                data[k] = np.concatenate((data[k], data[k][pad_id, :]), axis=0)
                num_pts = self.num_points

            xyz = data[k][:, :3]
            # choose the seed point of input region
            pick_idx = np.random.choice(num_pts, 1)
            # cloud_ind = int(np.argmin(self.min_possibility))
            # pick_idx = np.argmin(self.possibility[cloud_ind])
            center_point = xyz[pick_idx, :].reshape(1, -1)

            search_tree = KDTree(xyz)
            select_idx = search_tree.query(center_point, k=self.num_points)[1][0]

            select_idx = DP.shuffle_idx(select_idx)  # TODO
            select_points = data[k][select_idx]
            data[k] = select_points.astype(np.float32)
            # new_xyz = select_points - center_point

            # # update the possibility of the selected pc
            # dists = np.sum(np.square((selected_pc - pc[pick_idx]).astype(np.float32)), axis=1)
            # delta = np.square(1 - dists / np.max(dists))
            # self.possibility[cloud_ind][selected_idx] += delta
            # self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])
        return data

    def nn_search_knn(self, data_list_stack):
        """
        data_list_stack: dic of numpy array or cpu tensor
        """
        for k in ['points_src', 'points_ref']:
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            batch_pc = data_list_stack[k][:, :, :3].contiguous()

            for i in range(self.num_layers):
                neighbour_idx = DP.knn_search(batch_pc, batch_pc, self.num_knn)  # [B, N, num_knn]
                num = batch_pc.shape[1] // self.sub_sampling_ratio[i]            # N/(4^i)
                pool_i = neighbour_idx[:, :num, :]                               # [B, num, num_knn]
                sub_points = batch_pc[:, :num, :].contiguous()                   # [B, num, 3]
                up_i = DP.knn_search(sub_points, batch_pc, 1)                    # [B, N, 1]

                input_points.append(batch_pc)                                    # [B, num, 3]
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_pc = sub_points

            data_list_stack[k+'_xyz'] = torch.cat(input_points, axis=1)
            data_list_stack[k+'_neigh_idx'] = torch.from_numpy(np.concatenate(input_neighbors, axis=1)).long()
            data_list_stack[k+'_sub_idx'] = torch.from_numpy(np.concatenate(input_pools, axis=1)).long()
            data_list_stack[k+'_interp_idx'] = torch.from_numpy(np.concatenate(input_up_samples, axis=1)).long()
        return data_list_stack

    def nn_search_rnn(self, data_list_stack):
        """
        data_list_stack: dic of numpy array or cpu tensor
        """
        for k in ['points_src', 'points_ref']:
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            batch_pc = data_list_stack[k][:, :, :3].contiguous()

            for i in range(self.num_layers):
                neighbour_idx = DP.ball_query(batch_pc, batch_pc, 1.0, self.num_knn)  # [B, N, num_knn]
                num = batch_pc.shape[1] // self.sub_sampling_ratio[i]                 # N/(4^i)
                pool_i = neighbour_idx[:, :num, :]                                    # [B, num, num_knn]
                sub_points = batch_pc[:, :num, :].contiguous()                        # [B, num, 3]
                up_i = DP.ball_query(sub_points, batch_pc, 1.0, 1)                    # [B, N, 1]

                input_points.append(batch_pc)                                         # [B, num, 3]
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_pc = sub_points

            data_list_stack[k+'_xyz'] = torch.cat(input_points, axis=1)
            data_list_stack[k+'_neigh_idx'] = torch.cat(input_neighbors, axis=1)
            data_list_stack[k+'_sub_idx'] = torch.cat(input_pools, axis=1)
            data_list_stack[k+'_interp_idx'] = torch.cat(input_up_samples, axis=1)
        return data_list_stack

    def nn_search(self, data_list_stack):
        """
        data_list_stack: dic of numpy array or cpu tensor
        """
        for k in ['points_src', 'points_ref']:
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            batch_pc = data_list_stack[k][:, :, :3].contiguous()

            for i in range(self.num_layers):
                neighbour_idx, dist = Util.knn(batch_pc, batch_pc, self.num_knn)   # only on cpu
                # neighbour_idx, dist = Util.ball_query(1.0, self.num_knn, batch_pc, batch_pc) # [B, N, num_knn]
                num = batch_pc.shape[1] // self.sub_sampling_ratio[i]                       # N/(4^i)
                pool_i = neighbour_idx[:, :num, :]                                          # [B, num, num_knn]
                sub_points = batch_pc[:, :num, :].contiguous()                              # [B, num, 3]
                up_i, dist = Util.knn(sub_points, batch_pc, 1)                              # [B, N, 1]
                # up_i, dist = Util.ball_query(1.0, 1, sub_points, batch_pc)

                input_points.append(batch_pc)                                               # [B, num, 3]
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_pc = sub_points

            data_list_stack[k+'_xyz'] = torch.cat(input_points, axis=1)
            data_list_stack[k+'_neigh_idx'] = torch.cat(input_neighbors, axis=1)
            data_list_stack[k+'_sub_idx'] = torch.cat(input_pools, axis=1)
            data_list_stack[k+'_interp_idx'] = torch.cat(input_up_samples, axis=1)
        return data_list_stack

    def collate_fn_V2(self, data):
        """
        data: batches of (pt_src, pt_ref, T_gt) => dict of batch
        """
        data_list_stack = {}
        data_list_stack['points_src'] = torch.from_numpy(np.stack([d[0] for d in data])).float()
        data_list_stack['points_ref'] = torch.from_numpy(np.stack([d[1] for d in data])).float()
        data_list_stack['transform_gt'] = torch.from_numpy(np.stack([d[2] for d in data])).float()

        return self.nn_search(data_list_stack)

    def collate_fn(self, data):
        """ several batchs with dict data -> a dict with several batch data
            :param data: [dict, dict, ...]
        """
        data_list = defaultdict(list)
        for n in range(len(data)):             # for each batch
            for k in data[n]:                  # for each key
                data_list[k].append(data[n][k])

        # minimun number of points in a batch
        data_list_stack = {}
        data_list_stack['points_src'] = torch.from_numpy(np.stack(data_list['points_src'], axis=0)).float()
        data_list_stack['points_ref'] = torch.from_numpy(np.stack(data_list['points_ref'], axis=0)).float()
        data_list_stack['transform_gt'] = torch.from_numpy(np.stack(data_list['transform_gt'], axis=0)).float()
        data_list_stack['others'] = data_list['others']

        if 'labels_src' in data_list and 'labels_ref' in data_list:
            data_list_stack['labels_src'] = torch.from_numpy(np.stack(data_list['labels_src'], axis=0)).long()
            data_list_stack['labels_ref'] = torch.from_numpy(np.stack(data_list['labels_ref'], axis=0)).long()

        if 'matches' in data_list:
            data_list_stack['matches'] = data_list['matches']   # list in CPU, number of each is different

        return self.nn_search(data_list_stack)

    def apply_augment(self, xyz0, xyz1, M=np.identity(4), fixed=False):
        """ data augmentation
        xyz0, xyz1: [N, C], [x, y, z, nx, ny, nz, ...]
        """
        ###### add the additional rotation
        if self.random_rotation:
            T0 = sample_random_trans(xyz0[:, :3], self.rotation_range)
            T1 = sample_random_trans(xyz1[:, :3], self.rotation_range)
            xyz0 = apply_transform(xyz0, T0)
            xyz1 = apply_transform(xyz1, T1)
            trans = T1 @ M @ np.linalg.inv(T0)
        else:
            trans = M

        ###### fixed number of points
        if self.num_points > 0:
            if fixed:
                num = min(len(xyz0), len(xyz1))
                xyz0 = Transforms.FixedResampler._resample(xyz0, num)
                xyz1 = Transforms.FixedResampler._resample(xyz1, num)
            else:
                xyz0 = Transforms.Resampler._resample(xyz0, self.num_points)
                xyz1 = Transforms.Resampler._resample(xyz1, self.num_points)

        ###### data augmentation: noise
        if self.ramdom_jitter and random.random() < 0.95:
            xyz0[:, :3] += np.random.rand(xyz0.shape[0], 3) * self.jitter_scale
            xyz1[:, :3] += np.random.rand(xyz1.shape[0], 3) * self.jitter_scale

        ###### data augmentation: scale
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            xyz0[:, :3] = scale * xyz0[:, :3]
            xyz1[:, :3] = scale * xyz1[:, :3]

        return xyz0, xyz1, trans

    def apply_augment_V2(self, xyz0, xyz1, M=np.identity(4), fixed=False):
        """ data augmentation
        xyz0, xyz1: [N, C], [x, y, z, nx, ny, nz, ...]
        """
        ###### add the additional rotation
        if self.random_rotation:
            xyz0, _, T0 = self.transform_z(xyz0)
            xyz1, _, T1 = self.transform_z(xyz1)
            xyz0, _, T00 = self.transform_3d(xyz0)
            trans = T1 @ M @ np.linalg.inv(T0) @ np.linalg.inv(T00)
        else:
            trans = M

        ###### fixed number of points
        if self.num_points > 0:
            if fixed:
                if len(xyz0) < len(xyz1):
                    xyz0 = Transforms.FixedResampler._resample(xyz0, len(xyz1))
                else:
                    xyz1 = Transforms.FixedResampler._resample(xyz1, len(xyz0))
            else:
                # xyz0 = Transforms.Resampler._resample(xyz0, self.num_points)
                # xyz1 = Transforms.Resampler._resample(xyz1, self.num_points)
                xyz0 = Transforms.FixedResampler._resample(xyz0, self.num_points)
                xyz1 = Transforms.FixedResampler._resample(xyz1, self.num_points)
            assert len(xyz0) == len(xyz1)

        ###### data augmentation: noise
        if self.ramdom_jitter:
            xyz0 = self.xyz_jitter(xyz0)
            xyz1 = self.xyz_jitter(xyz1)

        ###### data augmentation: scale
        if self.random_scale:
            scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            xyz0[:, :3] = scale * xyz0[:, :3]
            xyz1[:, :3] = scale * xyz1[:, :3]

        return xyz0, xyz1, trans


def process_point_cloud(cloud, r_min=0.0, r_max=50.0, z_min=-3.0, z_max=10.0, grid_size=0.0):
    """ Crop point cloud: radius ball, Z height, grid sub-sample
        :param cloud: [N, C]
    """
    # Crop to the radius
    mask = np.sum(np.square(cloud[:, :3]), axis=1) <= r_max**2
    cloud = cloud[mask, :]

    mask = np.sum(np.square(cloud[:, :3]), axis=1) > r_min**2
    cloud = cloud[mask, :]

    # Crop to the height
    mask = (cloud[:, 2] >= z_min) & (cloud[:, 2] <= z_max)
    cloud = cloud[mask, :]

    # grid sub-sample
    # if grid_size > 0.0:
    #     sub_points, sub_feat = DP.grid_sub_sampling(cloud[:, :3], features=cloud[:, 3:], grid_size=grid_size)
    #     cloud = np.concatenate((sub_points, sub_feat), axis=1)

    # pcd = make_open3d_point_cloud(xyz)
    # pcd = pcd.voxel_down_sample(voxel_size)

    # random sub-sample
    # sel_xyz = Transforms.Resampler._resample(sel_xyz, num_points)

    return cloud


def farthest_point_sampler(pts, k):
    def calc_distances(p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    farthest_pts = np.zeros((k, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, k):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def save_data(path, data, delimiter=' '):
    if path.endswith('bin'):
        data = data.astype(np.float32)
        with open(path, 'wb') as f:
            data.tofile(f, sep='', format='%f')
    elif path.endswith('txt'):
        with open(path, 'w') as f:
            np.savetxt(f, data, delimiter=delimiter)
    elif path.endswith('npy'):
        with open(path, 'wb') as f:
            np.save(f, data)
    else:
        print('Unknown file type: %s' % path)
        exit()


def make_open3d_point_cloud(xyz, color=None, normal=None):
    assert xyz.shape[-1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if color is not None:
        assert len(color) == len(xyz) and color.shape[-1] == 3
        # color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)

    if normal is not None:
        assert len(normal) == len(xyz) and normal.shape[-1] == 3
        pcd.normals = o3d.utility.Vector3dVector(normal)

    return pcd


def pointcloud_to_spheres(pcd, voxel_size, color, sphere_size=0.6):
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    if isinstance(pcd, o3d.geometry.PointCloud):
        pcd = np.array(pcd.points)
    for i, p in enumerate(pcd):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        # si.paint_uniform_color(pcd.colors[i])
        spheres += si
    return spheres


def M(axis, theta):
    """ Rotation matrix along axis with angle theta
    """
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, rotation_range=0):
    """
    pcd: [N, 3]
    rotation_range: angle, in degree
    """
    randg = np.random.RandomState()
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


def apply_transform(p0, trans_mat):
    """
    p0: [N, C], [x, y, z, nx, ny, nz, ...]
    trans_mat: [3/4, 4]
    """
    def transform(trans, pts):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def rotate(R, pts):
        pts = pts @ R.T
        return pts

    p1 = transform(trans_mat, p0[:, :3])
    if p0.shape[1] == 6:
        n1 = rotate(trans_mat[:3, :3], p0[:, 3:6])
        p1 = np.concatenate((p1, n1), axis=-1)
    elif p0.shape[1] > 6:
        n1 = rotate(trans_mat[:3, :3], p0[:, 3:6])
        p1 = np.concatenate((p1, n1, p0[:, 6:]), axis=-1)

    return p1


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


