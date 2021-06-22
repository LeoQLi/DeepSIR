"""Utilities for PointNet related functions

Modified from:
    Pytorch Implementation of PointNet and PointNet++
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch


def index_points(points, idx):
    """Array indexing, i.e. retrieves relevant points based on indices

    Args:
        points: input points, [B, N, C]
        idx: sample index, [B, S]. S can be 2 dimensional
    Returns:
        new_points: indexed points, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # [B, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1                           # [1, S]
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)  # [B, S]
    new_points = points[batch_indices, idx, :]    # [B, S, C]
    return new_points


def farthest_point_sample(xyz, npoint):
    """Iterative farthest point sampling

    Args:
        xyz: pointcloud, [B, N, C]
        npoint: number of samples
    Returns:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz, itself_indices=None):
    """ Grouping layer in PointNet++.

    Inputs:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, (B, N, C)
        new_xyz: query points, (B, S, C)
        itself_indices (Optional): Indices of new_xyz into xyz (B, S).
          Used to try and prevent grouping the point itself into the neighborhood.
          If there is insufficient points in the neighborhood, or if left is none, the resulting cluster will
          still contain the center point.
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # (B, S, N)
    sqrdists = square_distance(new_xyz, xyz)

    if itself_indices is not None:
        # Remove indices of the center points so that it will not be chosen
        batch_indices = torch.arange(B, dtype=torch.long).to(device)[:, None].repeat(1, S)  # (B, S)
        row_indices = torch.arange(S, dtype=torch.long).to(device)[None, :].repeat(B, 1)  # (B, S)
        group_idx[batch_indices, row_indices, itself_indices] = N

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    if itself_indices is not None:
        group_first = itself_indices[:, :, None].repeat([1, 1, nsample])
    else:
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: torch.Tensor,
                     returnfps: bool=False):
    """
    Args:
        npoint (int): Set to negative to compute for all points
        radius:
        nsample:
        xyz: input points position, [B, N, C]
        points: input points, [B, N, D]
        returnfps (bool) Whether to return furthest point indices
    Returns:
        new_xyz: sampled points position, [B, 1, C]
        new_points: sampled points, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape

    if npoint > 0:
        S = npoint
        fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
    else:
        S = xyz.shape[1]
        fps_idx = torch.arange(0, xyz.shape[1])[None, ...].repeat(xyz.shape[0], 1)
        new_xyz = xyz

    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, N, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_multi(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, normals: torch.Tensor,
                           returnfps: bool = False):
    """ Sample and group for xyz, dxyz and ppf features

    Args:
        npoint(int): Number of clusters (equivalently, keypoints) to sample.
                     Set to negative to compute for all points
        radius(int): Radius of cluster for computing local features
        nsample: Maximum number of points to consider per cluster
        xyz: XYZ coordinates of the points (B, N, 3)
        normals: Corresponding normals for the points (required for ppf computation) (B, N, 3)
        returnfps: Whether to return indices of FPS points and their neighborhood

    Returns:
        Dictionary containing the following fields ['xyz', 'dxyz', 'ppf'].
        If returnfps is True, also returns: grouped_xyz, fps_idx
    """
    B, N, C = xyz.shape

    if npoint > 0:
        S = npoint
        fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)     # [B, npoint, C]
        nr = index_points(normals, fps_idx)[:, :, None, :]
    else:
        S = xyz.shape[1]
        fps_idx = torch.arange(0, xyz.shape[1])[None, ...].repeat(xyz.shape[0], 1).to(xyz.device) # (B, N)
        new_xyz = xyz
        nr = normals[:, :, None, :]

    idx = query_ball_point(radius, nsample, xyz, new_xyz, fps_idx)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)        # (B, npoint, nsample, C)
    d = grouped_xyz - new_xyz.view(B, S, 1, C)  # d = p_r - p_i  (B, npoint, nsample, 3)
    ni = index_points(normals, idx)

    nr_d = angle(nr, d)
    ni_d = angle(ni, d)
    nr_ni = angle(nr, ni)
    d_norm = torch.norm(d, dim=-1)

    xyz_feat = d  # (B, npoint, n_sample, 3)
    ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, npoint, n_sample, 4)

    if returnfps:
        return {'xyz': new_xyz, 'dxyz': xyz_feat, 'ppf': ppf_feat}, grouped_xyz, fps_idx
    else:
        return {'xyz': new_xyz, 'dxyz': xyz_feat, 'ppf': ppf_feat}




def gather_neighbour(inputs, neigh_idx):
    """ Gather the coordinates or features of neighboring points
        :param inputs: [B, N, C]
        :param neigh_idx: [B, N, nsample]

        :return: [B, N, nsample, C]
    """
    batch_size, num_points, dims = inputs.size()
    nsample = neigh_idx.shape[-1]
    neigh_idx = neigh_idx.reshape(batch_size, -1).unsqueeze(-1).repeat(1, 1, dims)  # [B, N*nsample, C]
    inputs = torch.gather(inputs, dim=1, index=neigh_idx)                           # [B, N*nsample, C]
    inputs = inputs.reshape(batch_size, num_points, nsample, dims)                  # [B, N, nsample, C]
    return inputs

def gather_neighbour_V2(inputs, neigh_idx):
    """ Gather the coordinates or features of neighboring points
    :param inputs: [B, C, N]
    :param neigh_idx: [B, N, nsample]

    :return: [B, C, N, nsample]
    """
    batch_size, dims, num_points = inputs.size()
    nsample = neigh_idx.shape[-1]
    neigh_idx = neigh_idx.reshape(batch_size, -1).unsqueeze(1).repeat(1, dims, 1)   # [B, C, N*nsample]
    inputs = torch.gather(inputs, dim=2, index=neigh_idx)                           # [B, C, N*nsample]
    inputs = inputs.reshape(batch_size, dims, num_points, nsample)                  # [B, C, N, nsample]
    return inputs

def gather_neighbour_V3(inputs, neigh_idx):
    """ Gather the coordinates or features of neighboring points
    :param inputs: [B, C, N]
    :param neigh_idx: [B, M]

    :return: [B, C, M]
    """
    dims = inputs.size()[1]
    neigh_idx = neigh_idx.unsqueeze(1).repeat(1, dims, 1)  # [B, C, M]
    inputs = torch.gather(inputs, dim=2, index=neigh_idx)  # [B, C, M]
    return inputs

def gather_neighbour_V4(inputs, neigh_idx):
    """ Gather the coordinates or features of neighboring points
    :param inputs: [B, N, C]
    :param neigh_idx: [B, M]

    :return: [B, M, C]
    """
    dims = inputs.size()[-1]
    neigh_idx = neigh_idx.unsqueeze(-1).repeat(1, 1, dims)  # [B, M, C]
    inputs = torch.gather(inputs, dim=1, index=neigh_idx)   # [B, M, C]
    return inputs

