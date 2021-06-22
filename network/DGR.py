import numpy as np
import open3d as o3d
import torch
import torch.optim as optim

# Feature-based registrations in Open3D
def registration_ransac_based_on_feature_matching(pcd0, pcd1, feats0, feats1,
                                                  distance_threshold, num_iterations):
    assert feats0.shape[1] == feats1.shape[1]

    source_feat = o3d.registration.Feature()
    source_feat.resize(feats0.shape[1], len(feats0))
    source_feat.data = feats0.astype('d').transpose()

    target_feat = o3d.registration.Feature()
    target_feat.resize(feats1.shape[1], len(feats1))
    target_feat.data = feats1.astype('d').transpose()

    result = o3d.registration.registration_ransac_based_on_feature_matching(
                pcd0, pcd1, source_feat, target_feat, distance_threshold,
                o3d.registration.TransformationEstimationPointToPoint(False), 4,
                [o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                o3d.registration.RANSACConvergenceCriteria(num_iterations, 1000))
    return result.transformation

def registration_ransac_based_on_correspondence(pcd0, pcd1, idx0, idx1,
                                                distance_threshold, num_iterations):
    corres = np.stack((idx0, idx1), axis=1)
    corres = o3d.utility.Vector2iVector(corres)

    result = o3d.registration.registration_ransac_based_on_correspondence(
                pcd0, pcd1, corres, distance_threshold,
                o3d.registration.TransformationEstimationPointToPoint(False), 4,
                o3d.registration.RANSACConvergenceCriteria(4000000, num_iterations))

    return result.transformation


class HighDimSmoothL1Loss:
    def __init__(self, weights, quantization_size=1, eps=np.finfo(np.float32).eps):
        self.eps = eps
        self.quantization_size = quantization_size
        self.weights = weights
        if self.weights is not None:
            self.w1 = weights.sum()

    def __call__(self, X, Y):
        sq_dist = torch.sum(((X - Y) / self.quantization_size)**2, dim=1, keepdim=True)
        use_sq_half = 0.5 * (sq_dist < 1).float()

        loss = (0.5 - use_sq_half) * (torch.sqrt(sq_dist + self.eps) -
                                    0.5) + use_sq_half * sq_dist

        if self.weights is None:
            return loss.mean()
        else:
            return (loss * self.weights).sum() / self.w1


def ortho2rotation(poses):
    r"""
    poses: batch x 6
    """
    def normalize_vector(v):
        r"""
        Batch x 3
        """
        v_mag = torch.sqrt((v**2).sum(1, keepdim=True))
        v_mag = torch.clamp(v_mag, min=1e-8)
        v = v / v_mag
        return v

    def cross_product(u, v):
        r"""
        u: batch x 3
        v: batch x 3
        """
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        i = i[:, None]
        j = j[:, None]
        k = k[:, None]
        return torch.cat((i, j, k), 1)

    def proj_u2a(u, a):
        r"""
        u: batch x 3
        a: batch x 3
        """
        inner_prod = (u * a).sum(1, keepdim=True)
        norm2 = (u**2).sum(1, keepdim=True)
        norm2 = torch.clamp(norm2, min=1e-8)
        factor = inner_prod / norm2
        return factor * u

    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw - proj_u2a(x, y_raw))
    z = cross_product(x, y)

    x = x[:, :, None]
    y = y[:, :, None]
    z = z[:, :, None]
    return torch.cat((x, y, z), 2)


class Transformation(torch.nn.Module):
    def __init__(self, R_init=None, t_init=None):
        torch.nn.Module.__init__(self)
        rot_init = torch.rand(1, 6)
        trans_init = torch.zeros(1, 3)
        if R_init is not None:
            rot_init[0, :3] = R_init[:, 0]
            rot_init[0, 3:] = R_init[:, 1]
        if t_init is not None:
            trans_init[0] = t_init

        self.rot6d = torch.nn.Parameter(rot_init)
        self.trans = torch.nn.Parameter(trans_init)

    def forward(self, points):
        """
        points: [1, N, 3]
        """
        rot_mat = ortho2rotation(self.rot6d)[0]
        points_out = points[0, :, :] @ rot_mat.t() + self.trans[0]

        return points_out[None, :, :]


def pose_optim(pose, loss_fn, break_threshold_ratio=1e-4, max_break_count=20, stat_freq=20, verbose=False):
    R = pose[:3, :3]
    t = pose[:3, 3]
    transformation = Transformation(R, t).cuda()

    optimizer = optim.Adam(transformation.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    loss_prev = loss_fn(transformation(points), trans_points).item()

    # Transform points
    break_counter = 0
    for i in range(max_iter):
        new_points = transformation(points)
        loss = loss_fn(new_points, trans_points)
        if loss.item() < 1e-7: break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % stat_freq == 0 and verbose:
            print(i, scheduler.get_lr(), loss.item())

        if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
            break_counter += 1
            if break_counter >= max_break_count: break

        loss_prev = loss.item()

    rot6d = transformation.rot6d.detach()
    trans = transformation.trans.detach()

    opt_result = {'iterations': i, 'loss': loss.item(), 'break_count': break_counter}

    return ortho2rotation(rot6d)[0], trans, opt_result

def GlobalRegistration(points,
        trans_points,
        weights=None,
        max_iter=1000,
        verbose=False,
        stat_freq=20,
        max_break_count=20,
        break_threshold_ratio=1e-5,
        loss_fn=None,
        quantization_size=1):
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()

    if isinstance(trans_points, np.ndarray):
        trans_points = torch.from_numpy(trans_points).float()

    if loss_fn is None:
        if weights is not None:
            weights.requires_grad = False
        loss_fn = HighDimSmoothL1Loss(weights, quantization_size)

    if weights is None:
        # Get the initialization using https://ieeexplore.ieee.org/document/88573
        R, t = argmin_se3_squared_dist(points, trans_points)
    else:
        R, t = weighted_procrustes(points, trans_points, weights, loss_fn.eps)

    transformation = Transformation(R, t).to(points.device)

    optimizer = optim.Adam(transformation.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    loss_prev = loss_fn(transformation(points), trans_points).item()
    break_counter = 0

    # Transform points
    for i in range(max_iter):
        new_points = transformation(points)
        loss = loss_fn(new_points, trans_points)
        if loss.item() < 1e-7:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % stat_freq == 0 and verbose:
            print(i, scheduler.get_lr(), loss.item())

        if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
            break_counter += 1
            if break_counter >= max_break_count:
                break

        loss_prev = loss.item()

    rot6d = transformation.rot6d.detach()
    trans = transformation.trans.detach()

    opt_result = {'iterations': i, 'loss': loss.item(), 'break_count': break_counter}

    return ortho2rotation(rot6d)[0], trans, opt_result


class DGR:
    def __init__(self):
        self.voxel_size = 0.3

        # Safeguard
        self.safeguard_method = 'correspondence'  # correspondence, feature_matching
        # Final tuning
        self.use_icp = True

        # Misc
        self.feat_timer = Timer()
        self.reg_timer = Timer()

    def safeguard_registration(self, pcd0, pcd1, idx0, idx1, feats0, feats1,
                             distance_threshold, num_iterations):
        if self.safeguard_method == 'correspondence':
            T = registration_ransac_based_on_correspondence(pcd0,
                                                            pcd1,
                                                            idx0.cpu().numpy(),
                                                            idx1.cpu().numpy(),
                                                            distance_threshold,
                                                            num_iterations=num_iterations)
        elif self.safeguard_method == 'fcgf_feature_matching':
            T = registration_ransac_based_on_feature_matching(pcd0, pcd1,
                                                            feats0.cpu().numpy(),
                                                            feats1.cpu().numpy(),
                                                            distance_threshold,
                                                            num_iterations)
        else:
            raise ValueError('Undefined')
        return T

    def register(xyz0, xyz1, inlier_thr=0.00):
        # Step 2: Coarse correspondences
        corres_idx0, corres_idx1 = self.fcgf_feature_matching(fcgf_feats0, fcgf_feats1)

        T = np.identity(4)
        if wsum >= wsum_threshold:
            try:
                rot, trans, opt_output = GlobalRegistration(xyz0[corres_idx0],
                                                            xyz1[corres_idx1],
                                                            weights=weights.detach().cpu(),
                                                            break_threshold_ratio=1e-4,
                                                            quantization_size=2 * self.voxel_size,
                                                            verbose=False)
                T[0:3, 0:3] = rot.detach().cpu().numpy()
                T[0:3, 3] = trans.detach().cpu().numpy()
                dgr_time = self.reg_timer.toc()
                print(f'=> DGR takes {dgr_time:.2} s')

            except RuntimeError:
                # Will directly go to Safeguard
                print('###############################################')
                print('# WARNING: SVD failed, weights sum: ', wsum)
                print('# Falling back to Safeguard')
                print('###############################################')

        else:
            # > Case 1: Safeguard RANSAC
            pcd0 = make_open3d_point_cloud(xyz0)
            pcd1 = make_open3d_point_cloud(xyz1)
            T = self.safeguard_registration(pcd0,
                                            pcd1,
                                            corres_idx0,
                                            corres_idx1,
                                            feats0,
                                            feats1,
                                            2 * self.voxel_size,
                                            num_iterations=80000)
            safeguard_time = self.reg_timer.toc()
            print(f'=> Safeguard takes {safeguard_time:.2} s')

        if self.use_icp:
            T = o3d.registration.registration_icp(
                        make_open3d_point_cloud(xyz0),
                        make_open3d_point_cloud(xyz1), self.voxel_size * 2, T,
                        o3d.registration.TransformationEstimationPointToPoint()).transformation
        return T


