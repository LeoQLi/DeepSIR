""" 3-d rigid body transformation group
"""
import torch


def identity(batch_size):
    return torch.eye(3, 4)[None, ...].repeat(batch_size, 1, 1)


def inverse(Rt):
    """ Returns the inverse of the SE3 transform

    Args:
        Rt: (B, 3/4, 4) transform

    Returns:
        (B, 3, 4) matrix containing the inverse

    """
    # Compute inverse
    rot = Rt[..., 0:3, 0:3]
    trans = Rt[..., 0:3, 3]
    inverse_transform = torch.cat([rot.transpose(-1, -2), rot.transpose(-1, -2) @ -trans[..., None]], dim=-1)

    return inverse_transform


def concatenate(a, b):
    """Concatenate two SE3 transforms,
    i.e. return a@b (but note that our SE3 is represented as a 3x4 matrix)

    Args:
        a: (B, 3/4, 4)
        b: (B, 3/4, 4)

    Returns:
        (B, 3/4, 4)
    """
    rot1   = a[..., :3, :3]
    trans1 = a[..., :3, 3]
    rot2   = b[..., :3, :3]
    trans2 = b[..., :3, 3]

    rot_cat = rot1 @ rot2
    trans_cat = rot1 @ trans2[..., None] + trans1[..., None]
    concatenated = torch.cat([rot_cat, trans_cat], dim=-1)

    return concatenated


def transform(Rt, a, normals=None):
    """ Applies the SE3 transform

    Args:
        Rt: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
        a: Points to be transformed (N, 3) or (B, N, 3)
        normals: (Optional). If provided, normals will be transformed

    Returns:
        transformed points of size (N, 3) or (B, N, 3)

    """
    R = Rt[..., :3, :3]  # (B, 3, 3)
    t = Rt[..., :3, 3]   # (B, 3)

    if len(Rt.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + t[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + t  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b


def transform_V2(Rt, a, normals=None):
    """ Applies the SE3 transform

    Args:
        Rt: SE3 transformation matrix of size (B, 3/4, 4)
        a: Points to be transformed (B, 3, N)
        normals: (Optional). If provided, normals will be transformed (B, 3, N)

    Returns:
        transformed points of size (B, 3, N)

    """
    R = Rt[:, :3, :3]  # (B, 3, 3)
    t = Rt[:, :3, 3]   # (B, 3)

    b = torch.matmul(R, a) + t[:, :, None]

    if normals is not None:
        rotated_normals = R @ normals
        return b, rotated_normals
    else:
        return b