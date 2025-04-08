import numpy as np
import torch

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

@torch.jit.script
def slerp(val0, val1, blend):
    return (1.0 - blend) * val0 + blend * val1

@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def get_quat_yaw(quat):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_yaw

def get_quat_no_yaw(quat):
    quat_no_yaw = quat.clone().view(-1, 4)
    quat_no_yaw[:, 2] = 0.  # Set the z-component to zero
    quat_no_yaw = normalize(quat_no_yaw)  # Re-normalize to maintain a valid quaternion
    return quat_no_yaw

def quat_to_angle(q):
    """Compute the rotation angle (in radians) from a quaternion."""
    w = q[..., -1]  # Scalar part
    theta = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))  # Clamp to avoid NaN due to precision
    return theta


def quat_diff(q0, q1):
    """Compute the relative quaternion difference q_diff = q1 * q0^-1."""
    q0_inv = normalize(quat_conjugate(q0))
    return quat_mul(q1, q0_inv)


# @torch.jit.script
def quat_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""

    _EPS = torch.finfo(torch.float32).eps * 4.0

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    precision_error = 0.00001
    if torch.any(d.abs() > 1.0 + precision_error):
        raise ValueError(f"Error in Quaternion SLERP. Argument to acos is larger than {1.0 + precision_error}.")
    else:
        d = torch.clip(d, -1.0, 1.0)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    final = q0 * (torch.sin((1.0 - fraction) * angle) * isin) + q1 * (torch.sin(fraction * angle) * isin)
    out[final_mask] = final[final_mask]
    return out


def quat_standardize(q):
    """Returns a quaternion where q.w >= 0 to remove redundancy due to q = -q.

    Args:
      q: A quaternion to be standardized.

    Returns:
      A quaternion with q.w >= 0.

    """
    mask = q[..., -1] < 0
    q[mask] = -q[mask]
    return q

def quat_rotate_batch(q, v):
    shape = q.shape
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(-1, 1, 3), v.view(
            -1, 3, 1)).reshape(shape[:-1]).unsqueeze(-1) * 2.0
    return a + b + c

def quat_rotate_inverse_batch(q, v):
    shape = q.shape
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(-1, 1, 3), v.view(
            -1, 3, 1)).reshape(shape[:-1]).unsqueeze(-1) * 2.0
    return a - b + c


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower

# @torch.jit.script
def torch_rand_float_ring(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return torch.sqrt((upper ** 2 - lower ** 2) * torch.rand(*shape, device=device) + lower ** 2)