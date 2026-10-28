"""Explicit SIFT input convention for this reconstruction, not a recovered config."""
import numpy as np

VISUAL_DIM = 262


def _endpoint(descriptors, scales, angles_degrees, image_hw):
    desc = np.asarray(descriptors, dtype=np.float32)
    if desc.ndim != 2 or desc.shape[1] != 128 or not np.isfinite(desc).all():
        raise ValueError("SIFT descriptors must be a finite N x 128 array.")
    n = len(desc)
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    angles = np.asarray(angles_degrees, dtype=np.float32).reshape(-1)
    hw = np.asarray(image_hw, dtype=np.float32).reshape(-1)
    if len(scales) != n or len(angles) != n or hw.size != 2:
        raise ValueError("Descriptor rows, scales, angles, and image size must align.")
    if (not np.isfinite(scales).all() or np.any(scales <= 0)
            or not np.isfinite(angles).all() or np.any(angles < 0)
            or np.any(angles >= 360) or not np.isfinite(hw).all()
            or np.any(hw <= 0)):
        raise ValueError("Expected positive SIFT sizes, angles in [0,360), and image H,W.")
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero SIFT descriptors are not valid visual evidence.")
    desc = desc / norms
    theta = np.deg2rad(angles)
    # image_hw and scales must refer to the same image resolution.
    size = np.log(scales / float(hw.max()))[:, None]
    orientation = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    return desc, size, orientation


def encode_matched_sift(desc0, desc1, size0, size1, angle0, angle1, hw0, hw1):
    """Rows MUST follow the exact same match indices/order as data['xs'].

    Layout: d0[128], d1[128], log(size0/L0), log(size1/L1),
    sin(theta0), cos(theta0), sin(theta1), cos(theta1).
    Subsampling, permutation, and filtering must be applied to xs and visual together.
    """
    d0, s0, a0 = _endpoint(desc0, size0, angle0, hw0)
    d1, s1, a1 = _endpoint(desc1, size1, angle1, hw1)
    if len(d0) != len(d1):
        raise ValueError("Both endpoints must describe the same matched rows.")
    out = np.concatenate([d0, d1, s0, s1, a0, a1], axis=1).astype(np.float32)
    if out.shape[1] != VISUAL_DIM or not np.isfinite(out).all():
        raise ValueError("Invalid encoded visual inputs.")
    return out
