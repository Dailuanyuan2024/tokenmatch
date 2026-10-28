#!/usr/bin/env python3
"""Reusable relative-pose evaluation utilities for TokenMatch.

Extracted from the supplied evaluation runner without changing its pose
estimation, geometric normalization, or AUC definitions. No external matcher model,
matcher repository, or experiment-specific cache is required.

Only call these routines with the same camera-coordinate conventions and
RANSAC settings as the intended evaluation protocol.
"""
from pathlib import Path
import math

import cv2
import numpy as np
import torch

def load_pairs(path):
    with path.open() as f:
        for idx, line in enumerate(f):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 38:
                raise ValueError(f"Pair line {idx+1}: expected 38 fields, got {len(fields)}")
            a, b = fields[:2]
            rot0, rot1 = map(int, fields[2:4])
            K0 = np.asarray(fields[4:13], dtype=np.float64).reshape(3, 3)
            K1 = np.asarray(fields[13:22], dtype=np.float64).reshape(3, 3)
            T = np.asarray(fields[22:38], dtype=np.float64).reshape(4, 4)
            yield idx, a, b, rot0, rot1, K0, K1, T


def resize_rotate_image(path, rotation, min_side, device):
    # Explicit rotation from test-pair metadata. EXIF handling should be checked
    # against the original evaluation when image files contain orientation tags.
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    h, w = img.shape
    s = float(min_side) / min(h, w)
    out_w = max(8, round(w * s / 8) * 8)
    out_h = max(8, round(h * s / 8) * 8)
    if out_w != w or out_h != h:
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if rotation:
        img = np.rot90(img, k=rotation).copy()
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
    return tensor, (h, w), (out_h, out_w), img.shape


def resized_rotated_K(K, orig_hw, resized_hw, rotated_hw, rot):
    oh, ow = orig_hw
    h, w = resized_hw
    KK = np.diag([w / ow, h / oh, 1.0]) @ K
    fx, fy, cx, cy = KK[0, 0], KK[1, 1], KK[0, 2], KK[1, 2]
    if rot == 1:  # 90 degrees CCW
        return np.array([[fy, 0, cy], [0, fx, w - 1 - cx], [0, 0, 1]], dtype=np.float64)
    if rot == 2:
        return np.array([[fx, 0, w - 1 - cx], [0, fy, h - 1 - cy], [0, 0, 1]], dtype=np.float64)
    if rot == 3:
        return np.array([[fy, 0, h - 1 - cy], [0, fx, cx], [0, 0, 1]], dtype=np.float64)
    return KK


def adjusted_gt(T, rot0, rot1):
    def plane_rotation(rot):
        theta = np.deg2rad([0, 270, 180, 90][rot])
        cc, ss = math.cos(theta), math.sin(theta)
        P = np.eye(4)
        P[:2, :2] = [[cc, -ss], [ss, cc]]
        return P
    return (plane_rotation(rot1) @ T) @ np.linalg.inv(plane_rotation(rot0))


def estimate_error(x0, x1, K0, K1, T, px_thresh, prob):
    if len(x0) < 5:
        return float("inf"), float("inf")
    p0 = (x0.astype(np.float64) - K0[:2, 2]) / np.diag(K0)[:2]
    p1 = (x1.astype(np.float64) - K1[:2, 2]) / np.diag(K1)[:2]
    # Use SuperGlue's focal normalization convention to assist comparison.
    f_mean = (K0[0, 0] + K1[1, 1]) / 2.0
    try:
        E, inlier_mask = cv2.findEssentialMat(
            p0, p1, np.eye(3), method=cv2.RANSAC,
            prob=prob, threshold=px_thresh / f_mean)
        if E is None or inlier_mask is None or E.shape[0] % 3:
            return float("inf"), float("inf")
        best_n = -1
        best_R = best_t = None
        for candidate in np.split(E, E.shape[0] // 3):
            rec = cv2.recoverPose(candidate, p0, p1, np.eye(3),
                                     1e9, mask=inlier_mask.copy())
            n, R, t = rec[:3]
            if n > best_n:
                best_n, best_R, best_t = n, R, t.reshape(3)
        if best_R is None:
            return float("inf"), float("inf")
        c = np.clip((np.trace(best_R.T @ T[:3, :3]) - 1) / 2, -1, 1)
        er = float(np.rad2deg(np.arccos(c)))
        truth_t = T[:3, 3]
        nt = np.linalg.norm(truth_t) * np.linalg.norm(best_t)
        if nt < 1e-12:
            return float("inf"), float("inf")
        ct = np.clip(float(np.dot(best_t, truth_t) / nt), -1, 1)
        et = float(np.rad2deg(np.arccos(ct)))
        et = min(et, 180 - et)
        return er, et
    except (cv2.error, ValueError, FloatingPointError):
        return float("inf"), float("inf")


def auc_pose(errors, thresholds=(5, 10, 20)):
    a = np.sort(np.asarray(errors, dtype=np.float64))
    if not len(a):
        return {f"AUC@{t}": None for t in thresholds}
    n = len(a)
    result = {}
    for th in thresholds:
        within = a[a < th]
        # Integrate the recall step curve, starting at (0,0).
        xs = np.r_[0.0, within, float(th)]
        ys = np.r_[0.0, np.arange(1, len(within)+1)/n,
                   (len(within)/n if len(within) else 0.0)]
        result[f"AUC@{th}"] = 100.0 * float(np.trapz(ys, x=xs) / th)
    return result


def mean_finite(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def normalized_input(p0, p1, K0, K1):
    """Convert paired pixel points to per-camera normalized coordinates.

    Verify the coordinate convention against the target TokenMatch dataset
    before using this helper for a reproduction claim.
    """
    if len(p0) != len(p1) or p0.ndim != 2 or p0.shape[1] != 2:
        raise ValueError(f"Bad match coordinates shapes: {p0.shape} and {p1.shape}")
    if (K0[0, 0] <= 0 or K0[1, 1] <= 0 or K1[0, 0] <= 0 or K1[1, 1] <= 0):
        raise ValueError("Non-positive focal length in cached intrinsics")
    a = (p0.astype(np.float64) - K0[:2, 2]) / np.array([K0[0, 0], K0[1, 1]])
    b = (p1.astype(np.float64) - K1[:2, 2]) / np.array([K1[0, 0], K1[1, 1]])
    xs = np.concatenate((a, b), axis=1).astype(np.float32)
    if not np.isfinite(xs).all():
        raise ValueError("NaN or Inf in normalized match coordinates")
    return xs


def mean_or_none(data):
    data = [float(x) for x in data if math.isfinite(float(x))]
    return float(np.mean(data)) if data else None
