"""Real image SIFT/geometry data, with aligned features and explicit split checks."""
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import tempfile
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from visual_inputs import encode_matched_sift
from audit_data import TEST_SCENES


def skew(t):
    x, y, z = np.asarray(t).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def epipolar_error(x0, x1, e):
    p0 = np.column_stack([x0, np.ones(len(x0))])
    p1 = np.column_stack([x1, np.ones(len(x1))])
    ep0, etp1 = p0 @ e.T, p1 @ e
    residual = np.sum(p1 * ep0, axis=1) ** 2
    return residual * (1 / (np.sum(ep0[:, :2] ** 2, axis=1) + 1e-15)
                       + 1 / (np.sum(etp1[:, :2] ** 2, axis=1) + 1e-15))


def read_camera(path):
    with h5py.File(path, "r") as f:
        k, r, t, wh = [np.asarray(f[x], dtype=np.float64) for x in ["K", "R", "T", "imsize"]]
    k, r, t, wh = k.reshape(3, 3), r.reshape(3, 3), t.reshape(3), wh.reshape(-1)[:2]
    if np.any(wh <= 0) or not np.isfinite(k).all():
        raise ValueError("Invalid camera calibration: " + str(path))
    # OANet stores principal-point offsets from the image center.
    k = k.copy()
    k[0, 2] += (wh[0] - 1) / 2
    k[1, 2] += (wh[1] - 1) / 2
    return k, r, t, wh


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class PairDataset(Dataset):
    def __init__(self, manifest, cache, num_keypoints=2000, max_side=0, expected_split=None, supervision=True):
        self.manifest = Path(manifest).resolve()
        self.records = [json.loads(x) for x in self.manifest.read_text().splitlines() if x.strip()]
        if not self.records:
            raise ValueError("Empty pair manifest.")
        splits = {x.get("split") for x in self.records}
        if len(splits) != 1 or not splits.issubset({"train", "val", "test"}):
            raise ValueError("Manifest must have one explicit split.")
        self.split = next(iter(splits))
        if expected_split and self.split != expected_split:
            raise ValueError("Expected " + expected_split + ", got " + str(self.split))
        if self.split in {"train", "val"} and any(r.get("scene") in TEST_SCENES for r in self.records):
            raise ValueError("Official held-out test scene in training/validation manifest.")
        self.cache = Path(cache).resolve()
        self.num_keypoints, self.max_side = int(num_keypoints), int(max_side)
        self.supervision = supervision
        self._sift = None

    def __len__(self):
        return len(self.records)

    def features(self, image_path):
        p = Path(image_path)
        stat = p.stat()
        settings = [str(p.resolve()), stat.st_size, stat.st_mtime_ns,
                    self.num_keypoints, self.max_side, cv2.__version__, "sift_ct1e-5_v2"]
        key = hashlib.sha256(json.dumps(settings).encode()).hexdigest()
        filename = self.cache / "sift" / key[:2] / (key + ".npz")
        if filename.is_file():
            with np.load(filename, allow_pickle=False) as f:
                return {k: f[k] for k in f.files}
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError("Cannot read image " + str(p))
        original_hw = np.asarray(image.shape[:2], dtype=np.int32)
        if self.max_side:
            scale = self.max_side / max(image.shape[:2])
            width, height = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        if self._sift is None:
            cv2.setNumThreads(1)
            self._sift = cv2.SIFT_create(nfeatures=self.num_keypoints, contrastThreshold=1e-5)
        points, desc = self._sift.detectAndCompute(image, None)
        if desc is None or len(points) < 16:
            raise ValueError("Fewer than 16 SIFT features in " + str(p))
        # Enforce an exact upper bound, including detector response ties.
        order = sorted(range(len(points)), key=lambda i: (-points[i].response, points[i].pt[0], points[i].pt[1]))[:self.num_keypoints]
        kp = np.asarray([[*points[i].pt, points[i].size, points[i].angle] for i in order], dtype=np.float32)
        result = {"keypoints": kp, "descriptors": desc[order].astype(np.float32),
                  "hw": np.asarray(image.shape[:2], dtype=np.int32), "original_hw": original_hw}
        atomic_npz(filename, **result)
        return result

    def __getitem__(self, index):
        record = self.records[index]
        f0, f1 = self.features(record["image0"]), self.features(record["image1"])
        if record["geometry_format"] == "oanet_centered_intrinsics":
            k0, r0, t0, wh0 = read_camera(record["camera0"])
            k1, r1, t1, wh1 = read_camera(record["camera1"])
            r = r1 @ r0.T
            t = t1 - r @ t0
        elif record["geometry_format"] == "pixel_K_relative_pose":
            k0, k1 = np.asarray(record["K0"]), np.asarray(record["K1"])
            r, t = np.asarray(record["R"]), np.asarray(record["t"])
            wh0, wh1 = f0["original_hw"][::-1], f1["original_hw"][::-1]
        else:
            raise ValueError("Unknown camera convention.")
        if np.linalg.norm(t) < 1e-8 or not np.isfinite(t).all():
            raise ValueError("Zero or invalid camera translation in pair " + str(index))
        t = t / np.linalg.norm(t)
        k0, k1 = np.array(k0, dtype=np.float64, copy=True), np.array(k1, dtype=np.float64, copy=True)
        k0[:2] *= (f0["hw"][::-1] / wh0)[:, None]
        k1[:2] *= (f1["hw"][::-1] / wh1)[:, None]
        d0, d1 = f0["descriptors"], f1["descriptors"]
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.match(d0, d1)
        if len(matches) != len(d0):
            raise ValueError("Nearest-neighbor matching returned incomplete rows.")
        matches = sorted(matches, key=lambda m: m.queryIdx)
        indices = np.asarray([m.trainIdx for m in matches], dtype=np.int64)
        kp0, kp1 = f0["keypoints"], f1["keypoints"][indices]
        p0, p1 = kp0[:, :2], kp1[:, :2]
        x0 = np.column_stack([p0, np.ones(len(p0))]) @ np.linalg.inv(k0).T
        x1 = np.column_stack([p1, np.ones(len(p1))]) @ np.linalg.inv(k1).T
        x0, x1 = x0[:, :2] / x0[:, 2:], x1[:, :2] / x1[:, 2:]
        visual = encode_matched_sift(d0, d1[indices], kp0[:, 2], kp1[:, 2], kp0[:, 3], kp1[:, 3], f0["hw"], f1["hw"])
        result = {"xs": np.concatenate([x0, x1], axis=1).astype(np.float32)[None],
                "visual": visual,
                "Rs": r.astype(np.float32), "ts": t.astype(np.float32),
                "K0": k0.astype(np.float32),
                "K1": k1.astype(np.float32), "pixels0": p0, "pixels1": p1,
                "index": index}
        if self.supervision:
            e = skew(t) @ r
            e /= np.linalg.norm(e)
            y = epipolar_error(x0, x1, e)
            grid = np.arange(-1, 1, 0.1, dtype=np.float32)
            gx, gy = np.meshgrid(grid, grid)
            points = np.column_stack([gx.ravel(), gy.ravel()]).reshape(1, -1, 2)
            v0, v1 = cv2.correctMatches(e, points.copy(), points.copy())
            virtual = np.concatenate([v0[0], v1[0]], axis=1)
            if not np.isfinite(virtual).all():
                raise ValueError("Non-finite virtual epipolar correspondences.")
            result.update(ys=y.astype(np.float32)[:, None], virtPts=virtual.astype(np.float32))
        return result


def collate_pairs(batch, max_correspondences=2000, random_subsample=True):
    count = min(max_correspondences, min(b["xs"].shape[1] for b in batch))
    output = {k: [] for k in ["xs", "visual", "ys", "Rs", "ts", "virtPts", "K0", "K1"]}
    for sample in batch:
        n = sample["xs"].shape[1]
        idx = np.random.choice(n, count, replace=False) if random_subsample and n > count else np.arange(count)
        for key in output:
            array = sample[key]
            output[key].append(array[:, idx] if key == "xs" else array[idx] if key in {"visual", "ys"} else array)
    return {k: torch.from_numpy(np.stack(v)).float() for k, v in output.items()}


def test_manifest(data_dir, output, limit=0):
    data_dir = Path(data_dir).resolve()
    root = data_dir / "raw_data" / "yfcc100m"
    records = []
    for line in (data_dir / "yfcc_test_pairs_with_gt.txt").read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        # Exactly the same 38-field format used by the existing evaluator.
        if len(fields) != 38:
            raise ValueError("Unsupported test pair row with " + str(len(fields)) + " fields.")
        if int(fields[2]) or int(fields[3]):
            raise ValueError("Nonzero test image rotations require explicit SIFT rotation support.")
        values = np.asarray(fields[4:], dtype=np.float64)
        if values.size != 34:
            raise ValueError("Expected 18 intrinsics and 16 relative-pose entries.")
        k0, k1, transform = values[:9].reshape(3, 3), values[9:18].reshape(3, 3), values[18:].reshape(4, 4)
        if fields[0].split('/')[0] not in TEST_SCENES or fields[1].split('/')[0] not in TEST_SCENES:
            raise ValueError("Unexpected test scene.")
        records.append({"split": "test", "scene": fields[0].split('/')[0],
                        "image0": str(root / fields[0]), "image1": str(root / fields[1]),
                        "K0": k0.tolist(), "K1": k1.tolist(), "R": transform[:3, :3].tolist(),
                        "t": transform[:3, 3].tolist(), "geometry_format": "pixel_K_relative_pose"})
        if limit and len(records) >= limit:
            break
    from prepare_data import write_manifest
    write_manifest(Path(output), records)
    return len(records)
