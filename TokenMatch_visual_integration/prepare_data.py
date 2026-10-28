"""Create split-aware pair manifests; never turn official test pairs into training."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from audit_data import DEFAULT_DATA, TEST_SCENES


def lines(path):
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def indexed_paths(folder, name):
    result = []
    for text in lines(folder / name):
        p = Path(text)
        p = p if p.is_absolute() else folder / p
        if not p.is_file():
            raise FileNotFoundError(str(p))
        result.append(str(p.resolve()))
    return result


def reservoir_pairs(folder, count, seed):
    images = indexed_paths(folder, "images.txt")
    cameras = indexed_paths(folder, "calibration.txt")
    visibilities = indexed_paths(folder, "visibility.txt")
    if not (len(images) == len(cameras) == len(visibilities)):
        raise ValueError("Mismatched image/calibration/visibility lists: " + str(folder))
    import h5py
    poses = []
    for path in cameras:
        with h5py.File(path, 'r') as f:
            poses.append((np.asarray(f['R']).reshape(3, 3), np.asarray(f['T']).reshape(3)))
    rng = random.Random(seed)
    selected, candidates = [], 0
    for i, p in enumerate(visibilities):
        row = np.loadtxt(p).reshape(-1)
        if len(row) != len(images):
            raise ValueError("Visibility row has wrong length: " + p)
        for j in np.flatnonzero(row > 50):
            j = int(j)
            if i == j:
                continue
            ri, ti = poses[i]; rj, tj = poses[j]
            if np.linalg.norm(tj - (rj @ ri.T) @ ti) <= 1e-5:
                continue  # Match the original dump pipeline's degenerate-baseline exclusion.
            candidates += 1
            pair = (i, j)
            if len(selected) < count:
                selected.append(pair)
            else:
                k = rng.randrange(candidates)
                if k < count:
                    selected[k] = pair
    rng.shuffle(selected)
    records = []
    for i, j in selected:
        records.append({"split": folder.name, "scene": folder.parent.name,
                        "image0": images[i], "image1": images[j],
                        "camera0": cameras[i], "camera1": cameras[j],
                        "geometry_format": "oanet_centered_intrinsics"})
    return records


def write_manifest(path, records):
    if path.exists():
        existing = path.read_text()
        desired = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
        if existing != desired:
            raise RuntimeError("Manifest exists with different contents; use a new prefix: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def prepare(data_dir, train_count=10000, val_count=100, max_scenes=0, prefix="full", seed=42):
    if min(train_count, val_count) <= 0 or max_scenes < 0:
        raise ValueError('Pair counts must be positive and max_scenes nonnegative.')
    data_dir = Path(data_dir).resolve()
    roots = [data_dir / "repro_raw" / "yfcc100m", data_dir / "raw_data" / "yfcc100m"]
    folders = {}
    for root in roots:
        if root.is_dir():
            for scene in sorted(root.iterdir()):
                if scene.is_dir() and scene.name not in TEST_SCENES:
                    for split in ["train", "val"]:
                        folder = scene / split
                        if all((folder / x).is_file() for x in ["images.txt", "calibration.txt", "visibility.txt"]):
                            folders.setdefault((scene.name, split), folder)
    official = set(lines(Path(__file__).with_name("official_train_scenes.txt")))
    scenes = sorted({scene for scene, split in folders if scene in official and (scene, "train") in folders and (scene, "val") in folders})
    missing = sorted(official - set(scenes))
    if max_scenes:
        scenes = scenes[:max_scenes]
    output = data_dir / "repro_cache"
    output.mkdir(parents=True, exist_ok=True)
    report = {"test_scenes_excluded": sorted(TEST_SCENES), "training_scenes": scenes,
              "missing_official_training_scenes": missing,
              "seed": seed, "source_format": "OANet raw image/calibration/visibility lists",
              "requested_train_pairs_per_scene": train_count, "requested_val_pairs_per_scene": val_count,
              "manifests": {}, "counts": {}}
    if not scenes:
        report["status"] = "missing_training_or_validation_raw_data"
        (output / (prefix + "_preparation.json")).write_text(json.dumps(report, indent=2))
        raise RuntimeError("No non-test scene has BOTH train and val image/calibration/visibility lists. "
                           "Official test pairs will not be used for training. See repro_cache/archive_audit.json.")
    if not max_scenes and missing:
        report["status"] = "incomplete_official_training_scenes"
        (output / (prefix + "_preparation.json")).write_text(json.dumps(report, indent=2))
        raise RuntimeError("Full preparation requires all official training scenes; missing: " + ", ".join(missing))
    for split, count in [("train", train_count), ("val", val_count)]:
        records = []
        for i, scene in enumerate(scenes):
            print("Preparing", scene, split, flush=True)
            records += reservoir_pairs(folders[(scene, split)], count, seed + i * 2 + (split == "val"))
        if not records:
            raise RuntimeError("No eligible image pairs for " + split)
        name = output / (prefix + "_" + split + ".jsonl")
        write_manifest(name, records)
        report["manifests"][split] = str(name)
        report["counts"][split] = len(records)
    report["status"] = "ready"
    (output / (prefix + "_preparation.json")).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--train-pairs-per-scene", type=int, default=10000)
    ap.add_argument("--val-pairs-per-scene", type=int, default=100)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--prefix", default="full")
    a = ap.parse_args()
    print(json.dumps(prepare(a.data_dir, a.train_pairs_per_scene, a.val_pairs_per_scene, a.max_scenes, a.prefix), indent=2))
