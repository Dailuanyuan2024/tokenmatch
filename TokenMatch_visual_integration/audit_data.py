"""Audit the local OANet archive, optionally extract training/validation data safely."""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "datasets" / "YFCC100M")
TEST_SCENES = {"buckingham_palace", "notre_dame_front_facade", "reichstag", "sacre_coeur"}


def canonical_member(name):
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ".." in parts or "yfcc100m" not in parts:
        return None
    rel = parts[parts.index("yfcc100m"):]
    if len(rel) < 3 or rel[2] not in {"train", "val", "test"}:
        return None
    return Path(*rel)


def audit(data_dir, extract=False):
    data_dir = Path(data_dir).resolve()
    archive = data_dir / "oanet_parts" / "raw_data_complete.tar.gz"
    destination = data_dir / "repro_raw"
    output = data_dir / "repro_cache"
    output.mkdir(parents=True, exist_ok=True)
    report = {"archive": str(archive), "exists": archive.is_file(),
              "extract_requested": extract, "extraction_root": str(destination),
              "counts": {}, "bytes": {}, "scenes_by_split": {},
              "extracted_files": 0, "reused_files": 0,
              "real_training_started": False, "started_unix": time.time()}
    counts, sizes, scenes = Counter(), Counter(), {k: set() for k in ["train", "val", "test"]}
    examples = []
    if archive.is_file():
        print("Scanning archive headers; a 4 GB gzip archive can take several minutes.", flush=True)
        # Stream one pass; only regular train/val files are eligible for extraction.
        with tarfile.open(archive, mode="r|*") as tf:
            for i, member in enumerate(tf):
                rel = canonical_member(member.name)
                if rel is None:
                    if len(examples) < 12:
                        examples.append(member.name)
                    continue
                scene, split = rel.parts[1:3]
                counts[split] += 1
                sizes[split] += member.size
                scenes[split].add(scene)
                if i and i % 3000 == 0:
                    print("Archive members scanned:", i, "counts:", dict(counts), flush=True)
                if not extract or split == "test" or scene in TEST_SCENES:
                    continue
                if not member.isfile():
                    continue
                target = destination / rel
                # Do not follow pre-existing symbolic links when writing a member.
                if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != destination.parent):
                    raise RuntimeError("Refusing extraction through a symbolic link: " + str(target))
                if target.exists():
                    if not target.is_file() or target.stat().st_size != member.size:
                        raise RuntimeError("Existing extracted member has an unexpected size: " + str(target))
                    report["reused_files"] += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(target.parent).free < member.size + 2 * 1024 ** 3:
                    raise RuntimeError("Less than 2 GB disk reserve; extraction stopped without overwriting files.")
                stream = tf.extractfile(member)
                if stream is None:
                    raise RuntimeError("Cannot read archive member " + member.name)
                temp = target.with_name(target.name + ".tokenmatch-partial")
                try:
                    with temp.open("wb") as f:
                        shutil.copyfileobj(stream, f, 1024 * 1024)
                    if temp.stat().st_size != member.size:
                        raise RuntimeError("Incomplete archive member " + member.name)
                    temp.replace(target)
                finally:
                    temp.unlink(missing_ok=True)
                report["extracted_files"] += 1
    roots = [data_dir / "raw_data" / "yfcc100m", destination / "yfcc100m"]
    available = {split: [] for split in ["train", "val", "test"]}
    for root in roots:
        if root.is_dir():
            for scene in sorted(root.iterdir()):
                if scene.is_dir():
                    for split in available:
                        folder = scene / split
                        if folder.is_dir():
                            available[split].append(str(folder))
    report.update(counts=dict(counts), bytes=dict(sizes),
                  scenes_by_split={k: sorted(v) for k, v in scenes.items()},
                  other_member_examples=examples, available_folders=available,
                  finished_unix=time.time())
    filename = output / "archive_audit.json"
    filename.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Audit written:", filename, flush=True)
    print("Archive scenes by split:", {k: len(v) for k, v in scenes.items()}, flush=True)
    print("Available folders:", {k: len(v) for k, v in available.items()}, flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()
    audit(args.data_dir, args.extract)
