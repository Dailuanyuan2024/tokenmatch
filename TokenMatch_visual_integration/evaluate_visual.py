"""SIFT+TokenMatch evaluation using the existing project's pose/AUC functions."""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from pair_dataset import PairDataset, test_manifest
from project_runtime import DEFAULT_PROJECT, EXPERIMENT, evaluator, load_checkpoint, provenance, sha256, integration_hashes


def evaluate_grid(model, dataset, device, helpers, score_thresholds, limit=0, csv_path=None):
    model.eval()
    thresholds = [float(x) for x in score_thresholds]
    if not thresholds or not all(math.isfinite(x) for x in thresholds):
        raise ValueError('Choose at least one finite score threshold.')
    if csv_path and len(thresholds) != 1:
        raise ValueError('Per-pair CSV output requires a single threshold.')
    errors = [[] for _ in thresholds]
    rows = [[] for _ in thresholds]
    count = min(len(dataset), limit) if limit else len(dataset)
    if count <= 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    for i in range(count):
        failure = None
        try:
            item = dataset[i]
            inputs = {"xs": torch.from_numpy(item["xs"][None]).to(device),
                      "visual": torch.from_numpy(item["visual"][None]).to(device)}
            with torch.inference_mode():
                logits, _ = model(inputs)
            scores = logits[-1].detach().cpu().numpy().reshape(-1)
            if not np.isfinite(scores).all():
                raise RuntimeError('Non-finite network scores at pair ' + str(i))
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = item["Rs"], item["ts"]
        except torch.cuda.OutOfMemoryError:
            raise
        except (ValueError, FileNotFoundError, cv2.error) as exc:
            failure = str(exc)
        for j, threshold in enumerate(thresholds):
            row = {"pair_idx": i, "num_matches": 0, "num_kept": 0}
            if failure is None:
                keep = scores > threshold
                cv2.setRNGSeed(0)
                er, et = helpers.estimate_error(item["pixels0"][keep], item["pixels1"][keep],
                                                item["K0"], item["K1"], transform, 1.0, 0.99999)
                error = max(er, et)
                row.update(num_matches=len(scores), num_kept=int(keep.sum()),
                           err_R_deg=float(er), err_t_deg=float(et),
                           pose_error_deg=float(error), status="ok" if math.isfinite(error) else "pose_failure")
            else:
                error = float('inf')
                row.update(err_R_deg=error, err_t_deg=error, pose_error_deg=error,
                           status='input_failure:' + failure)
            errors[j].append(float(error))
            rows[j].append(row)
        if (i + 1) % 25 == 0 or i + 1 == count:
            print("Evaluated", i + 1, "/", count, 'threshold', thresholds[0], helpers.auc_pose(errors[0]), flush=True)
    if csv_path:
        csv_path = Path(csv_path)
        if csv_path.exists():
            raise FileExistsError(str(csv_path))
        with csv_path.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0][0]))
            writer.writeheader(); writer.writerows(rows[0])
    return [{"split": dataset.split, "count": count, "auc_percent": helpers.auc_pose(err),
             "score_threshold": threshold, "ransac_px": 1.0, "ransac_confidence": 0.99999,
             "pose_failures": sum(not math.isfinite(e) for e in err),
             "input_failures": sum(r["status"].startswith("input_failure") for r in rs),
             "average_matches": float(np.mean([r["num_matches"] for r in rs])),
             "average_kept": float(np.mean([r["num_kept"] for r in rs]))}
            for threshold, err, rs in zip(thresholds, errors, rows)]


def evaluate(model, dataset, device, helpers, score_threshold=2.0, limit=0, csv_path=None):
    return evaluate_grid(model, dataset, device, helpers, [score_threshold], limit, csv_path)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--manifest")
    ap.add_argument("--checkpoint")
    ap.add_argument("--variant", choices=["coordinate", "visual"], default="coordinate")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--threshold-grid", help="Only accepted for a validation manifest.")
    ap.add_argument("--threshold-file", help="Threshold selected on validation, required for final visual test.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    root = Path(args.project).resolve()
    data_root = root / "datasets/YFCC100M"
    cache = root / "cache/sift_visual_repro"
    out = Path(args.output_dir) if args.output_dir else root / "results" / ("sift_repro_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    if args.manifest:
        manifest = Path(args.manifest)
    else:
        manifest = cache / ("test_" + str(args.limit or "full") + ".jsonl")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        test_manifest(data_root, manifest, args.limit)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        root / "TokenMatch/core" / EXPERIMENT / "train/model_best_tokenmatch.pth")
    model, _ = load_checkpoint(root, checkpoint, args.variant, device)
    dataset = PairDataset(manifest, cache, num_keypoints=4000, max_side=1600, supervision=False)
    helpers = evaluator(root)
    threshold = args.threshold
    if args.threshold_file:
        selection = json.loads(Path(args.threshold_file).read_text())
        if selection.get("selection_split") != "val":
            raise ValueError("Threshold selection file must originate from validation.")
        if selection.get('checkpoint_sha256') != sha256(checkpoint) or selection.get('variant') != args.variant:
            raise ValueError("Threshold selection belongs to a different checkpoint or model variant.")
        threshold = float(selection["selected_threshold"])
    if args.variant == "visual" and dataset.split == "test" and not args.threshold_file:
        raise ValueError("Select a visual model threshold on validation first; pass --threshold-file.")
    if args.threshold_grid:
        if dataset.split != "val":
            raise ValueError("Threshold grids must not run on the official test split.")
        reports = evaluate_grid(model, dataset, device, helpers,
                                [float(x) for x in args.threshold_grid.split(',')], args.limit)
        best = max(reports, key=lambda r: (r['auc_percent']['AUC@5'], r['auc_percent']['AUC@10']))
        if any(r['input_failures'] for r in reports):
            raise RuntimeError('Fix validation input failures before selecting a threshold.')
        result = {"selection_split": "val", "selection_objective": "AUC@5, tie-break AUC@10",
                  "selected_threshold": best['score_threshold'], "reports": reports,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "variant": args.variant}
        filename = out / "threshold_selection.json"
    else:
        result = evaluate(model, dataset, device, helpers, threshold, args.limit, out / "pairs.csv")
        result.update(checkpoint=str(checkpoint), variant=args.variant,
                      threshold_source="validation" if args.threshold_file else "fixed_predeclared_2_or_user_value")
        filename = out / "summary.json"
    result.update(manifest=str(manifest.resolve()), manifest_sha256=sha256(manifest),
                  integration_source=integration_hashes(), opencv_version=cv2.__version__,
                  torch_version=torch.__version__,
                  source=provenance(root), feature_settings={'type': 'OpenCV SIFT',
                  'max_keypoints': 4000, 'longest_side': 1600, 'contrast_threshold': 1e-5,
                  'matching': 'one-way nearest neighbor, no ratio or mutual filter'})
    filename.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)
    print("Saved:", filename, flush=True)


if __name__ == "__main__":
    main()
