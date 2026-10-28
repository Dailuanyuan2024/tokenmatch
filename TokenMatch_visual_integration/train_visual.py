"""Train the active TokenMatch backbone on real SIFT image pairs, from scratch."""
import argparse
import functools
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pair_dataset import PairDataset, collate_pairs
from project_runtime import DEFAULT_PROJECT, configuration, evaluator, fresh_model, provenance, sha256, integration_hashes
from evaluate_visual import evaluate


def worker_seed(worker_id):
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value); np.random.seed(value)


def save_checkpoint(path, content):
    temporary = path.with_name(path.name + ".partial")
    torch.save(content, temporary)
    os.replace(temporary, path)


def train(args):
    project = Path(args.project).resolve()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the active environment.")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    config = configuration()
    model = fresh_model(project, args.variant, config).to(device)
    from loss import MatchLoss
    from warmupMultiStepLR import WarmupMultiStepLR
    objective = MatchLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = WarmupMultiStepLR(optimizer, [200000, 400000], gamma=0.4,
                                 warmup_iters=100000, warmup_factor=0.01,
                                 warmup_method="linear")
    cache = project / "cache/sift_visual_repro"
    train_data = PairDataset(args.train_manifest, cache, args.keypoints, 0, "train")
    val_data = PairDataset(args.val_manifest, cache, args.keypoints, 0, "val")
    train_images = {r[k] for r in train_data.records for k in ['image0', 'image1']}
    if any(r[k] in train_images for r in val_data.records for k in ['image0', 'image1']):
        raise RuntimeError("Training and validation manifests share an image.")
    run = Path(args.run_dir).resolve()
    if run.exists() and not args.resume:
        raise FileExistsError("Choose a fresh run directory or explicitly resume: " + str(run))
    run.mkdir(parents=True, exist_ok=True)
    metadata = dict(vars(args))
    metadata.update(config=vars(config), source=provenance(project),
                    integration_source=integration_hashes(),
                    train_manifest_sha256=sha256(args.train_manifest),
                    val_manifest_sha256=sha256(args.val_manifest),
                    train_pairs=len(train_data), val_pairs=len(val_data),
                    schedule={"warmup_steps": 100000, "warmup_factor": 0.01,
                              "milestones": [200000, 400000], "gamma": 0.4},
                    visual_input={"dim": 262, "mlp": [262, 128, 128],
                                  "fusion": "add after weights_init.conv before ge_cluster_token"},
                    real_data_smoke_only=bool(args.smoke))
    start, best = 0, -1.0
    checkpoint_file = run / "checkpoint.pth"
    if args.resume:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        previous = checkpoint['metadata']
        for field in ["variant", "train_manifest_sha256", "val_manifest_sha256", "correspondences",
                      "batch_size", "keypoints", "lr", "seed", "config", "schedule"]:
            if previous[field] != metadata[field]:
                raise RuntimeError("Resume configuration mismatch: " + field)
        if any(previous['source'][k]['sha256'] != v['sha256'] for k, v in metadata['source'].items()):
            raise RuntimeError('Reference source files changed since the checkpoint was written.')
        if previous.get('integration_source') != metadata['integration_source']:
            raise RuntimeError('Integration code changed since the checkpoint was written; retain the original experiment code.')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start, best = int(checkpoint['step']), float(checkpoint['best_auc5'])
        random.setstate(checkpoint['python_rng']); np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        # Mid-epoch DataLoader prefetch state is not serialized; record this limitation.
        metadata['resume_sampling_note'] = 'data order restarts at an epoch boundary; optimizer/scheduler/RNG are restored'
    else:
        (run / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str))
    generator = torch.Generator().manual_seed(args.seed + start)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == 'cuda',
                        timeout=180 if args.workers else 0,
                        generator=generator, worker_init_fn=worker_seed,
                        collate_fn=functools.partial(collate_pairs, max_correspondences=args.correspondences))
    iterator = iter(loader)
    helpers = evaluator(project)
    print("Training", args.variant, "on", device, "train pairs", len(train_data),
          "validation pairs", len(val_data), "run", run, flush=True)

    def state(step):
        return {"step": step, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "best_auc5": best, "metadata": metadata,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}

    with (run / "training.jsonl").open("a") as log:
        for step in range(start, args.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, matrices = model(batch)
            terms = [objective.run(step, batch, a, b) for a, b in zip(logits, matrices)]
            loss = sum(t[0] for t in terms)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss at step " + str(step))
            loss.backward()
            # Infinity means no clipping; use the combined check to avoid one GPU sync per parameter.
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            visual_grad = float(torch.stack([p.grad.detach().norm()
                                            for p in model.visual_encoder.parameters()
                                            if p.grad is not None]).norm().cpu()) if args.variant == 'visual' else None
            lr = optimizer.param_groups[0]['lr']
            optimizer.step(); scheduler.step()
            row = {"step": step + 1, "lr": lr, "loss": float(loss.detach().cpu()),
                   "visual_gradient_norm": visual_grad,
                   "geometry_loss_active": step >= config.loss_essential_init_iter,
                   "geometry_loss_per_stage": [t[1] for t in terms],
                   "classification_loss_per_stage": [t[2] for t in terms]}
            log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
            if step == start or (step + 1) % args.print_every == 0:
                print(row, flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                save_checkpoint(checkpoint_file, state(step + 1))
            if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
                report = evaluate(model, val_data, device, helpers, args.score_threshold, args.val_limit)
                report['step'] = step + 1
                (run / ("validation_" + str(step + 1) + ".json")).write_text(json.dumps(report, indent=2))
                if report['input_failures']:
                    raise RuntimeError("Validation input errors detected; inspect validation report.")
                score = report['auc_percent']['AUC@5']
                if score > best:
                    best = score
                    save_checkpoint(run / "model_best.pth", state(step + 1))
                save_checkpoint(checkpoint_file, state(step + 1))
    summary = {"status": "completed", "steps": args.steps, "variant": args.variant,
               "run_dir": str(run), "best_validation_auc5": best,
               "real_data_smoke_only": bool(args.smoke), "real_training_started": True}
    (run / "completion.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", choices=["coordinate", "visual"], default="visual")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--keypoints", type=int, default=2000)
    ap.add_argument("--correspondences", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--validate-every", type=int, default=10000)
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--score-threshold", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if min(args.steps, args.batch_size, args.keypoints, args.correspondences, args.save_every,
           args.validate_every, args.print_every, args.cpu_threads) <= 0 or args.workers < 0 or args.val_limit < 0:
        ap.error("Steps, batch size, feature counts, and intervals must be positive.")
    return args


if __name__ == "__main__":
    train(arguments())
