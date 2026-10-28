"""Inspect/extract raw training data, then run a real-data 20-step GPU smoke job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from audit_data import audit


def bundle(output, data):
    candidates = list(output.glob('*.json')) + list(output.glob('*.log'))
    candidates += list((output / 'train').glob('*.json')) + list((output / 'train').glob('*.jsonl'))
    candidates += list((data / 'repro_cache').glob('*preparation.json'))
    candidates += [data / 'repro_cache/archive_audit.json']
    target = output / 'server_run_report.zip'
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in candidates:
            if path.is_file():
                base = output if path.is_relative_to(output) else data
                z.write(path, ('run/' if base == output else 'data/') + str(path.relative_to(base)))
    print('Report ZIP (logs only; no weights or images):', target, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-archive", action="store_true", help="Use only already extracted training folders.")
    ap.add_argument('--install-missing', action='store_true')
    ap.add_argument('--launch-full', action='store_true', help='After the smoke passes, prepare full manifests and launch paired 500000-step jobs.')
    ap.add_argument('--control-device', default='cuda:1')
    ap.add_argument('--workers', type=int, default=4, help='DataLoader workers per full-training job; the smoke always uses 0.')
    args = ap.parse_args()
    root = Path(args.project).resolve()
    data = root / "datasets/YFCC100M"
    output = root / "repro_runs" / ("visual_smoke_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    status = {"project": str(root), "real_training_started": False,
              "kind": "real_data_smoke_not_a_benchmark_result"}
    try:
        from check_environment import check
        status['environment'] = check(args.install_missing)
        import torch
        device = torch.device(args.device)
        if device.type == 'cuda' and (not torch.cuda.is_available() or device.index is None or device.index >= torch.cuda.device_count()):
            raise RuntimeError('Requested CUDA device is unavailable: ' + args.device)
        if not args.skip_archive:
            status['archive_audit'] = audit(data, extract=True)
        from prepare_data import prepare
        info = prepare(data, train_count=4, val_count=2, max_scenes=2,
                       prefix="smoke_" + time.strftime("%Y%m%d_%H%M%S"))
        status['data_preparation'] = info
        command = [sys.executable, '-u', str(Path(__file__).with_name('train_visual.py')),
                   '--project', str(root), '--train-manifest', info['manifests']['train'],
                   '--val-manifest', info['manifests']['val'], '--run-dir', str(output / 'train'),
                   '--device', args.device, '--steps', '20', '--batch-size', '1',
                   '--keypoints', '2000', '--correspondences', '256', '--workers', '0',
                   '--save-every', '20', '--validate-every', '20', '--val-limit', '2',
                   '--print-every', '1', '--smoke']
        status['training_command'] = command
        status['training_child_launched'] = True
        print('Starting real-data smoke training:', ' '.join(command), flush=True)
        with (output / 'smoke_training.log').open('x') as log:
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in result.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            result.wait()
        if result.returncode:
            raise RuntimeError('Training child exited with status ' + str(result.returncode))
        status['status'] = 'passed'
        status['real_training_started'] = True
        status['completion'] = json.loads((output / 'train/completion.json').read_text())
        code = 0
    except Exception as exc:
        status.update(status='blocked_or_failed', error=str(exc))
        progress_file = output / 'train/training.jsonl'
        status['real_training_started'] = progress_file.is_file() and progress_file.stat().st_size > 0
        print('Stopped:', exc, flush=True)
        code = 2
    filename = output / 'status.json'
    filename.write_text(json.dumps(status, indent=2, default=str))
    print('Status report:', filename, flush=True)
    if code == 0 and args.launch_full:
        try:
            from launch_full import launch
            full = launch(root, output, args.device, args.control_device, args.workers)
            status['full_launch'] = full
            if full['status'] != 'launched':
                code = 2
        except Exception as exc:
            status['full_launch'] = {'status': 'blocked', 'error': str(exc)}
            print('Full training launch stopped:', exc, flush=True)
            code = 2
        filename.write_text(json.dumps(status, indent=2, default=str))
    elif code == 0:
        print('Smoke passed; full training has not been launched (use --launch-full to request it).', flush=True)
    bundle(output, data)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
