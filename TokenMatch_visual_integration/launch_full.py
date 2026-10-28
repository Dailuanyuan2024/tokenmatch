"""Launch paired from-scratch training only after a real-data smoke job passes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from prepare_data import prepare


def launch(project, smoke_run, visual_device='cuda:0', coordinate_device='cuda:1', workers=4):
    import torch
    project, smoke_run = Path(project).resolve(), Path(smoke_run).resolve()
    smoke = json.loads((smoke_run / 'status.json').read_text())
    if smoke.get('status') != 'passed' or not smoke.get('real_training_started'):
        raise RuntimeError('A passed real-data smoke run is required before this launcher.')
    if Path(smoke['project']).resolve() != project:
        raise RuntimeError('Smoke job belongs to a different project.')
    devices = [torch.device(x) for x in [visual_device, coordinate_device]]
    if any(x.type != 'cuda' or x.index is None or x.index >= torch.cuda.device_count() for x in devices):
        raise ValueError('Choose two valid explicit CUDA devices, such as cuda:0 and cuda:1.')
    if devices[0] == devices[1]:
        raise ValueError('Use different GPUs for concurrent paired training, or launch the train script sequentially.')
    data = project / 'datasets/YFCC100M'
    manifests = prepare(data, prefix='full')
    group = project / 'repro_runs' / ('paired_full_' + time.strftime('%Y%m%d_%H%M%S'))
    group.mkdir(parents=True, exist_ok=False)
    report = {'project': str(project), 'smoke_run': str(smoke_run), 'manifests': manifests,
              'same_data_and_training_budget': True, 'jobs': [], 'status': 'launching'}
    processes = []
    for variant, device in [('visual', visual_device), ('coordinate', coordinate_device)]:
        run, log = group / variant, group / (variant + '.log')
        command = [sys.executable, '-u', str(Path(__file__).with_name('train_visual.py')),
                   '--project', str(project), '--train-manifest', manifests['manifests']['train'],
                   '--val-manifest', manifests['manifests']['val'], '--run-dir', str(run),
                   '--variant', variant, '--device', device, '--workers', str(workers)]
        with log.open('x') as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        report['jobs'].append({'variant': variant, 'device': device, 'pid': process.pid,
                               'run_dir': str(run), 'log': str(log), 'command': command})
        (group / 'launch.json').write_text(json.dumps(report, indent=2))
    # Detect immediate import/configuration errors; background status is not convergence.
    time.sleep(2)
    for item, process in zip(report['jobs'], processes):
        item['exit_status_at_launch_check'] = process.poll()
    report['status'] = 'launched' if all(p.poll() is None for p in processes) else 'child_failed_at_launch'
    (group / 'launch.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print('Full-run logs are in:', group, flush=True)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--smoke-run', required=True)
    ap.add_argument('--visual-device', default='cuda:0')
    ap.add_argument('--coordinate-device', default='cuda:1')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    launch(args.project, args.smoke_run, args.visual_device, args.coordinate_device, args.workers)
