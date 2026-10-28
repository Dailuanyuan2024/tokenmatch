"""Evaluate a completed paired run; select thresholds on validation, then freeze them."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

DEFAULT_GRID = '0,0.5,1,1.5,2,2.1,2.2,2.3,2.4,2.5,3'


def finish(project, group=None, device='cuda:0', grid=DEFAULT_GRID):
    project = Path(project).resolve()
    if group is None:
        groups = sorted((project / 'repro_runs').glob('paired_full_*'))
        if not groups:
            raise RuntimeError('No paired full-training run was found.')
        group = groups[-1]
    group = Path(group).resolve()
    launch = json.loads((group / 'launch.json').read_text())
    jobs = {x['variant']: x for x in launch['jobs']}
    if set(jobs) != {'coordinate', 'visual'}:
        raise ValueError('The run must contain both coordinate-only and visual variants.')
    configs = {}
    for variant, job in jobs.items():
        run = Path(job['run_dir'])
        done = json.loads((run / 'completion.json').read_text())
        if done.get('real_data_smoke_only') or done.get('status') != 'completed' or done.get('steps') != 500000:
            raise RuntimeError('Both full 500000-step runs must complete before manuscript evaluation.')
        configs[variant] = json.loads((run / 'run_config.json').read_text())
    for field in ['train_manifest_sha256', 'val_manifest_sha256', 'steps', 'batch_size',
                  'keypoints', 'correspondences', 'lr', 'seed', 'schedule', 'config']:
        if configs['coordinate'][field] != configs['visual'][field]:
            raise ValueError('Paired-run configuration differs: ' + field)
    output = group / ('evaluation_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    script = Path(__file__).with_name('evaluate_visual.py')
    results = {}
    for variant in ['coordinate', 'visual']:
        checkpoint = Path(jobs[variant]['run_dir']) / 'model_best.pth'
        selection = output / (variant + '_validation')
        common = [sys.executable, '-u', str(script), '--project', str(project),
                  '--device', device, '--variant', variant, '--checkpoint', str(checkpoint)]
        subprocess.run(common + ['--manifest', configs[variant]['val_manifest'],
                                 '--threshold-grid', grid, '--output-dir', str(selection)], check=True)
        test = output / (variant + '_test')
        subprocess.run(common + ['--threshold-file', str(selection / 'threshold_selection.json'),
                                 '--output-dir', str(test)], check=True)
        results[variant] = json.loads((test / 'summary.json').read_text())
        if results[variant]['input_failures']:
            raise RuntimeError('Fix input failures before reporting the test results: ' + variant)
    if results['coordinate']['manifest_sha256'] != results['visual']['manifest_sha256']:
        raise RuntimeError('The test image-pair sets differ between variants.')
    results['visual_gain_percentage_points'] = {
        metric: results['visual']['auc_percent'][metric] - results['coordinate']['auc_percent'][metric]
        for metric in ['AUC@5', 'AUC@10', 'AUC@20']}
    (output / 'paired_results.json').write_text(json.dumps(results, indent=2))
    print('Paired test results:', output / 'paired_results.json', flush=True)
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--group')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--threshold-grid', default=DEFAULT_GRID)
    args = ap.parse_args()
    finish(args.project, args.group, args.device, args.threshold_grid)
