"""Check actual imports; optionally install only missing small dependencies."""
import importlib
import json
import subprocess
import sys

PACKAGES = {'numpy': 'numpy==1.26.4', 'cv2': 'opencv-python-headless==4.10.0.84',
            'h5py': 'h5py==3.11.0', 'einops': 'einops==0.6.1'}


def check(install_missing=False):
    versions = {'python': sys.version, 'python_executable': sys.executable}
    for module, package in PACKAGES.items():
        try:
            value = importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != module or not install_missing:
                raise RuntimeError('Cannot import ' + module + '. Run with --install-missing for missing small dependencies.') from exc
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps', package], check=True)
            importlib.invalidate_caches()
            value = importlib.import_module(module)
        versions[module] = getattr(value, '__version__', 'unknown')
    import torch
    import cv2
    if not hasattr(cv2, 'SIFT_create'):
        raise RuntimeError('The active cv2 does not expose SIFT_create; use an OpenCV build with SIFT support.')
    cv2.SIFT_create(nfeatures=32)
    if int(versions['numpy'].split('.')[0]) >= 2 and torch.__version__.startswith('2.3.'):
        raise RuntimeError('This torch 2.3 environment requires a compatible NumPy 1.x build; preserve the original NumPy 1.26.4 environment.')
    versions.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                    cuda_devices=[{'index': i, 'name': torch.cuda.get_device_name(i),
                                   'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                                  for i in range(torch.cuda.device_count())])
    print(json.dumps(versions, indent=2), flush=True)
    return versions


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--install-missing', action='store_true')
    check(ap.parse_args().install_missing)
