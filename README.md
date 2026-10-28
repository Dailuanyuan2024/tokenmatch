# TokenMatch

An anonymized code release for **TokenMatch: Leveraging Learnable Cluster Tokens for Two-View Correspondence Pruning**.

This repository contains the coordinate-only TokenMatch model and its pretrained checkpoint, together with the implementation of the visual-enhanced variant, **TokenMatch\***.

## Repository structure

```text
TokenMatch/
  core/                         Model, training, and evaluation code
    tokenmatch.py               Coordinate-only TokenMatch backbone
    log/train/model_best_tokenmatch.pth
  test/                         Testing utilities
TokenMatch_visual_integration/
  tokenmatch_visual_additive_updated.py   TokenMatch* model
  visual_inputs.py              Visual feature construction
  pair_dataset.py               Data loading
  prepare_data.py               Dataset manifest preparation
  train_visual.py               Coordinate-only / visual training
  evaluate_visual.py            Evaluation and threshold selection
  pose_evaluation.py            Relative-pose estimation and AUC
  project_runtime.py            Model loading and shared configuration
  official_train_scenes.txt     Training scene list
```

## TokenMatch* visual input

Each correspondence has a 262-dimensional visual feature: two 128-dimensional SIFT descriptors, two keypoint scales, and the sine/cosine encoding of two keypoint orientations. A shared MLP maps these features to 128 dimensions; the result is added to the coordinate embedding before learnable cluster-token generation. The underlying pruning backbone remains unchanged.

## Requirements and data

The code uses Python, PyTorch, NumPy, OpenCV, h5py, and einops. It was smoke-tested with Python 3.10 and PyTorch 2.3.1 (CUDA 12.1). Obtain the YFCC100M data separately; images and dataset archives are not included in this release. Use non-test scenes for training and validation.

## Usage

Run these commands from the repository root. Supply paths to your own dataset and generated manifests.

```bash
# Prepare training/validation manifests
python TokenMatch_visual_integration/prepare_data.py \
  --data-dir /path/to/YFCC100M

# Train TokenMatch* from scratch
python TokenMatch_visual_integration/train_visual.py \
  --project . \
  --train-manifest /path/to/train.jsonl \
  --val-manifest /path/to/val.jsonl \
  --run-dir ./outputs/visual \
  --variant visual \
  --device cuda:0

# See evaluation arguments (including validation-selected thresholds)
python TokenMatch_visual_integration/evaluate_visual.py --help
```

Use `--variant coordinate` in the training script for the coordinate-only comparison. Consult the command-line help for further training, data, and evaluation options.

## Pretrained weights

- **TokenMatch (coordinate-only):** `TokenMatch/core/log/train/model_best_tokenmatch.pth`.


## Pretrained Checkpoint

The pretrained coordinate-only TokenMatch checkpoint is provided
in split files under `TokenMatch/checkpoint_parts/`.

To reconstruct the checkpoint, run from the repository root:

```bash
cat TokenMatch/checkpoint_parts/model_best_tokenmatch.pth.part-*   > TokenMatch/core/log/train/model_best_tokenmatch.pth

sha256sum -c   TokenMatch/checkpoint_parts/model_best_tokenmatch.sha256
```

The TokenMatch* implementation and training code are provided
without pretrained weights.
