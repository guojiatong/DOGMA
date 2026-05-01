# DOGMA IMU-Only Dog Motion Generation

This repository contains research code and example artifacts for generating short-horizon dog motion from dog-worn inertial measurement units (IMUs). The core task is:

> given 6 s of raw dog IMU history at 20 Hz, generate a plausible 2 s future motion clip in a derived pseudo-pose space.

The pipeline is built around DOGMA-style wearable sensing and includes IMU preprocessing, pseudo-pose construction, temporal VAE training, latent diffusion training, evaluation, and qualitative visualization.

## Important Scope Note

Pseudo-pose in this repository is a derived representation, not motion-capture ground truth. It is computed from dog-worn IMU signals and is useful for modeling, visualization, and relative evaluation, but it should not be interpreted as measured 3D dog pose.

## Repository Layout

```text
DOGMA_GIT/
├── figs/                         # Paper/report figures and metric plots
├── videos/                       # Qualitative example videos and payloads
├── src/
│   ├── build_imu_only_dataset.py # Build 20 Hz raw IMU feature files and manifests
│   ├── export_pseudo_pose_20hz.py # Export derived pseudo-pose NPZ files
│   ├── build_vae_manifest.py     # Build train/val/test VAE window manifests
│   ├── train_temporal_vae.py     # Train pseudo-pose temporal VAE
│   ├── train_raw_imu_temporal_vae.py
│   ├── train_latent_diffusion.py # Train pseudo-pose latent diffusion
│   ├── train_raw_imu_latent_diffusion.py
│   ├── evaluate_temporal_vae.py
│   ├── evaluate_latent_diffusion.py
│   ├── evaluate_latent_diffusion_plausibility.py
│   ├── visualize_pose.py
│   ├── visualize_dog_skeleton.py
│   ├── models/
│   │   ├── pose_generative.py
│   │   └── imu_baselines.py
│   └── motion_encoder/           # Frozen motion encoder used for distributional metrics
└── requirement.txt
```

## Installation

Create a clean Python environment, then install the dependencies.

```bash
cd /path/to/DOGMA_GIT
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

For GPU training, install the PyTorch build that matches your CUDA version if the default `pip install torch` wheel is not appropriate for your machine.

Video export uses Matplotlib animation writers. Install `ffmpeg` separately if you want MP4 output:

```bash
# macOS example
brew install ffmpeg
```

## Expected Data Layout

The raw DOGMA data is not bundled in this repository. Most scripts expect an external data root with filled IMU segments and generated 20 Hz artifacts.

Recommended layout:

```text
Data/
├── IMU_New2_Filled/              # Filled/synchronized per-segment IMU files
└── IMU_Only_20Hz_v1/
    ├── features_20hz/            # Built by build_imu_only_dataset.py
    ├── pseudo_pose_20hz/         # Built by export_pseudo_pose_20hz.py
    └── manifests/
        ├── segment_manifest.csv
        ├── window_index_train.csv
        ├── window_index_val.csv
        ├── window_index_test.csv
        ├── vae_segment_manifest.csv
        ├── vae_window_index_train.csv
        ├── vae_window_index_val.csv
        └── vae_window_index_test.csv
```

Use environment variables in the examples below to keep commands portable:

```bash
export FILLED_ROOT=/path/to/Data/IMU_New2_Filled
export DATA_ROOT=/path/to/Data/IMU_Only_20Hz_v1
export OUTPUT_ROOT=/path/to/outputs
```

## Pipeline

### 1. Build 20 Hz Raw IMU Features

This step downsamples filled IMU segments and writes `features_20hz/` plus raw IMU manifests.

```bash
python src/build_imu_only_dataset.py \
  --input-root "$FILLED_ROOT" \
  --output-root "$DATA_ROOT" \
  --window-frames 160 \
  --train-stride-frames 20 \
  --eval-stride-frames 120
```

### 2. Export Pseudo-Pose

This step derives pseudo-pose from the same IMU segments and writes `pseudo_pose_20hz/`.

```bash
python src/export_pseudo_pose_20hz.py \
  --filled-root "$FILLED_ROOT" \
  --feature-root "$DATA_ROOT/features_20hz" \
  --output-root "$DATA_ROOT/pseudo_pose_20hz" \
  --audit-output "$DATA_ROOT/visual_audit/pseudo_pose_export_audit.csv"
```

Each pseudo-pose NPZ stores:

- `relative_positions`: stern-centered relative joint positions
- `root_heading_6d`: 6D root-heading representation
- `joint_names`
- `packet_counter`
- `is_interpolated`

### 3. Build VAE Manifests

The canonical future-motion setup uses 8 s windows at 20 Hz: 6 s history plus 2 s future.

```bash
python src/build_vae_manifest.py \
  --data-root "$DATA_ROOT" \
  --window-frames 160 \
  --train-stride-frames 20 \
  --eval-stride-frames 120
```

### 4. Train a Pseudo-Pose Temporal VAE

```bash
python src/train_temporal_vae.py \
  --train-window-index-csv "$DATA_ROOT/manifests/vae_window_index_train.csv" \
  --val-window-index-csv "$DATA_ROOT/manifests/vae_window_index_val.csv" \
  --output-dir "$OUTPUT_ROOT/temporal_vae" \
  --window-frames 160 \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --latent-dim 64 \
  --hidden-dim 128 \
  --beta 1e-3 \
  --scheduler-type cosine \
  --scheduler-t-max 100 \
  --device auto
```

Evaluate a trained VAE:

```bash
python src/evaluate_temporal_vae.py \
  --checkpoint "$OUTPUT_ROOT/temporal_vae/best.pt" \
  --window-index-csv "$DATA_ROOT/manifests/vae_window_index_test.csv" \
  --output-dir "$OUTPUT_ROOT/eval_temporal_vae" \
  --device auto
```

### 5. Train Conditional Latent Diffusion

This model conditions on past raw IMU and denoises pseudo-pose VAE latents.

```bash
python src/train_latent_diffusion.py \
  --train-window-index-csv "$DATA_ROOT/manifests/vae_window_index_train.csv" \
  --val-window-index-csv "$DATA_ROOT/manifests/vae_window_index_val.csv" \
  --vae-checkpoint "$OUTPUT_ROOT/temporal_vae/best.pt" \
  --output-dir "$OUTPUT_ROOT/latent_diffusion" \
  --epochs 100 \
  --batch-size 16 \
  --past-frames 120 \
  --future-window-frames 40 \
  --condition-hidden-dim 128 \
  --denoiser-num-blocks 4 \
  --diffusion-steps 50 \
  --device auto
```

Evaluate latent diffusion:

```bash
python src/evaluate_latent_diffusion.py \
  --checkpoint "$OUTPUT_ROOT/latent_diffusion/best.pt" \
  --window-index-csv "$DATA_ROOT/manifests/vae_window_index_test.csv" \
  --output-dir "$OUTPUT_ROOT/eval_latent_diffusion" \
  --sample-count 10 \
  --device auto
```

### 6. Raw-IMU Ablations

Raw-IMU VAE and raw-IMU latent diffusion baselines are provided for representation ablations.

```bash
python src/train_raw_imu_temporal_vae.py \
  --train-window-index-csv "$DATA_ROOT/manifests/window_index_train.csv" \
  --val-window-index-csv "$DATA_ROOT/manifests/window_index_val.csv" \
  --output-dir "$OUTPUT_ROOT/raw_imu_temporal_vae" \
  --window-frames 160 \
  --epochs 100 \
  --batch-size 16 \
  --device auto
```

```bash
python src/train_raw_imu_latent_diffusion.py \
  --train-window-index-csv "$DATA_ROOT/manifests/vae_window_index_train.csv" \
  --val-window-index-csv "$DATA_ROOT/manifests/vae_window_index_val.csv" \
  --raw-imu-vae-checkpoint "$OUTPUT_ROOT/raw_imu_temporal_vae/best.pt" \
  --output-dir "$OUTPUT_ROOT/raw_imu_latent_diffusion" \
  --epochs 100 \
  --batch-size 16 \
  --past-frames 120 \
  --future-window-frames 40 \
  --device auto
```

## Evaluation Metrics

The code reports paired prediction metrics and distributional/plausibility diagnostics:

- MPJPE over pseudo-pose relative joints
- root-heading error in degrees
- latent-diffusion reconstruction metrics
- Motion-FID and Diversity@10 through the motion encoder
- jerk and range-of-motion plausibility violations

Because pseudo-pose is derived from IMU rather than motion capture, metrics should be read as pseudo-pose-space evaluation, not motion-capture ground-truth evaluation.

## Figures and Videos

The `figs/` directory contains pipeline diagrams, metric plots, and training curves used for reporting. The `videos/` directory contains qualitative examples, including VAE reconstructions and generated 2 s future-motion clips.

Example files:

- `figs/Generative_Pipeline.png`
- `figs/Evaluation_Metric.png`
- `videos/VAE_Reconstruction/`
- `videos/six_model_diffusion_generated_only_cynthia_segment5_start4320_20260423/`

## Notes for New Machines

Some manifest CSVs may contain absolute paths from the machine where they were generated. The dataloader utilities support rebuilding paths from a new `data_root` when records contain participant and segment identifiers. If path resolution fails, regenerate manifests with `src/build_vae_manifest.py` on the target machine.

Generated outputs such as checkpoints, metric CSVs, and sample videos can be large. Keep them outside the repository or under an ignored `outputs/` directory when running new experiments.
