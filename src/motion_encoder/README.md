# Motion Encoder

This directory contains a standalone motion encoder for dog pose clips. It is
intended for representation learning and downstream feature extraction, for
example computing a motion-FID style metric on DiT predictions.

What was kept from the HumanML3D-style code:

- `MotionEncoderBiGRUCo`: the motion-only bidirectional GRU encoder
- `MovementConvEncoder`: adapted here as `MotionConvEncoder`
- classic pairwise `ContrastiveLoss`

What was intentionally not copied:

- text encoders / text decoders
- VAE decoders unrelated to standalone motion encoding
- attention modules tied to text-motion generation
- motion length estimator

Current recommendation:

- train the encoder with `NT-Xent` first
- use the frozen sequence embedding as the feature space for Fréchet distance
- evaluate both GT and generated motion clips with the same frozen encoder

Example training command:

```bash
cd /projectnb/ml4adr/mlwei/Dogma/motion_encoder

python train.py \
  --pose-root /projectnb/ml4adr/mlwei/Dogma/from_jiatong/pseudo_pose_20hz \
  --results-dir /projectnb/ml4adr/mlwei/Dogma/motion_encoder/results \
  --encoder-type gru \
  --loss-type nt_xent
```
