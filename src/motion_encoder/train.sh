# Train an encoder for future-only clips.
# This is the recommended feature space for FID on unconditional DiT future generation.
python train.py \
  --pose-root /projectnb/ml4adr/mlwei/Dogma/from_jiatong/pseudo_pose_20hz \
  --results-dir logs_dir \
  --epochs 500 \
  --lr 2e-4 \
  --batch-size 128 \
  --gpus 0,1 \
  --encoder-type gru \
  --loss-type nt_xent \
  --clip-mode future_only

# Train an encoder on the exact conditional-model target window (4s future after 1s past).
# python train.py \
#   --pose-root /projectnb/ml4adr/mlwei/Dogma/from_jiatong/pseudo_pose_20hz \
#   --results-dir /projectnb/ml4adr/mlwei/Dogma/motion_encoder/results \
#   --encoder-type gru \
#   --loss-type nt_xent \
#   --clip-mode dit_conditional_target

# Train an encoder on full 5s clips.
# python train.py \
#   --pose-root /projectnb/ml4adr/mlwei/Dogma/from_jiatong/pseudo_pose_20hz \
#   --results-dir /projectnb/ml4adr/mlwei/Dogma/motion_encoder/results \
#   --encoder-type gru \
#   --loss-type nt_xent \
#   --clip-mode full_clip
