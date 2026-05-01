python eval_fid.py \
  --encoder-checkpoint /projectnb/ml4adr/mlwei/Dogma/motion_encoder/logs_dir/train/encoder_20260419_215045/best.pt \
  --eval-root /projectnb/ml4adr/mlwei/Dogma/DiT/logs_dir/eval_cond \
  --results-dir ./logs_dir/eval_cond \
  --clip-mode future_only \
  --feature-space embedding

# If your encoder was trained on full 5-second clips, switch to:
# python eval_fid.py \
#   --encoder-checkpoint /projectnb/ml4adr/mlwei/Dogma/motion_encoder/logs_dir/encoder_<run>/best.pt \
#   --eval-root /projectnb/ml4adr/mlwei/Dogma/DiT/logs_dir/eval_cond \
#   --results-dir /projectnb/ml4adr/mlwei/Dogma/motion_encoder/fid_results \
#   --clip-mode full_clip \
#   --feature-space embedding
