#!/usr/bin/env bash
# Full-model GR00T N1.7 training, matching the completed B2ARX run.
set -euo pipefail
: "${GR00T_REPO:?Set the checkout of NVIDIA/Isaac-GR00T}"
: "${GR00T_TRAIN_PYTHON:?Set the Python interpreter installed from its official uv.lock}"
: "${B2ARX_DATASET:?Set the LeRobot v2.1 train directory, never the holdout directory}"
: "${B2ARX_OUTPUT:?Set a new checkpoint output directory}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[[ "$(git -C "$GR00T_REPO" rev-parse HEAD)" == dac54b313160c201c450f97df0fd74ebd379c892 ]]
[[ -z "$(git -C "$GR00T_REPO" status --porcelain -- gr00t)" ]]
[[ ! -e "$B2ARX_OUTPUT" ]]
export B2ARX_EVIDENCE="${B2ARX_EVIDENCE:-${B2ARX_OUTPUT}_evidence}"
export PYTHONPATH="$GR00T_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONHOME
exec "$GR00T_TRAIN_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  "$repo_root/scripts/door/training/launch_b2arx_full.py" \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$B2ARX_DATASET" --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$repo_root/configs/dataset/b2arx_eef_n17.py" \
  --num-gpus 2 --global-batch-size 8 --gradient-accumulation-steps 8 \
  --max-steps 10000 --save-steps 200 --save-total-limit 3 \
  --dataloader-num-workers 2 --learning-rate 2e-5 \
  --tune-llm --tune-visual --tune-projector --tune-diffusion-model \
  --output-dir "$B2ARX_OUTPUT"
