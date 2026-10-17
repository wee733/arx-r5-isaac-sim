#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
#
# Full-parameter GR00T N1.7 fine-tune on an ARX door data v2 bundle.
# Run from the unpacked bundle on the training server. Every tunable module
# (LLM, vision encoder, projector, diffusion action head) is trained.
# Refuses to start unless the bundle's own checks hold; --print-command only
# prints the trainer command.
set -euo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The policy server loads checkpoints with this commit; train with the same one.
pinned_commit="${N17_PINNED_COMMIT:-d2b7e75b937e3ec9aa5dbc798f08b89692c49734}"
repo="${GR00T_REPO:?set GR00T_REPO to the Isaac-GR00T checkout at ${pinned_commit}}"
python_bin="${GR00T_TRAIN_PYTHON:-${repo}/.venv/bin/python}"
base_model="${N17_BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
output="${N17_OUTPUT_DIR:?set N17_OUTPUT_DIR to a new checkpoint directory}"
num_gpus="${N17_NUM_GPUS:-4}"
batch_size="${N17_GLOBAL_BATCH_SIZE:-64}"
accumulation="${N17_GRADIENT_ACCUMULATION_STEPS:-1}"
max_steps="${N17_MAX_STEPS:-15000}"
save_steps="${N17_SAVE_STEPS:-2500}"
save_limit="${N17_SAVE_TOTAL_LIMIT:-6}"
workers="${N17_DATALOADER_WORKERS:-4}"
# Not an official value: NVIDIA publishes no full fine-tune rate for N1.7.
# Half the official 1e-4 default, because the 3B backbone is now trained too.
learning_rate="${N17_LEARNING_RATE:-5e-5}"
min_vram_mib="${N17_MIN_VRAM_MIB:-79000}"
resume=false
print_only=false
while (($#)); do
  case "$1" in
    --resume) resume=true; shift ;;
    --print-command) print_only=true; shift ;;
    -h|--help) sed -n 4,9p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
dataset="${bundle_root}/datasets/train"
config="${bundle_root}/configs/door_v2_n17.py"

command=("$python_bin")
if ((num_gpus > 1)); then
  command+=(-m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=${num_gpus}")
fi
command+=("${repo}/gr00t/experiment/launch_finetune.py"
  --base-model-path "$base_model" --dataset-path "$dataset"
  --embodiment-tag NEW_EMBODIMENT --modality-config-path "$config"
  --num-gpus "$num_gpus" --global-batch-size "$batch_size"
  --gradient-accumulation-steps "$accumulation"
  --max-steps "$max_steps" --save-steps "$save_steps" --save-total-limit "$save_limit"
  --dataloader-num-workers "$workers" --learning-rate "$learning_rate"
  --tune-llm --tune-visual --tune-projector --tune-diffusion-model
  --output-dir "$output")
"$resume" && command+=(--resume-from-checkpoint)
if "$print_only"; then
  printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi

die() { printf 'Training refused: %s\n' "$*" >&2; exit 2; }
[[ -x "$python_bin" ]] || die "GPU Python is unavailable: $python_bin"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$pinned_commit" ]] || die "Isaac-GR00T must be at $pinned_commit"
[[ -z "$(git -C "$repo" status --porcelain -- gr00t pyproject.toml uv.lock)" ]] || die 'official trainer source has local changes'
((batch_size % num_gpus == 0)) || die 'global batch must be divisible by the GPU count'
if ! "$resume" && [[ -e "$output" ]]; then die 'output exists; choose a new directory or pass --resume'; fi
"$resume" && [[ ! -d "$output" ]] && die '--resume needs the existing output directory'

unset PYTHONPATH PYTHONHOME
"$python_bin" - "$bundle_root" "$num_gpus" "$min_vram_mib" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
root, gpus, min_vram = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
manifest = json.loads((root / 'manifest.json').read_text())
for name, digest in manifest['sha256'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f'bundle file changed after packaging: {name}')
train = root / 'datasets/train'
info = json.loads((train / 'meta/info.json').read_text())
if info['total_episodes'] != manifest['train']['episodes']:
    raise SystemExit('train episode count differs from the manifest')
report = json.loads((train / 'meta/n17_validation.json').read_text())
if report.get('status') != 'passed':
    raise SystemExit('train dataset has no passing official validation')
state = json.loads((train / 'meta/stats.json').read_text())['observation.state']
if (state['q01'][6], state['q99'][6]) != (0.0, 0.028):
    raise SystemExit('gripper state must be normalized over 0..0.028 m; rerun the bundle builder')
if not torch.cuda.is_available() or torch.cuda.device_count() < gpus:
    raise SystemExit(f'{gpus} CUDA GPUs required')
small = [i for i in range(gpus) if torch.cuda.get_device_properties(i).total_memory / 2**20 < min_vram]
if small:
    raise SystemExit(f'full fine-tuning needs >= {min_vram} MiB per GPU (official: 80 GB+); '
                     f'GPUs {small} are smaller. Set N17_MIN_VRAM_MIB only if you accept OOM risk.')
print(f'Bundle accepted: {info["total_episodes"]} train episodes, {gpus} GPUs, full fine-tune')
PY

export PYTHONPATH="$repo"
cd "$repo"
printf 'Starting full N1.7 fine-tune: lr=%s batch=%s steps=%s\n' "$learning_rate" "$batch_size" "$max_steps"
exec "${command[@]}"
