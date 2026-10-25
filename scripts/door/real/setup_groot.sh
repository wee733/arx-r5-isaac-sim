#!/usr/bin/env bash
# Reuse this Thor's CUDA-enabled PyTorch; keep other dependencies in a local venv.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
runtime="$root/generated/runtime/groot_b2arx"
source_dir="$runtime/source"
if [[ ! -d "$source_dir" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "$source_dir"
  git -C "$source_dir" checkout dac54b313160c201c450f97df0fd74ebd379c892
fi
# Existing checkouts are never reset; no weight hashing or full training install.
python3 -m venv --system-site-packages "$runtime/.venv"
"$runtime/.venv/bin/python" -m pip install \
  'transformers==4.57.3' 'diffusers==0.35.1' 'peft==0.17.1' \
  'albumentations==1.4.18' 'msgpack==1.1.0' 'msgpack-numpy==0.4.8' \
  'pydantic>=2,<3' dm-tree 'omegaconf==2.3.0' 'tyro==0.9.17' \
  'pandas==2.2.3' 'datasets==3.6.0' 'lmdb==1.7.5' 'jsonlines==4.0.0' 'gymnasium==1.2.2' 'wandb==0.23.0' av
printf '%s\n' 'Runtime ready. Run groot.sh verify to load the checkpoint and check GPU inference without robot commands.'
