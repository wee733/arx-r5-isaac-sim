#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

# Restore the historical stack independently of the machine's current SDK.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
command -v uv >/dev/null || { echo 'Install uv before restoring Isaac Sim 5.1.' >&2; exit 2; }
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
export UV_HTTP_TIMEOUT=300
runtime_dir="${repo_root}/generated/runtime/isaac51"
lab_root="${repo_root}/generated/runtime/isaac51-source"
revision='f4aa17f87e2e5db5484f0b5974918573e8918ce2'
if [[ ! -d "${lab_root}" ]]; then
  git init "${lab_root}"
  git -C "${lab_root}" remote add origin https://github.com/isaac-sim/IsaacLab.git
  git -C "${lab_root}" fetch --depth=1 origin "${revision}"
  git -C "${lab_root}" checkout --detach "${revision}"
fi
if [[ "$(git -C "${lab_root}" rev-parse HEAD)" != "${revision}" ]]; then
  echo 'The existing legacy Lab checkout does not match the pinned revision.' >&2
  exit 2
fi
if [[ ! -x "${runtime_dir}/.venv/bin/python" ]]; then
  uv python install 3.11
  uv venv --python 3.11 --seed "${runtime_dir}/.venv"
fi
python_bin="${runtime_dir}/.venv/bin/python"
if [[ "$("${python_bin}" -c 'import sys;print("%d.%d" % sys.version_info[:2])')" != '3.11' ]]; then
  echo 'The existing legacy runtime must use Python 3.11.' >&2
  exit 2
fi
uv pip install --python "${python_bin}" 'isaacsim[all,extscache]==5.1.0.0' \
  'numpy==1.26.0' --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install --python "${python_bin}" --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128'
uv pip install --python "${python_bin}" \
  --constraint "${repo_root}/configs/isaac51-door-constraints.txt" \
  --requirement "${repo_root}/configs/isaac51-door-requirements.txt"
python3 "${repo_root}/scripts/door/setup/setup_door_runtime.py" --profile isaac51 \
  --isaac-lab-root "${lab_root}" --source-python "${python_bin}" --keep-default
uv pip check --python "${python_bin}"
uv pip freeze --python "${python_bin}" > "${runtime_dir}/packages.txt"
