#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

# Run a stage-4 dataset tool (export_door_lerobot.py / validate_door_n17.py) in
# the isolated GR00T data venv, with the project's FFmpeg on PATH for ffprobe
# and on LD_LIBRARY_PATH for torchcodec. Usage:
#   scripts/door/dataset/run_dataset_tool.sh export_door_lerobot.py --batch-report ... --split candidate --output ...
#   scripts/door/dataset/run_dataset_tool.sh validate_door_n17.py --dataset ... --groot-root generated/runtime/groot_n17_source
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${GROOT_DATA_PYTHON:-${repo_root}/generated/runtime/groot_data/.venv/bin/python}"
tools="${repo_root}/generated/runtime/tools/usr"
if [[ ! -x "${python_bin}" ]]; then
  echo "GR00T data Python not found: ${python_bin}" >&2
  echo "Create it with: python3 ${repo_root}/scripts/door/setup/setup_groot_data_runtime.py" >&2
  exit 2
fi
tool="${1:?usage: run_dataset_tool.sh <export_door_lerobot.py|validate_door_n17.py> [args...]}"
shift
case "${tool}" in
  export_door_lerobot.py|export_autodata_lerobot.py|export_b2arx_lerobot.py|validate_b2arx_eef_n17.py|audit_autodata_pilot.py|validate_door_n17.py|audit_stage_prompter.py|build_v2_bundle.py) ;;
  *) echo "unknown dataset tool: ${tool}" >&2; exit 2 ;;
esac
# Never mix ROS or Isaac Sim Python packages into the data validator.
unset PYTHONPATH PYTHONHOME
if [[ -d "${tools}/bin" ]]; then
  export PATH="${tools}/bin:${PATH}"
  export LD_LIBRARY_PATH="${tools}/lib/x86_64-linux-gnu"
else
  unset LD_LIBRARY_PATH
fi
exec "${python_bin}" "${repo_root}/scripts/door/dataset/${tool}" "$@"
