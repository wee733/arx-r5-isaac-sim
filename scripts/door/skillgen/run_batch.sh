#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${ISAAC_SIM_PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ ! -f "${repo_root}/generated/runtime/local.json" ]]; then
    echo 'Create the project runtime with scripts/door/setup/setup_door_runtime.py first.' >&2
    exit 2
  fi
  python_bin="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["simulation_python"])' \
    "${repo_root}/generated/runtime/local.json")"
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Simulation Python is unavailable: ${python_bin}" >&2
  exit 2
fi
# Data generation uses an isolated native planner, not the ROS overlay.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
exec "${python_bin}" "${repo_root}/scripts/door/skillgen/run_door_skillgen_batch.py" "$@"
