#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

# Run the door pipeline tests, each group in the interpreter it targets:
#   sim      - selected Isaac Sim/Lab venv (generated/runtime/local.json)
#   dataset  - GR00T data venv (LeRobot export + official N1.7 loader checks)
# ROS package tests live in arx_r5_isaac_sim_bringup/test and need a
# ROS-sourced Python instead; they are not run here.
# Each file runs in its own process: Isaac Lab and GR00T keep process-global
# registries, and one collection error must not hide the remaining results.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
groups="${*:-sim dataset}"
sim_python="${ISAAC_SIM_PYTHON:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["simulation_python"])' \
  "${repo_root}/generated/runtime/local.json" 2>/dev/null)}"
data_python="${GROOT_DATA_PYTHON:-${repo_root}/generated/runtime/groot_data/.venv/bin/python}"
tools="${repo_root}/generated/runtime/tools/usr"
dataset_tests=(test_door_skillgen_lerobot.py test_door_n17_validation.py)
status=0

run_file() {
  local python="$1" file="$2"; shift 2
  if ! (cd "${repo_root}" && env -u PYTHONPATH -u PYTHONHOME "$@" \
        "${python}" -m pytest -q -p no:cacheprovider "tests/${file}"); then
    status=1
  fi
}

for group in ${groups}; do
  case "${group}" in
    sim)
      [[ -x "${sim_python}" ]] || { echo "simulation Python unavailable; run scripts/door/setup/setup_door_runtime.py" >&2; exit 2; }
      for path in "${repo_root}"/tests/test_*.py; do
        file="$(basename "${path}")"
        [[ " ${dataset_tests[*]} " == *" ${file} "* ]] && continue
        # usd_environment() refuses a ROS LD_LIBRARY_PATH inherited from the shell.
        run_file "${sim_python}" "${file}" -u LD_LIBRARY_PATH
      done ;;
    dataset)
      [[ -x "${data_python}" ]] || { echo "GR00T data Python unavailable; run scripts/door/setup/setup_groot_data_runtime.py" >&2; exit 2; }
      for file in "${dataset_tests[@]}"; do
        run_file "${data_python}" "${file}" PATH="${tools}/bin:${PATH}" \
          LD_LIBRARY_PATH="${tools}/lib/x86_64-linux-gnu"
      done ;;
    *) echo "unknown test group: ${group} (expected sim, dataset)" >&2; exit 2 ;;
  esac
done
exit "${status}"
