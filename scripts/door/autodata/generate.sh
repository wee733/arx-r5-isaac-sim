#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
# Run AutoData's official generate_dataset.py on the ARX door task.
# Use inside the container opened by run_container.sh; this script only fills in
# the ARX descriptors and extension entry points, the generation loop is upstream.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
autodata_root="${AUTODATA_ROOT:-/workspaces/autodata}"
# sudo in the official entrypoint strips PYTHONPATH even with --preserve-env.
# Restore the project extension path after crossing that user boundary.
export PYTHONPATH="${project_root}:${autodata_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
assets="" input="" output="" trials=1 num_envs=1 physics=source_cpu seed=13
usage() {
  cat <<'TEXT'
Usage: generate.sh --assets MANIFEST --input SOURCE_HDF5 --output DIR [--trials N] [--num-envs N]
                   [--physics source_cpu|lab_default] [-- extra generate_dataset.py options]
MANIFEST: asset bundle from prepare.py. SOURCE_HDF5: convert_seed.py output.
Writes DIR/generated.hdf5, DIR/generated_failed.hdf5 and DIR/result.json; then run accept.py.
TEXT
}
extra=()
while (($#)); do
  case "$1" in
    --assets) assets="$2"; shift 2 ;;
    --input) input="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --trials) trials="$2"; shift 2 ;;
    --num-envs) num_envs="$2"; shift 2 ;;
    --physics) physics="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "${assets}" && -n "${input}" && -n "${output}" ]] || { usage >&2; exit 2; }
[[ -f "${assets}" && -f "${input}" ]] || { echo "missing --assets or --input file" >&2; exit 2; }
[[ -x /isaac-sim/python.sh && -f "${autodata_root}/scripts/generate_dataset.py" ]] || {
  echo "run inside the AutoData container (scripts/door/autodata/run_container.sh --start)" >&2; exit 2; }
[[ "${physics}" == source_cpu || "${physics}" == lab_default ]] || { echo 'unknown physics profile' >&2; exit 2; }
[[ "${num_envs}" =~ ^[1-9][0-9]*$ && "${trials}" =~ ^[1-9][0-9]*$ ]] || { echo 'num-envs and trials must be positive integers' >&2; exit 2; }
/isaac-sim/python.sh -c 'from arx_door.autodata.runtime import runtime_provenance; runtime_provenance()'
if [[ -e "${output}" ]] && [[ -n "$(ls -A "${output}")" ]]; then
  echo "output must be a fresh directory: ${output}" >&2; exit 2
fi
mkdir -p "${output}"
assets="$(realpath "${assets}")" input="$(realpath "${input}")" output="$(realpath "${output}")"
configs="${project_root}/configs/autodata"
# source_cpu reproduces the seed scene's CPU PhysX settings; lab_default runs on the GPU.
device=cpu; [[ "${physics}" == source_cpu ]] || device=cuda:0

# Replay and evaluation rebuild the same env from this record (scripts/door/policy/replay_check.py).
# Paths are relative to the run directory, so they resolve inside (/outputs) and outside the container.
/isaac-sim/python.sh - "${output}" "${assets}" "${input}" "${physics}" "${num_envs}" "${trials}" "${seed}" "${configs}" <<'PY'
import json
import os
import sys
from pathlib import Path
import yaml
output, assets, source, physics, num_envs, trials, seed, configs = sys.argv[1:]
task = yaml.safe_load(Path(configs, 'tasks/arx_door_knob_hold_open.yaml').read_text())
task['generation_policy']['seed'] = int(seed)
Path(output, 'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
record = dict(schema='arx-autodata-run-v1', assets=os.path.relpath(assets, output),
              input=os.path.relpath(source, output), physics=physics,
              num_envs=int(num_envs), trials=int(trials), seed=int(seed),
              scene_parameters=json.loads(Path(assets).read_text()).get('trial_parameters', {}))
Path(output, 'run.json').write_text(json.dumps(record, indent=2) + '\n')
PY

cd "${autodata_root}"
# Options after "--" go verbatim to arx_door.autodata.register (the env provider).
exec /isaac-sim/python.sh scripts/generate_dataset.py \
  --env_name Arx-Door-KnobHoldOpen-v0 \
  --task_descriptor "${output}/task.yaml" \
  --embodiment "${configs}/embodiments/arx_r5a.yaml" \
  --env_profile "${configs}/env_profiles/arx_door.yaml" \
  --input_file "${input}" \
  --output_file "${output}/generated.hdf5" \
  --result_file "${output}/result.json" \
  --generation_num_trials "${trials}" --num_envs "${num_envs}" \
  --device "${device}" --enable_cameras --headless \
  --external_callback arx_door.autodata.register:register \
  --recorder_cfg arx_door.autodata.register:recorder_cfg \
  "${extra[@]}" \
  -- --assets "${assets}" --physics "${physics}"
