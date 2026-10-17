#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
# User-operated launcher. Default is a reviewable command preview; --start
# builds the unmodified official Dockerfile and opens the isolated container.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_root="${project_root}/generated/door_autodata"
model_root=""
planner_root="${project_root}/generated/runtime/cumotion"
do_start=false
bootstrap_source=false
official_only=false

usage() {
  cat <<'TEXT'
Usage: run_container.sh [--official | --model-dir DIR] [--output DIR]
                        [--planner-runtime DIR] [--bootstrap-source] [--start]
Default: preview commands only. --start: user builds and starts the container.
The AutoData source must contain its pinned Arena and Lab submodules.
--official opens the upstream example environment without ARX model/planner mounts.
--bootstrap-source initializes the declared Git submodule; it never follows main.
The planner runtime must be the separate native cuMotion CPython 3.12 environment.
TEXT
}
while (($#)); do
  case "$1" in
    --output|--model-dir|--planner-runtime)
      (($# >= 2)) || { usage >&2; exit 2; }
      case "$1" in
        --output) output_root="$2" ;;
        --model-dir) model_root="$2" ;;
        --planner-runtime) planner_root="$2" ;;
      esac
      shift 2 ;;
    --start) do_start=true; shift ;;
    --bootstrap-source) bootstrap_source=true; shift ;;
    --official) official_only=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

# The Python adapters and this launcher read the same reviewed dependency lock.
dependency_values="$(python3 - "${project_root}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from arx_door.autodata import DEPENDENCY, source_checkouts
for _, _, commit in source_checkouts():
    print(commit)
for key in ('base_image', 'image', 'container'):
    print(DEPENDENCY['runtime'][key])
print(DEPENDENCY['source']['path'])
PY
)"
mapfile -t values <<<"${dependency_values}"
pins=("${values[0]}" "${values[1]}" "${values[2]}")
base_image="${values[3]}"
image_name="${values[4]}"
container_name="${values[5]}"
source_root="${project_root}/${values[6]}"
paths=("${source_root}" "${source_root}/submodules/IsaacLab-Arena" "${source_root}/submodules/IsaacLab-Arena/submodules/IsaacLab")
if [[ "${bootstrap_source}" == true ]]; then
  recorded="$(git -C "${project_root}" ls-files --stage -- "${values[6]}" | awk '$1 == 160000 {print $2}')"
  [[ "${recorded}" == "${pins[0]}" ]] || { echo 'Git submodule and dependency lock disagree.' >&2; exit 2; }
  for i in 0 1 2; do
    if [[ -e "${paths[$i]}/.git" ]]; then
      [[ "$(git -C "${paths[$i]}" rev-parse HEAD)" == "${pins[$i]}" ]] || {
        echo "Refusing to reset existing checkout: ${paths[$i]}" >&2; exit 2;
      }
      git -C "${paths[$i]}" diff --quiet HEAD -- || { echo "Modified vendor source: ${paths[$i]}" >&2; exit 2; }
    fi
  done
  GIT_LFS_SKIP_SMUDGE=1 git -C "${project_root}" \
    -c url.https://github.com/.insteadOf=git@github.com: \
    submodule update --init --recursive --depth 1 -- "${values[6]}"
fi
for i in 0 1 2; do
  [[ -e "${paths[$i]}/.git" ]] || {
    echo "Missing pinned checkout: ${paths[$i]}" >&2
    echo 'Prepare the official checkout and its submodules manually before startup.' >&2
    exit 2
  }
  actual="$(git -C "${paths[$i]}" rev-parse HEAD)"
  [[ "${actual}" == "${pins[$i]}" ]] || { echo "Wrong source pin: ${paths[$i]} (${actual})" >&2; exit 2; }
  git -C "${paths[$i]}" diff --quiet HEAD -- || { echo "Modified vendor source: ${paths[$i]}" >&2; exit 2; }
done
python3 "${project_root}/scripts/door/autodata/preflight.py" --source-only
extra_mounts=()
extra_env=()
container_workdir=/workspaces/autodata
if [[ "${official_only}" != true ]]; then
  [[ -n "${model_root}" && -f "${model_root}/urdf/r5a_cumotion.urdf" && -f "${model_root}/xrdf/r5a.xrdf" ]] || {
    echo '--model-dir must contain urdf/r5a_cumotion.urdf and xrdf/r5a.xrdf' >&2; exit 2;
  }
  [[ -x "${planner_root}/.venv/bin/python" ]] || { echo "Missing isolated planner Python: ${planner_root}" >&2; exit 2; }
  source_root="$(realpath "${source_root}")"
  model_root="$(realpath "${model_root}")"
  planner_root="$(realpath "${planner_root}")"
  planner_interpreter="$(readlink -f "${planner_root}/.venv/bin/python")"
  interpreter_root="$(dirname "$(dirname "${planner_interpreter}")")"
  extra_mounts=(
    --mount "type=bind,src=${model_root},dst=/assets/arx_model,readonly"
    --mount "type=bind,src=${model_root},dst=${model_root},readonly"
    --mount "type=bind,src=${planner_root},dst=${planner_root},readonly"
    --mount "type=bind,src=${interpreter_root},dst=${interpreter_root},readonly")
  extra_env=(--env ARX_DESCRIPTION_DIR=/assets/arx_model
    --env "CUMOTION_PYTHON=${planner_root}/.venv/bin/python"
    --env "CUMOTION_ENV_DIR=${planner_root}")
  container_workdir=/workspaces/arx
fi
output_root="$(realpath -m "${output_root}")"
gpu_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')"
[[ "${gpu_arch}" =~ ^[0-9]+\.[0-9]+$ ]] || { echo 'Cannot determine CUDA architecture' >&2; exit 2; }

build=(docker build --pull --progress=plain --build-arg WORKDIR=/workspaces/autodata
  --build-arg "BASE_IMAGE=${base_image}" --build-arg INSTALL_CUROBO=true
  --build-arg "TORCH_CUDA_ARCH_LIST=${gpu_arch}+PTX" --build-arg MAX_JOBS=2
  --label "arx.autodata_commit=${pins[0]}" --label "arx.arena_commit=${pins[1]}"
  --label "arx.lab_commit=${pins[2]}"
  -t "${image_name}" -f "${source_root}/docker/Dockerfile.autodata" "${source_root}")
run=(docker run --rm -it --name "${container_name}" --gpus all --runtime nvidia
  --ipc host --net host --ulimit memlock=-1 --ulimit stack=-1
  # Keep the image's editable runtime writable: official entrypoint owns
  # chown and _isaac_sim symlinks there. The host source remains read-only.
  # The parent mount includes .git/modules, needed by the submodule gitfiles.
  --mount "type=bind,src=${project_root},dst=/workspaces/arx,readonly"
  --mount "type=bind,src=${project_root},dst=${project_root},readonly"
  --mount "type=bind,src=${output_root},dst=/outputs"
  "${extra_mounts[@]}"
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y
  --env ISAACLAB_PATH=/workspaces/autodata/submodules/IsaacLab-Arena/submodules/IsaacLab
  --env PYTHONPATH=/workspaces/arx:/workspaces/autodata
  "${extra_env[@]}"
  --env "DOCKER_RUN_USER_ID=$(id -u)" --env "DOCKER_RUN_USER_NAME=$(id -un)"
  --env "DOCKER_RUN_GROUP_ID=$(id -g)" --env "DOCKER_RUN_GROUP_NAME=$(id -gn)"
  --workdir "${container_workdir}" "${image_name}" bash)
printf '%q ' "${build[@]}"; printf '\n'
printf '%q ' "${run[@]}"; printf '\n'
if [[ "${do_start}" == true ]]; then
  # Refuse startup while swap is exhausted. The preview remains usable.
  swap_total="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
  if [[ "${swap_total}" -gt 0 && "${swap_free}" -eq 0 ]]; then
    echo 'Swap is full. Resolve memory pressure before starting the simulator.' >&2; exit 2;
  fi
  mkdir -p "${output_root}"
  # Rebuild against this exact clean source; Docker's layer cache avoids work
  # when unchanged. A globally reused image tag alone is not version evidence.
  "${build[@]}"
  "${run[@]}"
fi
