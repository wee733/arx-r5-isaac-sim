#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
# Launcher. Default prepares the derived Dockerfile and previews commands; --start
# builds the official Dockerfile with the documented project patch and opens the container.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_root="${project_root}/generated/door_autodata"
cache_root="${project_root}/generated/runtime/autodata_cache"
do_start=false
build_only=false
bootstrap_source=false
official_only=false
exec_command=()

usage() {
  cat <<'TEXT'
Usage: run_container.sh [--official] [--output DIR] [--bootstrap-source] [--start | --build-only]
       run_container.sh [--output DIR] --exec COMMAND [ARGS...]
Default: prepare the derived Dockerfile and preview commands; no Docker execution.
--start builds and starts the container.
--build-only builds the pinned image without opening an interactive shell.
--exec runs a command in the existing pinned image (no rebuild, no TTY).
The AutoData source must contain its pinned Arena and Lab submodules.
--official opens the upstream examples (workdir /workspaces/autodata) instead of the ARX project.
--bootstrap-source initializes the declared Git submodule; it never follows main.
ARX generation needs only the project mount: assets/, configs/ and arx_door/ live in it.
TEXT
}
while (($#)); do
  case "$1" in
    --output)
      (($# >= 2)) || { usage >&2; exit 2; }
      output_root="$2"
      shift 2 ;;
    --start) do_start=true; shift ;;
    --build-only) build_only=true; shift ;;
    --bootstrap-source) bootstrap_source=true; shift ;;
    --official) official_only=true; shift ;;
    --exec) shift; (($#)) || { usage >&2; exit 2; }; exec_command=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ "${do_start}" == true && "${build_only}" == true ]] ||
   { ((${#exec_command[@]})) && [[ "${do_start}" == true || "${build_only}" == true ]]; }; then
  echo 'Choose exactly one of --start, --build-only, or --exec.' >&2
  exit 2
fi

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
container_workdir=/workspaces/autodata
if [[ "${official_only}" != true ]]; then
  # The ARX model is a project asset, visible in the container under /workspaces/arx/assets.
  [[ -f "${project_root}/assets/arx_r5a/urdf/r5a_cumotion.urdf" && -f "${project_root}/assets/arx_r5a/xrdf/r5a.xrdf" ]] || {
    echo 'assets/arx_r5a must contain urdf/r5a_cumotion.urdf and xrdf/r5a.xrdf' >&2; exit 2;
  }
  container_workdir=/workspaces/arx
fi
source_root="$(realpath "${source_root}")"
output_root="$(realpath -m "${output_root}")"
dockerfile="${project_root}/generated/runtime/autodata_setup/Dockerfile.autodata"
if ((${#exec_command[@]} == 0)); then
  mkdir -p "$(dirname "${dockerfile}")"
  patch --batch --silent --fuzz=0 --output "${dockerfile}" \
    "${source_root}/docker/Dockerfile.autodata" \
    "${project_root}/dependencies/patches/autodata-early-runtime-constraints.patch"
fi
gpu_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')"
[[ "${gpu_arch}" =~ ^[0-9]+\.[0-9]+$ ]] || { echo 'Cannot determine CUDA architecture' >&2; exit 2; }

build=(docker build --pull --progress=plain --build-arg WORKDIR=/workspaces/autodata
  --build-arg "BASE_IMAGE=${base_image}" --build-arg INSTALL_CUROBO=true
  --build-arg "TORCH_CUDA_ARCH_LIST=${gpu_arch}+PTX" --build-arg MAX_JOBS=2
  --label "arx.autodata_commit=${pins[0]}" --label "arx.arena_commit=${pins[1]}"
  --label "arx.lab_commit=${pins[2]}"
  -t "${image_name}" -f "${dockerfile}" "${source_root}")
tty_args=(-it)
command_args=(bash)
if ((${#exec_command[@]})); then
  tty_args=()
  command_args=("${exec_command[@]}")
  # A persistent interactive container may already own the configured name.
  # Isolate command runs without stopping or repurposing that container.
  container_name="${container_name}-exec-$$"
fi
run=(docker run --rm "${tty_args[@]}" --name "${container_name}" --gpus all --runtime nvidia
  --ipc host --net host --ulimit memlock=-1 --ulimit stack=-1
  # Keep the image's editable runtime writable: official entrypoint owns
  # chown and _isaac_sim symlinks there. The host source remains read-only.
  # The parent mount includes .git/modules, needed by the submodule gitfiles.
  --mount "type=bind,src=${project_root},dst=/workspaces/arx,readonly"
  --mount "type=bind,src=${project_root},dst=${project_root},readonly"
  --mount "type=bind,src=${output_root},dst=/outputs"
  --mount "type=bind,src=${output_root},dst=${output_root}"
  # Upstream normally mounts its entire checkout. Its Dockerfile omits these
  # CLI entrypoints and LFS examples, so expose them explicitly, read-only.
  --mount "type=bind,src=${source_root}/scripts,dst=/workspaces/autodata/scripts,readonly"
  --mount "type=bind,src=${source_root}/datasets,dst=/workspaces/autodata/datasets,readonly"
  --mount "type=bind,src=${source_root}/autodata_tests/test_data,dst=/workspaces/autodata/autodata_tests/test_data,readonly"
  # This also creates /home/<user>, required by the official entrypoint.
  --mount "type=bind,src=${cache_root},dst=/home/$(id -un)/.cache"
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y
  --env ISAACLAB_PATH=/workspaces/autodata/submodules/IsaacLab-Arena/submodules/IsaacLab
  --env PYTHONPATH=/workspaces/arx:/workspaces/autodata
  --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONUNBUFFERED=1
  --env ISAAC_SIM_PYTHON=/isaac-sim/python.sh
  --env "ARX_AUTODATA_COMMIT=${pins[0]}" --env "ARX_ARENA_COMMIT=${pins[1]}"
  --env "ARX_LAB_COMMIT=${pins[2]}"
  --env "DOCKER_RUN_USER_ID=$(id -u)" --env "DOCKER_RUN_USER_NAME=$(id -un)"
  --env "DOCKER_RUN_GROUP_ID=$(id -g)" --env "DOCKER_RUN_GROUP_NAME=$(id -gn)"
  --workdir "${container_workdir}" "${image_name}" "${command_args[@]}")
printf '%q ' "${build[@]}"; printf '\n'
printf '%q ' "${run[@]}"; printf '\n'
if [[ "${do_start}" == true || "${build_only}" == true ]]; then
  mkdir -p "${output_root}"
  # Rebuild against this exact clean source; Docker's layer cache avoids work
  # when unchanged. A globally reused image tag alone is not version evidence.
  "${build[@]}"
fi
if [[ "${do_start}" == true && "${build_only}" != true ]] || ((${#exec_command[@]})); then
  # Validate the image, not just the tag, before claiming build provenance.
  for i in 0 1 2; do
    names=(autodata arena lab)
    actual="$(docker image inspect "${image_name}" --format "{{index .Config.Labels \"arx.${names[$i]}_commit\"}}")"
    [[ "${actual}" == "${pins[$i]}" ]] || { echo 'Image pins disagree; run --build-only.' >&2; exit 2; }
  done
  mkdir -p "${output_root}" "${cache_root}"
  "${run[@]}"
fi
