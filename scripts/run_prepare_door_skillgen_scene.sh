#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${ISAAC_SIM_PYTHON:-/home/lbz/miniforge3/envs/isaaclab/bin/python}"
isaacsim_root="$("${python_bin}" -c 'import importlib.util, pathlib; spec=importlib.util.find_spec("isaacsim"); print(pathlib.Path(next(iter(spec.submodule_search_locations))))')"
usd_libs=''
for candidate in "${isaacsim_root}/extscache"/omni.usd.libs-*; do
  if [[ -d "${candidate}/pxr" ]]; then
    usd_libs="${candidate}"
    break
  fi
done
if [[ -z "${usd_libs}" ]]; then
  echo "Could not find omni.usd.libs under ${isaacsim_root}" >&2
  exit 2
fi
python_libdir="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
export PYTHONPATH="${repo_root}/arx_r5_isaac_sim_bringup:${usd_libs}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${usd_libs}/bin${python_libdir:+:${python_libdir}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${python_bin}" "${repo_root}/scripts/prepare_door_skillgen_scene.py" "$@"
