#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

# Generate the door workcell and a reproducible placement manifest.
#
# The USD Python bindings ship inside the Isaac Sim wheel's extension cache and
# are not importable from a bare interpreter, so this wrapper puts them on
# PYTHONPATH and adds both the extension's own libraries and the interpreter's
# libpython to LD_LIBRARY_PATH before exec'ing the authoring script.
#
# Run from a clean terminal, without sourcing ROS (Isaac Sim uses Python 3.11).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_source="${repo_root}/arx_r5_isaac_sim_bringup"
python_bin="${ISAAC_SIM_PYTHON:-python}"

if [[ "${PYTHONPATH:-}" == *"python3.12"* ]]; then
  echo "Isaac Sim 5.1 uses Python 3.11, but PYTHONPATH contains Python 3.12." >&2
  echo "Open a clean terminal; do not source /opt/ros/jazzy before this script." >&2
  exit 2
fi

if ! isaacsim_root="$(
  "${python_bin}" -c \
    "import importlib.util, pathlib, sys; \
spec = importlib.util.find_spec('isaacsim'); \
sys.exit('the selected Python does not contain the isaacsim package') \
    if spec is None or spec.submodule_search_locations is None else None; \
print(pathlib.Path(next(iter(spec.submodule_search_locations))))"
)"; then
  echo "Unable to locate Isaac Sim with ${python_bin}." >&2
  echo "Set ISAAC_SIM_PYTHON to the Python executable from Isaac Sim 5.1." >&2
  exit 2
fi

usd_libs=''
for candidate in "${isaacsim_root}/extscache"/omni.usd.libs-*; do
  if [[ -d "${candidate}/pxr" ]]; then
    usd_libs="${candidate}"
    break
  fi
done

if [[ -z "${usd_libs}" ]]; then
  echo "Could not find the omni.usd.libs extension under ${isaacsim_root}." >&2
  exit 2
fi

python_libdir="$("${python_bin}" -c \
  'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"

export PYTHONPATH="${package_source}:${usd_libs}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${usd_libs}/bin${python_libdir:+:${python_libdir}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${python_bin}" "${repo_root}/scripts/author_door_scene.py" "$@"
