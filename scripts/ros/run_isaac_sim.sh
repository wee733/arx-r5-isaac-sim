#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
package_source="${repo_root}/arx_r5_isaac_sim_bringup"
python_bin="${ISAAC_SIM_PYTHON:-python}"
use_ros=true

for argument in "$@"; do
  if [[ "${argument}" == '--no-ros' ]]; then
    use_ros=false
  fi
done

if [[ "${LD_LIBRARY_PATH:-}" == *"/opt/ros/"* ]]; then
  echo "LD_LIBRARY_PATH contains a ROS installation, which can conflict with Isaac Sim." >&2
  echo "Open a clean Isaac Sim terminal and unset LD_LIBRARY_PATH before launching." >&2
  exit 2
fi

export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-25}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"

if ! sim_installation="$(
  "${python_bin}" - <<'PY'
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.find_spec('isaacsim')
if spec is None or not spec.submodule_search_locations:
    sys.exit('the selected Python does not contain the isaacsim package')
root = Path(next(iter(spec.submodule_search_locations))).resolve()
try:
    version = importlib.metadata.version('isaacsim')
except importlib.metadata.PackageNotFoundError:
    version_file = root / 'VERSION'
    if not version_file.is_file():
        sys.exit('the selected Isaac Sim installation has no version metadata')
    version = version_file.read_text(encoding='utf-8').strip().split('-')[0]
print(f'{root}\t{version}\t{sys.version_info.major}.{sys.version_info.minor}')
PY
)"; then
  echo "Unable to locate Isaac Sim with ${python_bin}." >&2
  echo "Set ISAAC_SIM_PYTHON to the selected Isaac Sim Python executable." >&2
  exit 2
fi

IFS=$'\t' read -r isaacsim_root isaacsim_version python_version <<< "${sim_installation}"
case "${isaacsim_version}" in
  5.1|5.1.*) expected_python='3.11' ;;
  6.1|6.1.*) expected_python='3.12' ;;
  *)
    echo "Unsupported Isaac Sim version ${isaacsim_version}; use the configured 5.1 or 6.1 runtime." >&2
    exit 2
    ;;
esac
if [[ "${python_version}" != "${expected_python}" ]]; then
  echo "Isaac Sim ${isaacsim_version} requires Python ${expected_python}, found ${python_version}." >&2
  exit 2
fi
if [[ "${expected_python}" == '3.11' && "${PYTHONPATH:-}" == *"python3.12"* ]]; then
  echo "Isaac Sim ${isaacsim_version} uses Python 3.11, but PYTHONPATH contains Python 3.12." >&2
  echo "Open a clean terminal; do not source /opt/ros/jazzy before this script." >&2
  echo "Your shell may be sourcing ROS from ~/.bashrc; source ROS only in the ROS terminal." >&2
  echo "One-shot workaround: env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH ..." >&2
  exit 2
fi

if [[ "${use_ros}" == true ]]; then
  if [[ "${expected_python}" == '3.12' ]]; then
    bridge_lib="${isaacsim_root}/exts/isaacsim.ros2.core/${ROS_DISTRO}/lib"
  else
    bridge_lib="${isaacsim_root}/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib"
  fi
  if [[ ! -d "${bridge_lib}" ]]; then
    echo "Isaac Sim ROS 2 bridge libraries were not found: ${bridge_lib}" >&2
    echo "Use an Isaac Sim distribution that provides ROS 2 ${ROS_DISTRO}." >&2
    exit 2
  fi

  case ":${LD_LIBRARY_PATH:-}:" in
    *":${bridge_lib}:"*) ;;
    *) export LD_LIBRARY_PATH="${bridge_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

export PYTHONPATH="${package_source}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m arx_r5_isaac_sim_bringup.simulation "$@"
