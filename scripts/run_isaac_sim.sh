#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_source="${repo_root}/arx_r5_isaac_sim_bringup"
python_bin="${ISAAC_SIM_PYTHON:-python}"
use_ros=true

for argument in "$@"; do
  if [[ "${argument}" == '--no-ros' ]]; then
    use_ros=false
  fi
done

if [[ "${PYTHONPATH:-}" == *"python3.12"* ]]; then
  echo "Isaac Sim 5.1 uses Python 3.11, but PYTHONPATH contains Python 3.12." >&2
  echo "Open a clean terminal; do not source /opt/ros/jazzy before this script." >&2
  echo "Your shell may be sourcing ROS from ~/.bashrc; source ROS only in the ROS terminal." >&2
  echo "One-shot workaround: env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH ..." >&2
  exit 2
fi

if [[ "${LD_LIBRARY_PATH:-}" == *"/opt/ros/"* ]]; then
  echo "LD_LIBRARY_PATH contains a ROS installation, which can conflict with Isaac Sim." >&2
  echo "Open a clean Isaac Sim terminal and unset LD_LIBRARY_PATH before launching." >&2
  exit 2
fi

export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-25}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"

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

if [[ "${use_ros}" == true ]]; then
  bridge_lib="${isaacsim_root}/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib"
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
