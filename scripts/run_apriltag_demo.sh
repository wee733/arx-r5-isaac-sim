#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_setup="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
default_isaac_ros_ws="${HOME}/workspace/isaac_ros_source"
isaac_ros_ws="${ISAAC_ROS_WS:-${default_isaac_ros_ws}}"
# ISAAC_ROS_WS is commonly exported by unrelated Isaac ROS projects (for
# example a ZED workspace).  Do not let such a value silently suppress the
# common source tree needed by this demo.  A caller can still select another
# source tree explicitly, provided it has the expected Isaac ROS layout.
if [[ ! -f "${isaac_ros_ws}/install/setup.bash" ||
      ! -d "${isaac_ros_ws}/isaac_ros_common/isaac_ros_test" ]]; then
  if [[ "${isaac_ros_ws}" != "${default_isaac_ros_ws}" &&
        -f "${default_isaac_ros_ws}/install/setup.bash" &&
        -d "${default_isaac_ros_ws}/isaac_ros_common/isaac_ros_test" ]]; then
    echo "Ignoring ISAAC_ROS_WS=${isaac_ros_ws}; it is not the ARX Isaac ROS source workspace." >&2
    echo "Using ${default_isaac_ros_ws}. Set ISAAC_ROS_WS explicitly to a workspace with isaac_ros_common/isaac_ros_test to override." >&2
    isaac_ros_ws="${default_isaac_ros_ws}"
  fi
fi
if [[ ! -f "${isaac_ros_ws}/install/setup.bash" ]]; then
  echo "Isaac ROS workspace is not built: ${isaac_ros_ws}/install/setup.bash" >&2
  echo "Set ISAAC_ROS_WS to the source workspace that contains the official Isaac ROS overlay." >&2
  exit 2
fi
export ISAAC_ROS_WS="${isaac_ros_ws}"
manipulation_ws="${ARX_R5A_MANIPULATION_WS:-${isaac_ros_ws}/isaac_ros_manipulation_arx_r5a}"
sim_ws="${ARX_SIM_WS:-${HOME}/workspace/arx_r5_sim_ws}"
isaac_ros_python_site="${ISAAC_ROS_PYTHON_SITE:-/var/lib/isaac-ros-cli/isaac-ros/lib/python3.12/site-packages}"
isaac_ros_test_python_site="${ISAAC_ROS_TEST_PYTHON_SITE:-${isaac_ros_ws}/isaac_ros_common/isaac_ros_test}"
canonical_isaac_ros_test_python_site="${default_isaac_ros_ws}/isaac_ros_common/isaac_ros_test"
# A common mistake is exporting the parent ``isaac_ros_common`` directory (or
# an unrelated workspace) instead of the Python package root.  That creates a
# namespace package named ``isaac_ros_test`` which imports but does not expose
# IsaacROSBaseTest, and the launch then shuts down after partially starting.
if [[ ! -f "${isaac_ros_test_python_site}/isaac_ros_test/__init__.py" &&
      -f "${canonical_isaac_ros_test_python_site}/isaac_ros_test/__init__.py" ]]; then
  echo "Ignoring ISAAC_ROS_TEST_PYTHON_SITE=${isaac_ros_test_python_site}; it is not a Python package root." >&2
  echo "Using ${canonical_isaac_ros_test_python_site}." >&2
  isaac_ros_test_python_site="${canonical_isaac_ros_test_python_site}"
fi
export ISAAC_ROS_TEST_PYTHON_SITE="${isaac_ros_test_python_site}"
wait_seconds="${ARX_DEMO_WAIT_SECONDS:-30}"
demo_launch_file="${ARX_DEMO_LAUNCH_FILE:-arx_r5a_apriltag_demo.launch.py}"
demo_camera_label="${ARX_DEMO_CAMERA_LABEL:-generated fixed Camera_1}"
color_image_topic="${ARX_DEMO_COLOR_IMAGE_TOPIC:-/camera_1/color/image_raw}"
color_info_topic="${ARX_DEMO_COLOR_INFO_TOPIC:-/camera_1/color/camera_info}"
sim_start_command="${ARX_DEMO_SIM_COMMAND:-${repo_root}/scripts/run_isaac_sim.sh --scene tabletop}"

usage() {
  cat <<EOF
Run the ROS 2 half of the ARX R5A tabletop AprilTag pick-and-place demo.

Start Isaac Sim first in a clean conda isaaclab terminal:
  ${sim_start_command}

Then run this script inside an \`isaac-ros activate\` shell:
  ${0} [ROS launch arguments]

Examples:
  ${0}
  ${0} headless:=True start_rviz:=False
  ${0} auto_start:=False

Selected camera: ${demo_camera_label}
ROS launch file: ${demo_launch_file}

Environment overrides:
  ROS_SETUP, ISAAC_ROS_WS, ARX_R5A_MANIPULATION_WS,
  ARX_R5A_MANIPULATION_SETUP, ARX_SIM_WS, ISAAC_ROS_PYTHON_SITE,
  ISAAC_ROS_TEST_PYTHON_SITE, ARX_DEMO_WAIT_SECONDS, ROS_DOMAIN_ID,
  RMW_IMPLEMENTATION, ROS_AUTOMATIC_DISCOVERY_RANGE
EOF
}

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
  usage
  exit 0
fi

source_setup() {
  local setup_file="$1"
  if [[ -f "${setup_file}" ]]; then
    # shellcheck disable=SC1090
    set +u
    source "${setup_file}"
    set -u
  fi
}

prepend_python_path() {
  local directory="$1"
  [[ -d "${directory}" ]] || return 0
  case ":${PYTHONPATH:-}:" in
    *":${directory}:"*) ;;
    *) export PYTHONPATH="${directory}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

if [[ ! -f "${ros_setup}" ]]; then
  echo "ROS setup file not found: ${ros_setup}" >&2
  exit 2
fi

source_setup "${ros_setup}"
source_setup "${isaac_ros_ws}/install/setup.bash"

if [[ -n "${ARX_R5A_MANIPULATION_SETUP:-}" ]]; then
  if [[ ! -f "${ARX_R5A_MANIPULATION_SETUP}" ]]; then
    echo "ARX manipulation setup file not found: ${ARX_R5A_MANIPULATION_SETUP}" >&2
    exit 2
  fi
  source_setup "${ARX_R5A_MANIPULATION_SETUP}"
elif [[ -f "${manipulation_ws}/install/setup.bash" ]]; then
  source_setup "${manipulation_ws}/install/setup.bash"
elif [[ -f "${manipulation_ws}/install_ros/setup.bash" ]]; then
  source_setup "${manipulation_ws}/install_ros/setup.bash"
fi

if [[ ! -f "${sim_ws}/install/setup.bash" ]]; then
  echo "Simulation workspace is not built: ${sim_ws}/install/setup.bash" >&2
  echo "Build arx_r5_isaac_sim_bringup first; see docs/apriltag-pick-place-demo.md." >&2
  exit 2
fi
source_setup "${sim_ws}/install/setup.bash"

prepend_python_path "${isaac_ros_python_site}"
prepend_python_path "${isaac_ros_test_python_site}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-25}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 was not found. Run this script inside an 'isaac-ros activate' shell." >&2
  exit 2
fi

required_packages=(
  arx_r5_isaac_sim_bringup
  isaac_ros_apriltag
  isaac_ros_apriltag_interfaces
  isaac_ros_cumotion
  isaac_ros_cumotion_object_attachment
  isaac_ros_manipulation_arx_r5a_apriltag
  isaac_ros_manipulation_arx_r5a_bringup
  isaac_ros_manipulation_pick_and_place
)
missing_packages=()
for package_name in "${required_packages[@]}"; do
  if ! ros2 pkg prefix "${package_name}" >/dev/null 2>&1; then
    missing_packages+=("${package_name}")
  fi
done

if (( ${#missing_packages[@]} > 0 )); then
  printf 'Missing ROS package: %s\n' "${missing_packages[@]}" >&2
  echo "Source/build the Isaac ROS and ARX manipulation overlays." >&2
  if [[ " ${missing_packages[*]} " == *' isaac_ros_apriltag '* ]] || \
     [[ " ${missing_packages[*]} " == *' isaac_ros_cumotion_object_attachment '* ]]; then
    echo "Install the released binary dependencies with:" >&2
    echo "  sudo apt install ros-jazzy-isaac-ros-apriltag \\" >&2
    echo "    ros-jazzy-isaac-ros-cumotion-object-attachment" >&2
  fi
  exit 2
fi

if ! /usr/bin/python3 -c 'import torch' >/dev/null 2>&1; then
  echo "/usr/bin/python3 cannot import torch." >&2
  echo "Set ISAAC_ROS_PYTHON_SITE to the Isaac ROS Python site-packages directory." >&2
  echo "Current value: ${isaac_ros_python_site}" >&2
  exit 2
fi

if ! /usr/bin/python3 -c 'import py_trees, py_trees_ros' >/dev/null 2>&1; then
  echo "/usr/bin/python3 cannot import py_trees and py_trees_ros." >&2
  echo "Install the behavior-tree runtime with:" >&2
  echo "  sudo apt install ros-jazzy-py-trees ros-jazzy-py-trees-ros" >&2
  exit 2
fi

if ! isaac_ros_test_origin="$(
  /usr/bin/python3 -c \
    'import isaac_ros_test; from isaac_ros_test import IsaacROSBaseTest; print(isaac_ros_test.__file__ or "")' \
    2>/dev/null
)" || [[ -z "${isaac_ros_test_origin}" ]]; then
  echo "/usr/bin/python3 cannot import IsaacROSBaseTest from isaac_ros_test." >&2
  echo "The launch stack imports this Isaac ROS test utility at runtime." >&2
  echo "Expected source directory: ${isaac_ros_test_python_site}" >&2
  echo "Set ISAAC_ROS_WS or ISAAC_ROS_TEST_PYTHON_SITE to the directory containing isaac_ros_test/__init__.py." >&2
  exit 2
fi
echo "Using isaac_ros_test from ${isaac_ros_test_origin}."

description_share="$(
  ros2 pkg prefix --share isaac_ros_manipulation_arx_r5a_robot_description
)"
cumotion_urdf="${description_share}/urdf/r5a_cumotion.urdf"
cumotion_xrdf="${description_share}/xrdf/r5a.xrdf"
if ! grep -Fq '<link name="grasp_frame"' "${cumotion_urdf}" || \
   ! grep -Fq 'frame_name: "attached_object"' "${cumotion_xrdf}"; then
  echo "The installed ARX description is older than v0.3.0." >&2
  echo "The pick-and-place tree requires grasp_frame and attached_object." >&2
  echo "Checkout v0.3.0 in Isaac_Ros_CuMotion_ArxR5a, rebuild, and re-source the simulation overlay." >&2
  exit 2
fi

if [[ ! "${wait_seconds}" =~ ^[0-9]+$ ]]; then
  echo "ARX_DEMO_WAIT_SECONDS must be a non-negative integer: ${wait_seconds}" >&2
  exit 2
fi

required_topics=(
  /clock
  /isaac_joint_states
  "${color_image_topic}"
  "${color_info_topic}"
)

if (( wait_seconds > 0 )); then
  echo "Waiting up to ${wait_seconds}s for ${demo_camera_label}..."
  deadline=$((SECONDS + wait_seconds))
  while true; do
    topic_list="$(ros2 topic list 2>/dev/null || true)"
    missing_topics=()
    for topic_name in "${required_topics[@]}"; do
      if ! grep -Fxq "${topic_name}" <<<"${topic_list}"; then
        missing_topics+=("${topic_name}")
      fi
    done
    if (( ${#missing_topics[@]} == 0 )); then
      break
    fi
    if (( SECONDS >= deadline )); then
      printf 'Missing Isaac Sim topic: %s\n' "${missing_topics[@]}" >&2
      echo "Start Terminal 1 with: ${sim_start_command}" >&2
      exit 2
    fi
    sleep 1
  done
fi

clock_publishers="$(
  ros2 topic info /clock 2>/dev/null |
    awk -F ': ' '$1 == "Publisher count" { print $2 }' || true
)"
if [[ ! "${clock_publishers}" =~ ^[0-9]+$ ]] || (( clock_publishers != 1 )); then
  echo "Expected exactly one /clock publisher, found ${clock_publishers:-unknown}." >&2
  echo "Stop duplicate Isaac Sim processes before launching the ROS stack." >&2
  exit 2
fi

echo "Starting ${demo_camera_label} manipulation on ROS_DOMAIN_ID=${ROS_DOMAIN_ID}."
exec ros2 launch arx_r5_isaac_sim_bringup \
  "${demo_launch_file}" \
  "$@"
