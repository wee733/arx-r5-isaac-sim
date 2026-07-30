#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

# Terminal 2: run the ROS half of the VLA data-collection workcell.
#
# Waits for BOTH camera streams, unlike run_apriltag_demo.sh which waits for
# one. The collection launch has no AprilTag pipeline, no object server and no
# behavior tree, so the required package set is much smaller too.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_setup="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
default_isaac_ros_ws="${HOME}/workspace/isaac_ros_source"
isaac_ros_ws="${ISAAC_ROS_WS:-${default_isaac_ros_ws}}"
sim_ws="${ARX_SIM_WS:-${HOME}/workspace/arx_r5_sim_ws}"
isaac_ros_python_site="${ISAAC_ROS_PYTHON_SITE:-/var/lib/isaac-ros-cli/isaac-ros/lib/python3.12/site-packages}"
wait_seconds="${ARX_VLA_WAIT_SECONDS:-30}"
graph_check_seconds="${ARX_VLA_GRAPH_CHECK_SECONDS:-2}"
sim_start_command="${repo_root}/scripts/run_vla_sim.sh"

usage() {
  cat <<EOF
Run the ROS 2 half of the ARX R5A VLA data-collection workcell.

Start Isaac Sim first in a clean conda isaaclab terminal:
  ${sim_start_command}

Then run this script inside an \`isaac-ros activate\` shell:
  ${0} [ROS launch arguments]

Examples:
  ${0}
  ${0} auto_start:=True episode_count:=20
  ${0} start_rviz:=False record_depth:=True

Trigger a run manually when auto_start is False:
  ros2 service call /vla_episode_driver/collect std_srvs/srv/Trigger

Environment overrides:
  ROS_SETUP, ISAAC_ROS_WS, ARX_SIM_WS, ISAAC_ROS_PYTHON_SITE,
  ARX_VLA_WAIT_SECONDS, ROS_DOMAIN_ID, RMW_IMPLEMENTATION,
  ROS_AUTOMATIC_DISCOVERY_RANGE, ARX_VLA_GRAPH_CHECK_SECONDS
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
if [[ ! -f "${isaac_ros_ws}/install/setup.bash" ]]; then
  echo "Isaac ROS workspace is not built: ${isaac_ros_ws}/install/setup.bash" >&2
  exit 2
fi
if [[ ! -f "${sim_ws}/install/setup.bash" ]]; then
  echo "Simulation workspace is not built: ${sim_ws}/install/setup.bash" >&2
  echo "Build arx_r5_isaac_sim_bringup first." >&2
  exit 2
fi

export ISAAC_ROS_WS="${isaac_ros_ws}"
source_setup "${ros_setup}"
source_setup "${isaac_ros_ws}/install/setup.bash"
source_setup "${sim_ws}/install/setup.bash"

prepend_python_path "${isaac_ros_python_site}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-25}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 was not found. Run this script inside an 'isaac-ros activate' shell." >&2
  exit 2
fi

# Collection needs the motion stack only: no AprilTag, no object server, no
# behavior tree.
required_packages=(
  arx_r5_isaac_sim_bringup
  isaac_ros_cumotion
  isaac_ros_cumotion_interfaces
  isaac_ros_cumotion_object_attachment
  isaac_ros_cumotion_moveit
)
missing_packages=()
for package_name in "${required_packages[@]}"; do
  if ! ros2 pkg prefix "${package_name}" >/dev/null 2>&1; then
    missing_packages+=("${package_name}")
  fi
done
if (( ${#missing_packages[@]} > 0 )); then
  printf 'Missing ROS package: %s\n' "${missing_packages[@]}" >&2
  echo "Source/build the Isaac ROS overlay." >&2
  exit 2
fi

if ! /usr/bin/python3 -c 'import torch' >/dev/null 2>&1; then
  echo "/usr/bin/python3 cannot import torch (needed by cuMotion)." >&2
  echo "Set ISAAC_ROS_PYTHON_SITE to the Isaac ROS Python site-packages directory." >&2
  echo "Current value: ${isaac_ros_python_site}" >&2
  exit 2
fi

if [[ ! "${wait_seconds}" =~ ^[0-9]+$ ]]; then
  echo "ARX_VLA_WAIT_SECONDS must be a non-negative integer: ${wait_seconds}" >&2
  exit 2
fi
if [[ ! "${graph_check_seconds}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ARX_VLA_GRAPH_CHECK_SECONDS must be positive: ${graph_check_seconds}" >&2
  exit 2
fi

# A previous launch can leave a controller/action server alive after its ROS
# terminal was interrupted. Starting a second MoveIt/cuMotion stack then
# produces the misleading rclpy "unexpected result" warning and can route a
# trajectory to the wrong server. Refuse to launch until every singleton is
# gone; this check is intentionally before the camera wait so the failure is
# immediate and actionable.
stale_nodes=(
  /controller_manager
  /move_group
  /cumotion_container
  /manipulator_container
  /vla_episode_driver
  /vla_recorder
)
stale_actions=(
  /manipulator_controller/follow_joint_trajectory
  /gripper_controller/gripper_cmd
  /cumotion/motion_plan
  /execute_trajectory
  /attach_object
)
if ! node_list="$(
  ros2 node list \
    --no-daemon \
    --spin-time "${graph_check_seconds}"
)"; then
  echo 'Failed to inspect the ROS node graph; refusing an unchecked launch.' >&2
  exit 2
fi
# Jazzy's ``ros2 action list`` verb does not expose the generic discovery
# options (``--no-daemon``/``--spin-time``).  Inspect the action send-goal
# services instead; they are the canonical hidden service endpoints and the
# service verb does support a bounded, daemon-free discovery pass.
if ! service_list="$(
  ros2 service list \
    --no-daemon \
    --spin-time "${graph_check_seconds}" \
    --include-hidden-services
)"; then
  echo 'Failed to inspect ROS action services; refusing an unchecked launch.' >&2
  exit 2
fi
graph_conflicts=()
for node_name in "${stale_nodes[@]}"; do
  if grep -Fxq "${node_name}" <<<"${node_list}"; then
    graph_conflicts+=("node ${node_name}")
  fi
done
for action_name in "${stale_actions[@]}"; do
  send_goal_service="${action_name%/}/_action/send_goal"
  if grep -Fxq "${send_goal_service}" <<<"${service_list}"; then
    graph_conflicts+=("action ${action_name}")
  fi
done
if (( ${#graph_conflicts[@]} > 0 )); then
  printf 'ROS graph conflict: %s\n' "${graph_conflicts[@]}" >&2
  echo 'Stop the previous VLA/MoveIt stack (and its Isaac Sim process) before relaunching.' >&2
  exit 2
fi

# Both cameras must be streaming; a single-camera sim would silently record
# half the observations the dataset promises.
required_topics=(
  /clock
  /isaac_joint_states
  /vla/object_attached
  /vla/gripper_close_intent
  /vla/scene_reset_ack
  /zed_x/left/image_raw
  /zed_x/left/camera_info
  /d455/color/image_raw
  /d455/color/camera_info
)

if (( wait_seconds > 0 )); then
  echo "Waiting up to ${wait_seconds}s for both Isaac Sim cameras..."
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

echo "Starting VLA data collection on ROS_DOMAIN_ID=${ROS_DOMAIN_ID}."
exec ros2 launch arx_r5_isaac_sim_bringup arx_r5a_vla_collect.launch.py "$@"
