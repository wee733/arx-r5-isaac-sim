#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

declare -A expected_topics=(
  [/clock]='rosgraph_msgs/msg/Clock'
  [/isaac_joint_states]='sensor_msgs/msg/JointState'
  [/isaac_joint_commands]='sensor_msgs/msg/JointState'
  [/joint_states]='sensor_msgs/msg/JointState'
)
declare -A expected_actions=(
  [/manipulator_controller/follow_joint_trajectory]='control_msgs/action/FollowJointTrajectory'
  [/gripper_controller/gripper_cmd]='control_msgs/action/GripperCommand'
)

topics="$(ros2 topic list)"
actions="$(ros2 action list)"
missing=0

for topic in "${!expected_topics[@]}"; do
  if ! grep -Fxq "${topic}" <<<"${topics}"; then
    echo "missing topic: ${topic}" >&2
    missing=1
    continue
  fi
  actual_type="$(ros2 topic type "${topic}")"
  if [[ "${actual_type}" != "${expected_topics[$topic]}" ]]; then
    echo "wrong type for ${topic}: ${actual_type}" >&2
    missing=1
  fi
done

for action in "${!expected_actions[@]}"; do
  if ! grep -Fxq "${action}" <<<"${actions}"; then
    echo "missing action: ${action}" >&2
    missing=1
    continue
  fi
  actual_type="$(ros2 action type "${action}")"
  if [[ "${actual_type}" != "${expected_actions[$action]}" ]]; then
    echo "wrong type for ${action}: ${actual_type}" >&2
    missing=1
  fi
done

controllers="$(ros2 control list_controllers)"
for controller in \
  joint_state_broadcaster \
  manipulator_controller \
  gripper_controller; do
  if ! awk -v name="${controller}" \
    '$1 == name && $NF == "active" { found = 1 } END { exit !found }' \
    <<<"${controllers}"; then
    echo "controller is not active: ${controller}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

printf '%s\n' "${controllers}"
echo "ARX R5A Isaac Sim ROS graph contract is present."
