#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

mode='basic'
camera_label=''
color_image_topic=''
color_info_topic=''
depth_image_topic=''
depth_info_topic=''
detections_topic=''
raw_detections_topic=''
perception_namespace=''
case "${1:-}" in
  '') ;;
  --zedx)
    mode='zedx'
    camera_label='ZED X eye-to-hand'
    color_image_topic='/zed_x/left/image_raw'
    color_info_topic='/zed_x/left/camera_info'
    depth_image_topic='/zed_x/aligned_depth_to_left/image_raw'
    depth_info_topic='/zed_x/aligned_depth_to_left/camera_info'
    detections_topic='/zed_x/tag_detections'
    raw_detections_topic='/zed_x/tag_detections_raw'
    perception_namespace='zed_x'
    ;;
  --d455)
    mode='d455'
    camera_label='D455 eye-in-hand'
    color_image_topic='/d455/color/image_raw'
    color_info_topic='/d455/color/camera_info'
    depth_image_topic='/d455/aligned_depth_to_color/image_raw'
    depth_info_topic='/d455/aligned_depth_to_color/camera_info'
    detections_topic='/d455/tag_detections'
    raw_detections_topic='/d455/tag_detections_raw'
    perception_namespace='d455'
    ;;
  -h|--help)
    cat <<'EOF'
Usage: check_ros_graph.sh [--zedx|--d455]

Without arguments, check the ARX R5A Isaac Sim joint/control contract.
With a camera mode, also check that mode's image, CameraInfo, depth, Isaac ROS
AprilTag CUDA output, drop-pose, and pick-and-place action contracts. --zedx
is the fixed eye-to-hand camera; --d455 is the moving eye-in-hand camera. This
script is read-only and never sends an action goal.
EOF
    exit 0
    ;;
  *)
    echo "unknown argument: ${1}" >&2
    echo "usage: ${0##*/} [--zedx|--d455]" >&2
    exit 2
    ;;
esac

if (( $# > 1 )); then
  echo "usage: ${0##*/} [--zedx|--d455]" >&2
  exit 2
fi

discovery_seconds="${ARX_GRAPH_DISCOVERY_SECONDS:-5}"
probe_timeout_seconds="${ARX_GRAPH_PROBE_TIMEOUT_SECONDS:-5}"
controller_timeout_seconds="${ARX_GRAPH_CONTROLLER_TIMEOUT_SECONDS:-10}"
for value_name in \
  discovery_seconds \
  probe_timeout_seconds \
  controller_timeout_seconds; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
     ! awk -v value="${value}" 'BEGIN { exit !(value > 0.0) }'; then
    echo "${value_name} must be a positive number: ${value}" >&2
    exit 2
  fi
done

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

if [[ "${mode}" != 'basic' ]]; then
  expected_topics["${color_image_topic}"]='sensor_msgs/msg/Image'
  expected_topics["${color_info_topic}"]='sensor_msgs/msg/CameraInfo'
  expected_topics["${depth_image_topic}"]='sensor_msgs/msg/Image'
  expected_topics["${depth_info_topic}"]='sensor_msgs/msg/CameraInfo'
  expected_topics["/${perception_namespace}/apriltag/image_rect"]='sensor_msgs/msg/Image'
  expected_topics["/${perception_namespace}/apriltag/camera_info_rect"]='sensor_msgs/msg/CameraInfo'
  if [[ -n "${raw_detections_topic}" ]]; then
    expected_topics["${raw_detections_topic}"]='isaac_ros_apriltag_interfaces/msg/AprilTagDetectionArray'
  fi
  expected_topics["${detections_topic}"]='isaac_ros_apriltag_interfaces/msg/AprilTagDetectionArray'
  expected_topics[/arx_r5_demo/drop_pose]='geometry_msgs/msg/PoseStamped'
  expected_actions+=(
    [/get_objects]='isaac_ros_manipulation_interfaces/action/GetObjects'
    [/get_object_pose]='isaac_ros_manipulation_interfaces/action/GetObjectPose'
    [/get_selected_object]='isaac_ros_manipulation_interfaces/action/GetSelectedObject'
    [/multi_object_pick_and_place]='isaac_ros_manipulation_interfaces/action/MultiObjectPickAndPlace'
    [/cumotion/motion_plan]='isaac_ros_cumotion_interfaces/action/MotionPlan'
    [/execute_trajectory]='moveit_msgs/action/ExecuteTrajectory'
    [/attach_object]='isaac_ros_cumotion_interfaces/action/AttachObject'
  )
fi

printf 'Mode: %s\n' "${mode}"
if [[ "${mode}" != 'basic' ]]; then
  printf 'Camera: %s\n' "${camera_label}"
fi
printf 'ROS_DOMAIN_ID: %s\n' "${ROS_DOMAIN_ID:-0 (default)}"
printf 'RMW_IMPLEMENTATION: %s\n' "${RMW_IMPLEMENTATION:-<default>}"

if ! topics="$(
  ros2 topic list \
    --no-daemon \
    --spin-time "${discovery_seconds}" \
    --show-types
)"; then
  echo 'failed to discover ROS topics' >&2
  exit 1
fi
if ! services="$(
  ros2 service list \
    --no-daemon \
    --spin-time "${discovery_seconds}" \
    --include-hidden-services \
    --show-types
)"; then
  echo 'failed to discover ROS services/action servers' >&2
  exit 1
fi

missing=0

interface_type() {
  local interface_name="$1"
  local entries="$2"
  awk -v name="${interface_name}" '
    $1 == name {
      gsub(/^\[/, "", $2)
      gsub(/\]$/, "", $2)
      print $2
      exit
    }
  ' <<<"${entries}"
}

for topic in "${!expected_topics[@]}"; do
  actual_type="$(interface_type "${topic}" "${topics}")"
  if [[ -z "${actual_type}" ]]; then
    echo "missing topic: ${topic}" >&2
    missing=1
    continue
  fi
  if [[ "${actual_type}" != "${expected_topics[$topic]}" ]]; then
    echo "wrong type for ${topic}: ${actual_type}" >&2
    missing=1
  fi
done

for action in "${!expected_actions[@]}"; do
  send_goal_service="${action%/}/_action/send_goal"
  actual_send_goal_type="$(interface_type "${send_goal_service}" "${services}")"
  if [[ -z "${actual_send_goal_type}" ]]; then
    echo "missing action: ${action}" >&2
    missing=1
    continue
  fi
  expected_send_goal_type="${expected_actions[$action]}_SendGoal"
  if [[ "${actual_send_goal_type}" != "${expected_send_goal_type}" ]]; then
    echo "wrong type for ${action}: ${actual_send_goal_type%_SendGoal}" >&2
    missing=1
  fi
done

if [[ "${mode}" != 'basic' ]]; then
  if clock_info="$(
    ros2 topic info /clock \
      --no-daemon \
      --spin-time "${discovery_seconds}" 2>/dev/null
  )"; then
    clock_publishers="$(
      awk -F ': ' '$1 == "Publisher count" { print $2 }' <<<"${clock_info}"
    )"
  else
    clock_publishers=''
  fi
  if [[ ! "${clock_publishers}" =~ ^[0-9]+$ ]] || \
     (( clock_publishers != 1 )); then
    echo "expected exactly one /clock publisher, found ${clock_publishers:-unknown}" >&2
    missing=1
  else
    echo '/clock publisher count: 1'
  fi
fi

probe_topic() {
  local topic_name="$1"
  local message_type
  message_type="$(interface_type "${topic_name}" "${topics}")"
  if [[ -z "${message_type}" ]]; then
    return
  fi
  if timeout "${probe_timeout_seconds}s" \
    ros2 topic echo "${topic_name}" "${message_type}" --once >/dev/null 2>&1; then
    echo "message probe passed: ${topic_name}"
  else
    echo "message probe timed out after ${probe_timeout_seconds}s: ${topic_name}" >&2
    missing=1
  fi
}

probe_topic /clock
probe_topic /isaac_joint_states
if [[ "${mode}" != 'basic' ]]; then
  probe_topic "${color_image_topic}"
  probe_topic "${raw_detections_topic}"
  probe_topic "${detections_topic}"
  if apriltag_backend="$(
    timeout "${probe_timeout_seconds}s" \
      ros2 param get "/${perception_namespace}/apriltag" backends 2>/dev/null
  )" && grep -Eq '(^|[[:space:]])CUDA([[:space:]]|$)' \
      <<<"${apriltag_backend}"; then
    echo 'AprilTag backend: CUDA (Isaac ROS cuAprilTag)'
  else
    echo 'Isaac ROS AprilTag CUDA backend was not confirmed' >&2
    missing=1
  fi
fi

if ! controllers="$(
  timeout "${controller_timeout_seconds}s" ros2 control list_controllers
)"; then
  echo 'failed to query controller_manager' >&2
  exit 1
fi
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
if [[ "${mode}" != 'basic' ]]; then
  echo "ARX R5A Isaac Sim ${camera_label} ROS graph contract is present."
else
  echo 'ARX R5A Isaac Sim ROS graph contract is present.'
fi
