#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(realpath "${script_dir}/../..")
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ETC_DIR
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS
unset RMW_IMPLEMENTATION CYCLONEDDS_URI ROS_DISCOVERY_SERVER
source /opt/ros/jazzy/setup.bash
source "$script_dir/workspace_env.sh"
export ROS_DOMAIN_ID=18
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
mode="${1:-}"
case "$mode" in
  driver)
    ip link show slcan1 >/dev/null
    if pgrep -x R5Controller >/dev/null; then
      echo "R5Controller 已在运行，拒绝重复占用 CAN。"; exit 1
    fi
    exec 9>"${XDG_RUNTIME_DIR:-/tmp}/arx-vla-driver.lock"
    flock -n 9 || exit 1
    exec ros2 run arx_r5_controller R5Controller --ros-args \
      -r __node:=arm_pico -p arm_can_id:=slcan1 -p arm_control_type:=normal \
      -p arm_end_type:=0 -p arm_pub_topic_name:=/pico_r5/arm_status -p arm_sub_topic_name:=/pico_r5/arm_cmd
    ;;
  ik-preview|execute|execute-full)
    exec 9>"${XDG_RUNTIME_DIR:-/tmp}/r5-pico-ik-${UID}.lock"
    flock -n 9 || { echo "Pico IK 已在运行"; exit 1; }
    execute=false
    [[ "$mode" == ik-preview ]] || execute=true
    extra_args=()
    if [[ "$mode" == execute-full || ( "$mode" == ik-preview && -n ${PICO_CONFIG:-} ) ]]; then
      config_tmp=$(mktemp /tmp/r5-pico-config.XXXXXX.yaml)
      trap 'rm -f -- "$config_tmp"' EXIT
      python3 "$script_dir/pico_config.py" \
        "${PICO_CONFIG:-${repo_dir}/ROS2/R5_ws/src/pico_r5_bridge/config/preview.yaml}" > "$config_tmp"
      ros2 run pico_r5_bridge pico_ik_bridge --ros-args -p execute:="$execute" --params-file "$config_tmp"
      exit $?
    fi
    exec ros2 run pico_r5_bridge pico_ik_bridge --ros-args -p execute:="$execute" "${extra_args[@]}"
    ;;
  status)
    exec ros2 topic echo /pico_ik/status std_msgs/msg/String --full-length
    ;;
  feedback)
    exec ros2 topic echo /pico_r5/arm_status arx5_arm_msg/msg/RobotStatus --once
    ;;
  orientation-on|gripper-on)
    parameter=orientation_enabled
    [[ "$mode" != gripper-on ]] || parameter=gripper_enabled
    exec ros2 param set /pico_ik_bridge "$parameter" true
    ;;
  *)
    echo "用法: bash $0 driver|ik-preview|execute|execute-full|status|feedback|orientation-on|gripper-on"
    exit 2
    ;;
esac
