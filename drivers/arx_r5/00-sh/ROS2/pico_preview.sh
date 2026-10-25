#!/usr/bin/env bash
set -eo pipefail
# Deliberately separate from remote_env.sh: preview stays on loopback/domain 18.
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
if [[ "${1:-}" == "--status" ]]; then
  exec ros2 topic echo /pico_preview/status std_msgs/msg/String --full-length
fi
if [[ $# -ne 0 ]]; then
  echo "用法: bash $0 [--status]"
  exit 2
fi
# Lock prevents duplicate SDK consumers and misleading interleaved preview topics.
exec 9>"${XDG_RUNTIME_DIR:-/tmp}/r5-pico-preview-${UID}.lock"
if ! flock -n 9; then
  echo "Pico preview 已运行；查看状态: bash $0 --status"
  exit 1
fi
config_tmp=$(mktemp /tmp/r5-pico-config.XXXXXX.yaml)
trap 'rm -f -- "$config_tmp"' EXIT
python3 "$script_dir/pico_config.py" \
  "${PICO_CONFIG:-${repo_dir}/ROS2/R5_ws/src/pico_r5_bridge/config/preview.yaml}" > "$config_tmp"
ros2 run pico_r5_bridge pico_preview --ros-args --params-file "$config_tmp"
