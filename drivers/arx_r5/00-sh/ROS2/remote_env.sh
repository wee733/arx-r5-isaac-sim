#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "此脚本需要在当前终端中 source，不能直接执行。"
  echo "用法: source $0 master|slave"
  exit 2
fi

if [[ $# -ne 1 || ( "$1" != "master" && "$1" != "slave" ) ]]; then
  echo "用法: source ${BASH_SOURCE[0]} master|slave"
  return 2
fi

R5_REMOTE_ROLE="$1"
R5_REMOTE_PEER_IPS="${R5_REMOTE_PEER_IPS:-}"
env_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R5_REMOTE_REPO_DIR=$(realpath "${env_script_dir}/../..")
R5_REMOTE_R5_WS="${R5_REMOTE_REPO_DIR}/ROS2/R5_ws"

if [[ "$R5_REMOTE_ROLE" == "master" ]]; then
  R5_REMOTE_EXPECTED_IP="192.168.17.161"
  R5_REMOTE_DEFAULT_PEER_IP="192.168.17.160"
  R5_REMOTE_CAN_INTERFACE="can0"
  R5_REMOTE_NODE_NAME="arm_master_l"
  R5_REMOTE_LAUNCH_FILE="open_remote_master_single.launch.py"
else
  R5_REMOTE_EXPECTED_IP="192.168.17.160"
  R5_REMOTE_DEFAULT_PEER_IP="192.168.17.161"
  R5_REMOTE_CAN_INTERFACE="slcan1"
  R5_REMOTE_NODE_NAME="arm_slave_l"
  R5_REMOTE_LAUNCH_FILE="open_remote_slave_single.launch.py"
fi

R5_REMOTE_PEER_IPS="${R5_REMOTE_PEER_IPS:-$R5_REMOTE_DEFAULT_PEER_IP}"
IFS=, read -ra requested_peer_ips <<< "$R5_REMOTE_PEER_IPS"
peer_ips=()
declare -A seen_peer_ips=()
for peer_ip in "${requested_peer_ips[@]}"; do
  if [[ ! "$peer_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "错误: 对端 IP 格式无效: $peer_ip"
    return 2
  fi

  IFS=. read -r peer_a peer_b peer_c peer_d <<< "$peer_ip"
  for peer_octet in "$peer_a" "$peer_b" "$peer_c" "$peer_d"; do
    if ((10#$peer_octet > 255)); then
      echo "错误: 对端 IP 格式无效: $peer_ip"
      return 2
    fi
  done

  if [[ -z "${seen_peer_ips[$peer_ip]+x}" ]]; then
    seen_peer_ips[$peer_ip]=1
    peer_ips+=("$peer_ip")
  fi
done
unset requested_peer_ips seen_peer_ips peer_a peer_b peer_c peer_d peer_octet peer_ip

R5_REMOTE_PEER_IPS=$(IFS=,; echo "${peer_ips[*]}")
# CycloneDDS 0.10.x 不支持 Peers 的 AddLocalhost 属性；显式加入
# localhost，确保禁用 multicast 后本机 ros2 CLI 仍能发现控制节点。
cyclone_peers='<Peer Address="localhost"/>'
for peer_ip in "${peer_ips[@]}"; do
  cyclone_peers+="<Peer Address=\"${peer_ip}\"/>"
done
unset peer_ips peer_ip

ros_setup="/opt/ros/jazzy/setup.bash"
if [[ ! -f "$ros_setup" ]]; then
  echo "错误: 找不到 $ros_setup"
  return 1
fi
if ! ip -4 -o addr show | grep -qw "$R5_REMOTE_EXPECTED_IP"; then
  echo "错误: 本机没有配置 ${R5_REMOTE_EXPECTED_IP}/24"
  return 1
fi

# 避免当前 shell 中残留的 ROS 1、其他 ROS 2 发行版或 DDS 配置污染环境。
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ETC_DIR
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE RMW_IMPLEMENTATION CYCLONEDDS_URI

source "$ros_setup"
source "$env_script_dir/workspace_env.sh" || return 1

export ROS_DOMAIN_ID=17
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset ROS_LOCALHOST_ONLY
export CYCLONEDDS_URI="<CycloneDDS><Domain id=\"any\"><General><Interfaces><NetworkInterface address=\"${R5_REMOTE_EXPECTED_IP}\"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><Peers>${cyclone_peers}</Peers></Discovery></Domain></CycloneDDS>"
unset cyclone_peers

export R5_REMOTE_ROLE R5_REMOTE_REPO_DIR R5_REMOTE_R5_WS
export R5_REMOTE_EXPECTED_IP R5_REMOTE_PEER_IPS R5_REMOTE_CAN_INTERFACE
export R5_REMOTE_NODE_NAME R5_REMOTE_LAUNCH_FILE

if ! ros2 pkg prefix rmw_cyclonedds_cpp >/dev/null 2>&1; then
  echo "错误: 当前环境找不到 rmw_cyclonedds_cpp。"
  return 1
fi

echo "已加载 R5 $R5_REMOTE_ROLE 环境: domain=$ROS_DOMAIN_ID, DDS=${R5_REMOTE_EXPECTED_IP} -> peers=${R5_REMOTE_PEER_IPS}"
