#!/usr/bin/env bash
set -eo pipefail

usage() {
  echo "用法: $0 master|slave [--peer-ip IP]..."
  echo "  master: Jazzy + 系统 Cyclone DDS, can0, 192.168.17.161"
  echo "  slave:  Jazzy + 系统 Cyclone DDS, slcan1, 192.168.17.160"
  echo "  --peer-ip IP: 指定 DDS 对端；可重复使用，未指定时使用另一端默认 IP"
}

if [[ $# -lt 1 || ( "$1" != "master" && "$1" != "slave" ) ]]; then
  usage
  exit 2
fi

role="$1"
peer_ips=()
shift
while (( $# > 0 )); do
  case "$1" in
    --peer-ip)
      if [[ $# -lt 2 ]]; then
        echo "错误: --peer-ip 后需要提供 IP 地址。"
        usage
        exit 2
      fi
      peer_ips+=("$2")
      shift 2
      ;;
    *)
      echo "错误: 未知参数 $1"
      usage
      exit 2
      ;;
  esac
done

if (( ${#peer_ips[@]} > 0 )); then
  R5_REMOTE_PEER_IPS=$(IFS=,; echo "${peer_ips[*]}")
  export R5_REMOTE_PEER_IPS
else
  unset R5_REMOTE_PEER_IPS
fi
unset peer_ips

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/remote_env.sh" "$role"

expected_ip="$R5_REMOTE_EXPECTED_IP"
can_interface="$R5_REMOTE_CAN_INTERFACE"
node_name="$R5_REMOTE_NODE_NAME"
launch_file="$R5_REMOTE_LAUNCH_FILE"

can_wait_attempts=15
for ((attempt = 1; attempt <= can_wait_attempts; attempt++)); do
  if ip link show "$can_interface" >/dev/null 2>&1; then
    break
  fi

  if ((attempt == can_wait_attempts)); then
    echo "错误: 连续检查 ${can_wait_attempts} 次后仍找不到 CAN 接口 $can_interface。"
    echo "请先启动对应的 CAN 转换程序，再重新运行本脚本。"
    exit 1
  fi

  echo "等待 CAN 接口 $can_interface (${attempt}/${can_wait_attempts})..."
  sleep 1
done

while read -r controller_pid; do
  [[ -n "$controller_pid" ]] || continue
  controller_cmd=$(tr '\0' ' ' < "/proc/${controller_pid}/cmdline" 2>/dev/null || true)
  if [[ "$controller_cmd" == *"__node:=${node_name}"* ]]; then
    echo "错误: ${node_name} 已经在运行，请勿重复启动。"
    echo "PID: ${controller_pid}"
    echo "命令: ${controller_cmd}"
    echo "若确认这是残留进程，请先执行: kill -INT ${controller_pid}"
    exit 1
  fi
done < <(pgrep -x R5Controller 2>/dev/null || true)

# The consolidated single-arm project shares CAN ownership with Pico and VLA.
if pgrep -x R5Controller >/dev/null; then
  echo '已有 R5Controller；请先停止当前控制链再切换主从/Pico/VLA。'
  exit 1
fi
exec 9>"${XDG_RUNTIME_DIR:-/tmp}/arx-vla-driver.lock"
flock -n 9 || { echo '机械臂控制权已被占用'; exit 1; }

echo "角色: $role"
echo "IP: $expected_ip/24"
echo "CAN: $can_interface"
echo "节点: $node_name"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "RMW: $RMW_IMPLEMENTATION"
echo "DDS 对端: $R5_REMOTE_PEER_IPS (multicast disabled)"

launch_pid=""
terminate_group() {
  local group_pid="$1"
  [[ -n "$group_pid" ]] || return 0
  kill -0 -- "-${group_pid}" 2>/dev/null || return 0

  kill -INT -- "-${group_pid}" 2>/dev/null || true
  for _ in 1 2 3; do
    kill -0 -- "-${group_pid}" 2>/dev/null || return 0
    sleep 1
  done

  kill -TERM -- "-${group_pid}" 2>/dev/null || true
  for _ in 1 2 3; do
    kill -0 -- "-${group_pid}" 2>/dev/null || return 0
    sleep 1
  done

  kill -KILL -- "-${group_pid}" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  terminate_group "$launch_pid"
  [[ -z "$launch_pid" ]] || wait "$launch_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 独立进程组保证 Ctrl-C 能完整退出 launch 及其子进程。
setsid --wait ros2 launch arx_r5_controller "$launch_file" \
  arm_name:="$node_name" \
  arm_can_id:="$can_interface" &
launch_pid=$!
wait "$launch_pid"
