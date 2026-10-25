#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$VLA_ROOT"
mode=${1:-help}
[[ $# -eq 0 ]] || shift
case "$mode" in
  groot) exec bash scripts/door/real/groot.sh "$@";;
  build|build-pico)
    extra_paths=();extra_packages=()
    if [[ $mode == build-pico ]];then
      extra_paths=(drivers/arx_r5/ROS2/R5_ws/src/pico_r5_bridge)
      extra_packages=(pico_r5_bridge)
    fi
    exec colcon --log-base log/real build --build-base build/real --install-base install/real \
      --base-paths drivers/arx_r5/ROS2/R5_ws/src/ARX_R5_ros2_V7/arx_r5_controller \
      drivers/arx_r5/ROS2/R5_ws/src/arxmsgros2 ros2/arx_vla_control "${extra_paths[@]}" \
      --packages-up-to arx_r5_controller arx_vla_control "${extra_packages[@]}" --cmake-args -DCMAKE_BUILD_TYPE=Release "$@"
    ;;
  driver)
    ip -o link show "${VLA_CAN:-slcan1}" | grep -q '<[^>]*UP[,>]' || { echo 'CAN is not UP'; exit 1; }
    if pgrep -x R5Controller >/dev/null; then echo 'R5Controller already running';exit 1;fi
    exec 9>"${XDG_RUNTIME_DIR:-/tmp}/arx-vla-driver.lock";flock -n 9 || exit 1
    exec ros2 run arx_r5_controller R5Controller --ros-args -r __node:=arm_vla \
      -p arm_control_type:=normal -p arm_end_type:=0 -p arm_can_id:="${VLA_CAN:-slcan1}" \
      -p arm_pub_topic_name:=/vla_r5/arm_status -p arm_sub_topic_name:=/vla_r5/arm_cmd "$@"
    ;;
  executor|preview|watchdog)
    target=$mode;extra=()
    if [[ $mode == preview ]];then target=executor;extra=(-p execute:=false);fi
    if [[ $mode == executor ]];then extra=(-p execute:=true);fi
    exec 9>"${XDG_RUNTIME_DIR:-/tmp}/arx-vla-${target}.lock";flock -n 9 || exit 1
    exec ros2 run arx_vla_control "$target" --ros-args \
      -p config:="${VLA_CONFIG:-$VLA_ROOT/configs/real/arx_r5.json}" "${extra[@]}" "$@"
    ;;
  test) exec ros2 run arx_vla_control core_test "$@";;
  probe) exec ros2 run arx_vla_control kinematics_probe "$@";;
  smoke) exec python3 scripts/door/real/smoke.py "$@";;
  inspect) exec ros2 topic echo /vla/observation std_msgs/msg/String --once "$@";;
  *) echo 'Usage: control.sh build|build-pico|driver|watchdog|executor|preview|test|probe|inspect|smoke|groot';exit 2;;
esac
