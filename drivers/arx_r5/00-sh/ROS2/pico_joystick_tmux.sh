#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(realpath "${script_dir}/../..")
export PICO_CONFIG="$repo_dir/ROS2/R5_ws/src/pico_r5_bridge/config/joystick.yaml"
echo '摇杆平移无需握持：左摇杆前后/左右，右摇杆升降；右握持仅启用相对姿态。'
echo '回中停止平移，松开握持停止姿态跟随。旧 r5-pico 会话存在时不会切换模式。'
exec bash "$script_dir/pico_tmux.sh" "$@"
