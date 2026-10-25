#!/usr/bin/env bash
# Full live-control stack. Selected YAML takes effect on node restart.
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(realpath "${script_dir}/../..")
config_path=${PICO_CONFIG:-${repo_dir}/ROS2/R5_ws/src/pico_r5_bridge/config/preview.yaml}
session=r5-pico
icu_dir=/home/thor/.local/share/xrobotoolkit-compat/icu70/usr/lib/aarch64-linux-gnu
service_dir=/opt/apps/roboticsservice

port_ready() { ss -H -lnt "sport = :$1" | grep -q .; }
can_ready() {
  ip -details link show slcan1 2>/dev/null | grep -q 'can state ERROR-ACTIVE' &&
    ip -o link show slcan1 | grep -q '<[^>]*UP[,>]'
}
wait_ready() {
  local label=$1 count=$2
  shift 2
  for ((i=0;i<count;i++)); do
    if "$@"; then return 0; fi
    sleep 1
  done
  echo "$label 未就绪，停止启动后续节点。已有窗格保留，请检查日志。" >&2
  echo "查看：tmux attach -t $session" >&2
  return 1
}

# Internal pane entry points keep the individual processes visible in tmux.
if [[ ${1:-} == --pane ]]; then
  case "${2:-}" in
    xrobot)
      if port_ready 63901 && port_ready 60061; then
        echo '复用已运行的 XRoboToolkit 服务；此窗格只显示日志，不拥有该服务。'
        exec tail -n 15 -F /home/thor/.local/share/xrobotoolkit-compat/service.log
      fi
      cd "$service_dir"
      export LD_LIBRARY_PATH="$icu_dir:$service_dir:$service_dir/lib:$service_dir/SDK/arm64"
      export QT_PLUGIN_PATH="$service_dir/plugins"
      export QT_QML_PATH="$service_dir/qml"
      # Same environment as runService.sh, but foreground for tmux ownership.
      exec ./RoboticsServiceProcess
      ;;
    can) cd "$repo_dir"; exec bash ./ARX_CAN/arx_can/arx_can1.sh ;;
    driver) exec bash "$script_dir/pico_control.sh" driver ;;
    preview) exec bash "$script_dir/pico_preview.sh" ;;
    execute) exec bash "$script_dir/pico_control.sh" execute-full ;;
    *) exit 2 ;;
  esac
fi

case "${1:-}" in
  ''|--detach|--check) ;;
  *) echo "用法：bash $0 [--detach|--check]"; exit 2 ;;
esac
for binary in tmux ss ip timeout pgrep flock python3; do command -v "$binary" >/dev/null; done
for file in "$icu_dir/libicuuc.so.70" "$service_dir/RoboticsServiceProcess" \
  "$repo_dir/ARX_CAN/arx_can/arx_can1.sh" "$repo_dir/../../install/real/local_setup.bash" \
  "$config_path"; do
  [[ -r "$file" ]] || { echo "缺少文件：$file"; exit 1; }
done
python3 "$script_dir/pico_config.py" \
  "$config_path" >/dev/null
if [[ ${1:-} == --check ]]; then
  echo '依赖和文件检查通过；未启动任何服务或机械臂。'
  exit 0
fi
if tmux has-session -t "=$session" 2>/dev/null; then
  existing_config=$(tmux show-options -qv -t "$session" @pico_config)
  if [[ -n "$existing_config" && "$existing_config" != "$config_path" ]]; then
    echo "已有会话使用另一配置：$existing_config"
    echo "本次请求：$config_path。不会静默复用或自动切换，请先妥善停止旧会话。"
    exit 1
  fi
  echo "会话 $session 已存在，不重复启动，也不自动重试失败步骤。"
  [[ ${1:-} != --detach ]] || exit 0
  if [[ -n ${TMUX:-} ]]; then exec tmux switch-client -t "$session"; fi
  exec tmux attach -t "$session"
fi
exec 8>"${XDG_RUNTIME_DIR:-/tmp}/r5-pico-tmux-${UID}.lock"
flock -n 8 || { echo '另一个启动流程正在运行'; exit 1; }
for name in R5Controller pico_preview pico_ik_bridge; do
  if pgrep -x "$name" >/dev/null; then
    echo "$name 已在其他终端运行。先松开握持并确认停止，妥善停止旧进程再启动；脚本不会强杀。"
    exit 1
  fi
done
if pgrep -f 'bash .*arx_can1.sh' >/dev/null; then
  echo '已有 arx_can1.sh 管理 CAN，请先停止旧管理脚本（不关闭 CAN 接口）。'; exit 1
fi
if pgrep -f '^\./RoboticsServiceProcess$|^/opt/apps/roboticsservice/RoboticsServiceProcess' >/dev/null; then
  if ! port_ready 63901 || ! port_ready 60061; then
    echo 'XRoboToolkit 进程存在但端口未就绪，请先排查，不重复启动。'; exit 1
  fi
fi
echo '即将启动真实机械臂控制。请松开握持键、确认周围无障碍。'
echo "读取配置：$config_path；摇杆模式启动前须双摇杆回中。"
read -r -p '输入 START 确认：' confirmation
[[ $confirmation == START ]] || exit 1

new_pane() {
  local title=$1 command pane
  printf -v command 'env PICO_CONFIG=%q bash %q --pane %q' "$config_path" "$script_dir/pico_tmux.sh" "$title"
  if [[ $title == xrobot ]]; then
    pane=$(tmux new-session -d -s "$session" -n control -x 180 -y 54 -P -F '#{pane_id}' 8>&-)
    tmux set-option -t "$session" @pico_config "$config_path"
    tmux set-window-option -t "$session:control" remain-on-exit on
    tmux set-window-option -t "$session:control" pane-border-status top
    tmux set-window-option -t "$session:control" pane-border-format '#{pane_index}: #{pane_title}'
    tmux respawn-pane -k -t "$pane" "$command"
  else
    pane=$(tmux split-window -d -t "$session:control" -P -F '#{pane_id}' "$command")
  fi
  tmux select-pane -t "$pane" -T "$title"
  tmux select-layout -t "$session:control" tiled >/dev/null
}
new_pane xrobot
wait_ready 'XRoboToolkit 网络端口' 20 port_ready 63901
wait_ready 'XRoboToolkit SDK 端口' 20 port_ready 60061
new_pane can
echo "如需输入 CAN sudo 密码，可在另一个终端运行：tmux attach -t $session"
wait_ready 'CAN（若需要 sudo 密码，请在 can 窗格输入）' 60 can_ready
new_pane driver
feedback_ready() { timeout 10s bash "$script_dir/pico_control.sh" feedback >/dev/null; }
wait_ready '机械臂反馈' 3 feedback_ready
new_pane preview
wait_ready 'Pico 位姿进程' 15 pgrep -x pico_preview
# Ensure the SDK connected to the service; Pico itself may still await user connection.
sdk_ready() { ss -H -ntp 'dport = :60061' | grep -q pico_preview; }
wait_ready 'Pico 本地 SDK 连接' 20 sdk_ready
new_pane execute
wait_ready '执行进程' 15 pgrep -x pico_ik_bridge
flock -u 8
echo "启动完成。Pico 连接 Thor Wi-Fi IP，开启 Controller/Send。"
echo "摇杆模式：先双摇杆回中并松开握持；平移无需握持，右握持只启用姿态。"
echo "tmux attach -t $session；Ctrl+b 然后 d 脱离（程序继续运行）。"
[[ ${1:-} != --detach ]] || exit 0
if [[ -n ${TMUX:-} ]]; then exec tmux switch-client -t "$session"; fi
exec tmux attach -t "$session"
