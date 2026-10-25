#!/usr/bin/env bash
# Same CANable/slcand settings as the proven ARX scripts, without a restart loop.
set -euo pipefail
device=${VLA_CAN_DEVICE:-/dev/arxcan1}
interface=${VLA_CAN:-slcan1}
if ip link show "$interface" >/dev/null 2>&1; then
  ip -details link show "$interface"
  ip -o link show "$interface" | grep -q '<[^>]*UP[,>]' || {
    echo 'Existing interface is down; inspect ownership before restarting it.' >&2;exit 1;
  }
  exit 0
fi
[[ -e "$device" ]] || { echo "Missing device: $device" >&2;exit 1; }
if pgrep -x slcand >/dev/null || pgrep -x R5Controller >/dev/null;then
  echo 'Existing CAN/arm process detected; do not create a second owner.' >&2;exit 1
fi
sudo -n modprobe slcan
sudo -n slcand -o -f -s8 "$device" "$interface"
sudo -n ip link set "$interface" up
ip -details link show "$interface"
