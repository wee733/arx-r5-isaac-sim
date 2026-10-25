#!/bin/bash

set -u

SLCAN_SPEED="${SLCAN_SPEED:-8}"
CHECK_INTERVAL="${CHECK_INTERVAL:-1}"

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

require_var() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "$name is not set"
        exit 1
    fi
}

start_slcan() {
    if [ ! -e "$CAN_DEVICE" ]; then
        echo "设备 $CAN_DEVICE 不存在"
        return 1
    fi

    if ! sudo modprobe slcan 2>/dev/null; then
        echo "当前内核未提供 slcan 模块，无法使用 slcand。"
        echo "如需使用 USB SLCAN 设备，请安装或编译 slcan 模块；否则请使用原生 CAN 接口。"
        return 1
    fi

    sudo pkill -f "slcand.*${CAN_DEVICE}.*${CAN_INTERFACE}" 2>/dev/null || true
    sudo ip link set "$CAN_INTERFACE" down 2>/dev/null || true

    echo "启动 slcand..."
    if ! sudo slcand -o -f -s"$SLCAN_SPEED" "$CAN_DEVICE" "$CAN_INTERFACE"; then
        echo "slcand 启动失败"
        return 1
    fi

    echo "配置 $CAN_INTERFACE 接口..."
    if ! sudo ip link set "$CAN_INTERFACE" up; then
        echo "启动 $CAN_INTERFACE 接口失败"
        return 1
    fi

    echo "$CAN_INTERFACE 启动成功"
    return 0
}

start_can() {
    start_slcan
}

check_can() {
    local link_info
    local can_state

    link_info="$(ip -details link show "$CAN_INTERFACE" 2>/dev/null)" || return 1

    echo "$link_info" | grep -q "UP" || return 1
    echo "$link_info" | grep -q "link/can" || return 1

    can_state="$(echo "$link_info" | sed -n 's/.*can state \([^ ]*\).*/\1/p')"
    [ "$can_state" = "ERROR-ACTIVE" ]
}

stop_can() {
    sudo ip link set "$CAN_INTERFACE" down 2>/dev/null || true
    sudo pkill -f "slcand.*${CAN_DEVICE}.*${CAN_INTERFACE}" 2>/dev/null || true
}

require_var CAN_DEVICE
require_var CAN_INTERFACE

while true; do
    if check_can; then
        echo "CAN 接口 $CAN_INTERFACE 正常工作"
    else
        echo "$CAN_INTERFACE 掉线，重启中..."
        stop_can
        sleep 1

        if ! start_can; then
            echo "重启 CAN 接口失败，请检查硬件或驱动。"
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
