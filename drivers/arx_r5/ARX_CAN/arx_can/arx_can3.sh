#!/bin/bash

CAN_DEVICE="/dev/arxcan3"
CAN_INTERFACE="slcan3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/arx_can_common.sh"
