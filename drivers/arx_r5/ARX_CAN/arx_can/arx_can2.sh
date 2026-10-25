#!/bin/bash

CAN_DEVICE="/dev/arxcan2"
CAN_INTERFACE="slcan2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/arx_can_common.sh"
