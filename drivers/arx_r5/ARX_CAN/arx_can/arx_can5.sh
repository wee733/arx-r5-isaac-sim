#!/bin/bash

CAN_DEVICE="/dev/arxcan5"
CAN_INTERFACE="slcan5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/arx_can_common.sh"
