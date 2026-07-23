#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ARX_DEMO_LAUNCH_FILE='arx_r5a_d455_eye_in_hand.launch.py'
export ARX_DEMO_CAMERA_LABEL='D455 eye-in-hand camera'
export ARX_DEMO_COLOR_IMAGE_TOPIC='/d455/color/image_raw'
export ARX_DEMO_COLOR_INFO_TOPIC='/d455/color/camera_info'
export ARX_DEMO_SIM_COMMAND="${repo_root}/scripts/run_d455_sim.sh"

exec "${repo_root}/scripts/run_apriltag_demo.sh" "$@"
