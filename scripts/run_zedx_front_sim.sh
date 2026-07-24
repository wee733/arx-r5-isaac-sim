#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
demo_config="${repo_root}/arx_r5_isaac_sim_bringup/config/front_demo_apriltag_demo.yaml"

exec "${repo_root}/scripts/run_isaac_sim.sh" \
  --usd \
  --camera-mode zedx \
  --authored-layout front-demo \
  --demo-config "${demo_config}" \
  --reset-usd-joints \
  --initial-positions=0,1,1.5,0,0,0,0.044,0.044 \
  "$@"
