#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep this vector synchronized with config/d455_observation_positions.yaml.
# It is a retained startup seed, not an authored-USD end-to-end acceptance
# result. Camera/TF/AprilTag perception is verified; original-layout cuMotion
# currently returns INVERSE_KINEMATICS_FAILURE for the source pose.
exec "${repo_root}/scripts/run_isaac_sim.sh" \
  --usd \
  --camera-mode d455 \
  --authored-layout as-authored \
  --reset-usd-joints \
  --initial-positions=-1.3919817209,1.8859872818,0.5746622086,-0.6957563162,-0.6273930669,-1.9993159771,0.044,0.044 \
  "$@"
