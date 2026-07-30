#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

# Terminal 1: run the VLA collection workcell in Isaac Sim with both cameras.
#
# Unlike the AprilTag demo scripts, this one deliberately keeps --camera-mode
# both: ZED X and D455 are triggered by the same OnPhysicsStep node and publish
# with the same simulation timestamp, so the two streams are hard-synchronized
# for recording.
#
# Run from a clean conda isaaclab terminal, without sourcing ROS.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_source="${repo_root}/arx_r5_isaac_sim_bringup"
scene_usd="${ARX_VLA_SCENE_USD:-${package_source}/assets/scenes/arx_vla_scene.usd}"
scene_config="${ARX_VLA_SCENE_CONFIG:-${package_source}/config/vla_scene.yaml}"
task_config="${ARX_VLA_TASK_CONFIG:-${package_source}/config/vla_task.yaml}"
initial_positions="${ARX_VLA_INITIAL_POSITIONS:-0,1,1.5,0,0,0,0.044,0.044}"
# The authored R5A drive defaults are deliberately conservative for interactive
# viewing.  Collection trajectories carry the arm against gravity for several
# seconds. Runtime logs with k=1000 still showed a 0.031--0.043 rad gravity
# offset on joint3 against a 0.030 rad controller goal tolerance. Raising k by
# 1.6x removes that steady-state margin without weakening the success gate;
# damping scales with sqrt(k) to preserve the same near-critical response.
drive_stiffness="${ARX_VLA_DRIVE_STIFFNESS:-1600.0}"
drive_damping="${ARX_VLA_DRIVE_DAMPING:-80.0}"

if [[ ! -f "${scene_usd}" ]]; then
  echo "VLA scene USD not found: ${scene_usd}" >&2
  echo "Generate it first with: ${repo_root}/scripts/run_author_vla_scene.sh" >&2
  exit 2
fi

exec "${repo_root}/scripts/run_isaac_sim.sh" \
  --usd "${scene_usd}" \
  --usd-scene-config "${scene_config}" \
  --task-config "${task_config}" \
  --camera-mode both \
  --reset-usd-joints \
  --drive-stiffness "${drive_stiffness}" \
  --drive-damping "${drive_damping}" \
  --initial-positions="${initial_positions}" \
  "$@"
