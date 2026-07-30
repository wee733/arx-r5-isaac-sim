#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

# Request one episode-scene reset by seed, and report what the graph sees.
#
# Exists because two things about this topic are easy to get wrong by hand:
#
#  1. ROS_DOMAIN_ID. The sim and the collection stack both run on 25; a shell
#     without it lands on domain 0 and `ros2 topic pub` reports
#     "Waiting for at least 1 matching subscription(s)" forever because the two
#     DDS domains cannot see each other.
#  2. The wire value packs a seed and a request token. Isaac Sim's
#     ROS2Subscriber reports 0 before any message arrives, so the low seed
#     field uses seed + 1. A new token makes a same-seed retry a real reset.
#
# The subscription warning can also appear even when everything is correct: the
# subscriber lives inside an OmniGraph node and is not always visible to the
# publisher's matching check. Confirm the reset in the Isaac Sim terminal, which
# prints "[arx-r5-sim] scene reset seed=N: block at (...)".

set -euo pipefail

seed="${1:-0}"
request_token="${2:-1}"
if [[ ! "${seed}" =~ ^[0-9]+$ ]] || (( seed > 65534 )); then
  echo "Usage: ${0} [seed] [request-token] (seed must be 0..65534)" >&2
  exit 2
fi
if [[ ! "${request_token}" =~ ^[0-9]+$ ]] || (( request_token < 1 || request_token > 32767 )); then
  echo "Usage: ${0} [seed] [request-token] (token must be 1..32767)" >&2
  exit 2
fi
command_value=$(( (request_token << 16) | (seed + 1) ))

ros_setup="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
sim_ws="${ARX_SIM_WS:-${HOME}/workspace/arx_r5_sim_ws}"

if [[ ! -f "${ros_setup}" ]]; then
  echo "ROS setup file not found: ${ros_setup}" >&2
  exit 2
fi

set +u
# shellcheck disable=SC1090
source "${ros_setup}"
if [[ -f "${sim_ws}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${sim_ws}/install/setup.bash"
fi
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-25}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"

if ! ros2 topic list 2>/dev/null | grep -Fxq /clock; then
  echo >&2
  echo "No /clock on domain ${ROS_DOMAIN_ID}: Isaac Sim is not visible." >&2
  echo "Start it with scripts/run_vla_sim.sh, or set ROS_DOMAIN_ID to match." >&2
  exit 2
fi

# The OmniGraph subscriber is not always visible to a publisher's matching
# check, so report the count without treating zero as fatal.
subscriber_count="$(
  ros2 topic info /vla/scene_command 2>/dev/null |
    awk -F ': ' '$1 == "Subscription count" { print $2 }' || true
)"
echo "Subscription count on /vla/scene_command: ${subscriber_count:-unknown}"
if [[ "${subscriber_count:-0}" == '0' ]]; then
  echo "  (0 is not necessarily wrong: the subscriber lives in an OmniGraph"
  echo "   node. Judge by the Isaac Sim terminal output, not by this count.)"
fi

echo "Publishing seed ${seed}, request token ${request_token}, as scene_command ${command_value}."
# Do not block on the matching check; the message goes out regardless.
timeout 5 ros2 topic pub --once /vla/scene_command \
  std_msgs/msg/Int32 "{data: ${command_value}}" 2>&1 |
  grep -v 'Waiting for at least' || true

echo
echo "Check the Isaac Sim terminal for:"
echo "  [arx-r5-sim] scene reset seed=${seed}: block at (...)"
echo "and /vla/scene_reset_ack for the same integer ${command_value}."
echo
echo "If nothing appeared there, the graph node is not receiving. Check that"
echo "the sim was started with --task-config (scripts/run_vla_sim.sh does)."
