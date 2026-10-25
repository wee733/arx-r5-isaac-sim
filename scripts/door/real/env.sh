#!/usr/bin/env bash
# Source this file. Use local_setup to avoid copied build-time absolute prefixes.
VLA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ETC_DIR
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS ROS_DISCOVERY_SERVER
source /opt/ros/jazzy/setup.bash
if [[ -f "$VLA_ROOT/install/real/local_setup.bash" ]]; then
  source "$VLA_ROOT/install/real/local_setup.bash"
fi
export ROS_DOMAIN_ID=18
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
export PYTHONPATH="$VLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
