#!/usr/bin/env bash
# Shared install for the consolidated project; standalone SDK checkouts still work.
arx_env_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ARX_INSTALL_PREFIX=$(realpath -m "$arx_env_repo/../../install/real")
if [[ ! -f "$ARX_INSTALL_PREFIX/local_setup.bash" ]]; then
  ARX_INSTALL_PREFIX="$arx_env_repo/ROS2/R5_ws/install"
fi
if [[ ! -f "$ARX_INSTALL_PREFIX/local_setup.bash" ]]; then
  echo 'ARX workspace not built. In the main project run: bash scripts/door/real/control.sh build-pico' >&2
  return 1
fi
source "$ARX_INSTALL_PREFIX/local_setup.bash"
unset arx_env_repo
