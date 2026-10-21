#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${GROOT_DATA_PYTHON:-${project_root}/generated/runtime/groot_data/.venv/bin/python}"
packages="${project_root}/generated/runtime/lingbot_data/packages"
tool="${1:?usage: run_lingbot_tool.sh <convert_lingbot_dataset.py|validate_lingbot_dataset.py> [args...]}"
shift
case "${tool}" in
  convert_lingbot_dataset.py|validate_lingbot_dataset.py) ;;
  *) echo 'unsupported LingBot data tool' >&2; exit 2 ;;
esac
[[ -d "${packages}/lerobot" ]] || { echo 'isolated LingBot data packages are missing' >&2; exit 2; }
unset PYTHONHOME
export PYTHONPATH="${packages}"
export PATH="${project_root}/generated/runtime/tools/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${project_root}/generated/runtime/tools/usr/lib/x86_64-linux-gnu"
exec "${python_bin}" "${project_root}/scripts/door/dataset/${tool}" "$@"
