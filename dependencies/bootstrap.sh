#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
# Initialise the pinned AutoData checkout from dependencies/autodata.lock.json.
# Never follows a branch: git checks out the gitlink recorded in this repository.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
with_lfs=false
check_only=false

usage() {
  cat <<'TEXT'
Usage: dependencies/bootstrap.sh [--with-lfs] [--check]
Default: initialise the pinned AutoData/Arena/Lab submodules, then verify pins.
--check       verify the pins only; change nothing.
--with-lfs    additionally fetch the AutoData LFS example datasets.
The version source is dependencies/autodata.lock.json; the gitlink in the
parent tree must agree with it, otherwise this script refuses to run.
TEXT
}
while (($#)); do
  case "$1" in
    --with-lfs) with_lfs=true; shift ;;
    --check) check_only=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

source_path="$(python3 - "${project_root}" <<'PY'
import json, pathlib, sys
lock = pathlib.Path(sys.argv[1], 'dependencies/autodata.lock.json')
print(json.loads(lock.read_text())['source']['path'])
PY
)"

verify() {
  # --require-checkout only after initialisation; --check reports absence.
  if [[ "$1" == required ]]; then
    python3 "${project_root}/dependencies/verify_pins.py" --project-root "${project_root}" \
      --require-checkout --strict-clean
  else
    python3 "${project_root}/dependencies/verify_pins.py" --project-root "${project_root}"
  fi
}

# The gitlink is authoritative for checkout; the lock is authoritative for the
# container and its labels. Refuse to initialise when the two records disagree.
if ! verify check; then
  echo 'Refusing to initialise: the gitlink and dependencies/autodata.lock.json disagree.' >&2
  echo 'Bump both in one commit before running bootstrap.' >&2
  exit 2
fi
if [[ "${check_only}" == true ]]; then
  verify check
  exit 0
fi

# An existing checkout is never reset: a different revision may be deliberate.
for path in "$source_path" \
            "$source_path/submodules/IsaacLab-Arena" \
            "$source_path/submodules/IsaacLab-Arena/submodules/IsaacLab"; do
  [[ -e "${project_root}/${path}/.git" ]] || continue
  if ! git -C "${project_root}/${path}" diff --quiet HEAD --; then
    echo "Modified vendor source, refusing to touch it: ${path}" >&2
    exit 2
  fi
done

# The nested .gitmodules uses SSH URLs; rewrite them for this command only.
# GIT_LFS_SKIP_SMUDGE keeps the example datasets as pointers until --with-lfs.
if ! GIT_LFS_SKIP_SMUDGE=1 git -C "${project_root}" \
     -c url.https://github.com/.insteadOf=git@github.com: \
     submodule update --init --recursive --depth 1 -- "${source_path}"; then
  echo 'Shallow fetch failed; retrying without --depth.' >&2
  GIT_LFS_SKIP_SMUDGE=1 git -C "${project_root}" \
    -c url.https://github.com/.insteadOf=git@github.com: \
    submodule update --init --recursive -- "${source_path}"
fi

if [[ "${with_lfs}" == true ]]; then
  git -C "${project_root}/${source_path}" lfs install --local
  git -C "${project_root}/${source_path}" lfs pull \
    --include='datasets/annotated_datasets/dataset_franka_annotated.hdf5,datasets/annotated_datasets/dataset_franka_skillgen_annotated.hdf5'
fi

verify required
echo "AutoData ready at ${source_path}. Next: bash scripts/door/autodata/run_container.sh --official"
