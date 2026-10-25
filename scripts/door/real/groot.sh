#!/usr/bin/env bash
# B2ARX runtime, independent of the legacy joint-action policy server.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$VLA_ROOT"
mode=${1:-help}
[[ $# -eq 0 ]] || shift
if [[ $mode == setup ]]; then
  exec bash scripts/door/real/setup_groot.sh "$@"
fi
config=${VLA_GROOT_CONFIG:-$VLA_ROOT/configs/real/groot.json}
if [[ $mode == stop ]]; then
  exec python3 scripts/door/real/run_groot.py stop --config "$config" "$@"
fi
mapfile -t paths < <(python3 - "$config" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
for key in ('policy_python','official_source'):print(Path(c[key]).expanduser().absolute())
PY
)
python=${paths[0]}
[[ -x "$python" ]] || { echo 'Run: bash scripts/door/real/groot.sh setup'; exit 1; }
export PYTHONPATH="$VLA_ROOT:${paths[1]}${PYTHONPATH:+:$PYTHONPATH}"
export GROOT_PATCH_MISTRAL=1 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
case "$mode" in
  serve|verify) exec "$python" scripts/door/real/serve_groot.py "$mode" --config "$config" "$@";;
  run|preview|stop) exec "$python" scripts/door/real/run_groot.py "$mode" --config "$config" "$@";;
  *) echo 'Usage: groot.sh setup|serve|verify|run|preview|stop'; exit 2;;
esac
