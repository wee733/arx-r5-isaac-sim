#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

# Stable launcher path recorded in existing campaign queues; forwards to the
# stage-3 batch launcher. New commands may call scripts/door/skillgen/run_batch.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/door/skillgen/run_batch.sh" "$@"
