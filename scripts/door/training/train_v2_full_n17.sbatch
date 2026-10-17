#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
#
# Slurm wrapper for scripts/train_v2_full_n17.sh. Edit the #SBATCH lines and
# the two paths for your cluster, then submit FROM THE BUNDLE ROOT:
#   cd <bundle> && sbatch scripts/train_v2_full_n17.sbatch
#SBATCH --job-name=arx-door-v2-full
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export GR00T_REPO="${GR00T_REPO:-/data2/buaalibozhao/gr00t-n1.7/Isaac-GR00T}"
export GR00T_TRAIN_PYTHON="${GR00T_TRAIN_PYTHON:-/data2/buaalibozhao/gr00t-n1.7/venv/bin/python}"
export N17_OUTPUT_DIR="${N17_OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/checkpoints/arx_door_v2_full_${SLURM_JOB_ID}}"
export N17_NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"
export OMP_NUM_THREADS=2
printf 'Job %s on %s at %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
exec srun --unbuffered --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpu-bind=none \
  bash "${SLURM_SUBMIT_DIR}/scripts/train_v2_full_n17.sh"
