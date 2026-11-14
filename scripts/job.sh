#!/usr/bin/env bash
# Array-job wrapper: bash scripts/job.sh <script.py> [extra args]; the shard is the Slurm array task id.
set -euo pipefail
export HF_HUB_OFFLINE=1
script="$1"; shift
python -u "scripts/${script}" --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${NSHARDS:-4}" "$@"
