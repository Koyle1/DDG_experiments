#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/../venv/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/run_${timestamp}.txt"

"${PYTHON_BIN}" "${ROOT_DIR}/learn_embedding_gp.py" \
  --generations 40 \
  --num-islands 4 \
  --migration-interval 10 \
  --migration-size 2 \
  --topology ring \
  --population 12 \
  --offspring 48 \
  --rng-seed 42 \
  --grammar lite \
  --num-graphs 36 \
  --min-nodes 18 \
  --max-nodes 42 \
  --max-pairs 1500 \
  --knn-k 6 \
  --max-train-graphs 20 \
  --stress-weight 1.0 \
  --knn-weight 0.35 \
  --length-penalty \
  --log-interval 2 \
  --checkpoint-interval 10 \
  --log-file "${log_file}" \
  "$@"

echo "Log saved to: ${log_file}"
