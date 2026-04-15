#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
  PYTHON_BIN_DEFAULT="${ROOT_DIR}/venv/bin/python"
else
  PYTHON_BIN_DEFAULT="python3"
fi

PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
QDRANT_SRC="${QDRANT_SRC:-$HOME/qdrant_build_src}"
EXPERIMENT_ROOT="${QDRANT_EXPERIMENT_ROOT:-$HOME/qdrant_artifacts/qdrant_log}"
QDRANT_TARGET_DIR="${QDRANT_TARGET_DIR:-$EXPERIMENT_ROOT/.cov/qdrant/target}"
QDRANT_BINARY="${QDRANT_BINARY:-$QDRANT_TARGET_DIR/debug/qdrant}"
QDRANT_HOST="${QDRANT_HOST:-127.0.0.1}"
QDRANT_PORT="${QDRANT_PORT:-9411}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-9412}"
RUN_ID="${RUN_ID:-final-2h-scalar-$(date -u +%Y%m%dT%H%M%SZ)}"

export QDRANT_SKIP_BUILD="${QDRANT_SKIP_BUILD:-1}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/run_qdrant_cov_suite.py" \
  --qdrant-src "${QDRANT_SRC}" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --qdrant-target-dir "${QDRANT_TARGET_DIR}" \
  --binary "${QDRANT_BINARY}" \
  --run-id "${RUN_ID}" \
  --host "${QDRANT_HOST}" \
  --port "${QDRANT_PORT}" \
  --grpc-port "${QDRANT_GRPC_PORT}" \
  --no-stress \
  --coverage-timeline \
  "$@"
