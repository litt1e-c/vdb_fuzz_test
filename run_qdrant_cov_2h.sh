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
QDRANT_MAX_WORKERS="${QDRANT_MAX_WORKERS:-8}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-7200}"
BUDGET_SCHEDULE="${BUDGET_SCHEDULE:-random}"
BUDGET_SEED="${BUDGET_SEED:-$(date -u +%s)}"
BUDGET_MIN_FILL_JOB_SECONDS="${BUDGET_MIN_FILL_JOB_SECONDS:-2.0}"
BUDGET_FILL_OVERPROVISION="${BUDGET_FILL_OVERPROVISION:-1.25}"
BUDGET_MAX_PLANNED_JOBS="${BUDGET_MAX_PLANNED_JOBS:-5000}"
BUDGET_LAUNCH_GUARD_SECONDS="${BUDGET_LAUNCH_GUARD_SECONDS:-8}"
COVERAGE_TIMELINE_INTERVAL_JOBS="${COVERAGE_TIMELINE_INTERVAL_JOBS:-1}"
COVERAGE_TIMELINE_MIN_INTERVAL_SECONDS="${COVERAGE_TIMELINE_MIN_INTERVAL_SECONDS:-360}"
COVERAGE_TIMELINE_DENSE_JOBS="${COVERAGE_TIMELINE_DENSE_JOBS:-12}"
COVERAGE_TIMELINE_DENSE_MINUTES="${COVERAGE_TIMELINE_DENSE_MINUTES:-20}"
RUN_ID="${RUN_ID:-final-2h-coverfirst-$(date -u +%Y%m%dT%H%M%SZ)}"

export QDRANT_SKIP_BUILD="${QDRANT_SKIP_BUILD:-1}"
export QDRANT_MAX_WORKERS

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
  --coverage-timeline-interval-jobs "${COVERAGE_TIMELINE_INTERVAL_JOBS}" \
  --coverage-timeline-min-interval-seconds "${COVERAGE_TIMELINE_MIN_INTERVAL_SECONDS}" \
  --coverage-timeline-dense-jobs "${COVERAGE_TIMELINE_DENSE_JOBS}" \
  --coverage-timeline-dense-minutes "${COVERAGE_TIMELINE_DENSE_MINUTES}" \
  --time-budget-seconds "${TIME_BUDGET_SECONDS}" \
  --budget-schedule "${BUDGET_SCHEDULE}" \
  --budget-seed "${BUDGET_SEED}" \
  --budget-min-fill-job-seconds "${BUDGET_MIN_FILL_JOB_SECONDS}" \
  --budget-fill-overprovision "${BUDGET_FILL_OVERPROVISION}" \
  --budget-max-planned-jobs "${BUDGET_MAX_PLANNED_JOBS}" \
  --budget-launch-guard-seconds "${BUDGET_LAUNCH_GUARD_SECONDS}" \
  "$@"
