#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
GO_BIN="${GO_BIN:-$(command -v go)}"
LAUNCHER="${LAUNCHER:-$ROOT_DIR/start_weaviate_cov.sh}"

TOTAL_BUDGET_SECONDS="${TOTAL_BUDGET_SECONDS:-7200}"
POSTPROCESS_RESERVE_SECONDS="${POSTPROCESS_RESERVE_SECONDS:-300}"
BASELINE_SUITE="${BASELINE_SUITE:-comprehensive}"
BASELINE_TIMELINE="${BASELINE_TIMELINE:-0}"
BASELINE_TIMELINE_INTERVAL_CASES="${BASELINE_TIMELINE_INTERVAL_CASES:-2}"
RANDOM_SUITE="${RANDOM_SUITE:-comprehensive}"
RANDOM_SCHEDULE="${RANDOM_SCHEDULE:-random}"
RANDOM_SUITE_SEED="${RANDOM_SUITE_SEED:-20260416}"
TIMELINE_INTERVAL_CASES="${TIMELINE_INTERVAL_CASES:-4}"
HOST="${WEAVIATE_HOST:-127.0.0.1}"
HTTP_PORT="${WEAVIATE_PORT:-18080}"
GRPC_PORT="${WEAVIATE_GRPC_PORT:-15051}"
AUTO_PORTS="${AUTO_PORTS:-1}"
RUN_ID="${1:-weaviate-2h-guaranteed-$(date -u +%Y%m%d_%H%M%S)}"

LOG_ROOT="${ROOT_DIR}/weaviate_log/${RUN_ID}"
COV_ROOT="${ROOT_DIR}/.cov/${RUN_ID}"
DATA_ROOT="${ROOT_DIR}/data/${RUN_ID}"
mkdir -p "$LOG_ROOT" "$COV_ROOT" "$DATA_ROOT"

port_free() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

if [[ "$AUTO_PORTS" == "1" ]]; then
  while ! port_free "$HTTP_PORT" \
      || ! port_free "$GRPC_PORT" \
      || ! port_free "$((HTTP_PORT + 220))" \
      || ! port_free "$((HTTP_PORT + 221))" \
      || ! port_free "$((HTTP_PORT + 222))" \
      || ! port_free "$((HTTP_PORT + 223))"; do
    HTTP_PORT="$((HTTP_PORT + 10))"
    GRPC_PORT="$((GRPC_PORT + 10))"
  done
fi

BASELINE_RUN_ID="baseline-${RUN_ID}"
RANDOM_RUN_ID="random-${RUN_ID}"
MERGED_RUN_ID="merged-${RUN_ID}"

BASELINE_COV_DIR="${COV_ROOT}/baseline"
BASELINE_DATA_DIR="${DATA_ROOT}/baseline"
RANDOM_COV_DIR="${COV_ROOT}/random"
RANDOM_DATA_DIR="${DATA_ROOT}/random"

BASELINE_LOG_DIR="${LOG_ROOT}/coverage_suite_${BASELINE_RUN_ID}"
RANDOM_LOG_DIR="${LOG_ROOT}/coverage_suite_${RANDOM_RUN_ID}"
MERGED_LOG_DIR="${LOG_ROOT}/coverage_suite_${MERGED_RUN_ID}"
FIGURE_DIR="${LOG_ROOT}/figures"
MERGED_TIMELINE_DIR="${LOG_ROOT}/merged_timeline"
mkdir -p "$FIGURE_DIR" "$MERGED_TIMELINE_DIR"

METADATA_JSON="${LOG_ROOT}/run_metadata.json"
cat > "$METADATA_JSON" <<EOF
{
  "run_id": "${RUN_ID}",
  "python_bin": "${PYTHON_BIN}",
  "go_bin": "${GO_BIN}",
  "launcher": "${LAUNCHER}",
  "total_budget_seconds": ${TOTAL_BUDGET_SECONDS},
  "postprocess_reserve_seconds": ${POSTPROCESS_RESERVE_SECONDS},
  "baseline_suite": "${BASELINE_SUITE}",
  "baseline_timeline": ${BASELINE_TIMELINE},
  "baseline_timeline_interval_cases": ${BASELINE_TIMELINE_INTERVAL_CASES},
  "random_suite": "${RANDOM_SUITE}",
  "random_schedule": "${RANDOM_SCHEDULE}",
  "random_suite_seed": ${RANDOM_SUITE_SEED},
  "timeline_interval_cases": ${TIMELINE_INTERVAL_CASES},
  "host": "${HOST}",
  "port": ${HTTP_PORT},
  "grpc_port": ${GRPC_PORT}
}
EOF

echo "[run] run_id=${RUN_ID}"
echo "[run] port=${HTTP_PORT} grpc=${GRPC_PORT}"
echo "[run] metadata=${METADATA_JSON}"

SUITE_START_TS="$(date +%s)"

BASELINE_EXTRA_ARGS=()
if [[ "$BASELINE_TIMELINE" == "1" ]]; then
  BASELINE_EXTRA_ARGS+=(--coverage-timeline --coverage-timeline-interval-cases "$BASELINE_TIMELINE_INTERVAL_CASES")
fi

echo "[stage-a] baseline suite=${BASELINE_SUITE}"
set +e
"$PYTHON_BIN" "$ROOT_DIR/run_weaviate_cov_suite.py" \
  --suite "$BASELINE_SUITE" \
  --seed-strategy fixed \
  --launcher "$LAUNCHER" \
  --python-bin "$PYTHON_BIN" \
  --go-bin "$GO_BIN" \
  --host "$HOST" \
  --port "$HTTP_PORT" \
  --grpc-port "$GRPC_PORT" \
  --run-id "$BASELINE_RUN_ID" \
  --log-root "$LOG_ROOT" \
  --cov-dir "$BASELINE_COV_DIR" \
  --data-dir "$BASELINE_DATA_DIR" \
  "${BASELINE_EXTRA_ARGS[@]}"
BASELINE_EXIT=$?
set -e
echo "[stage-a] exit=${BASELINE_EXIT}"

BASELINE_SUMMARY_JSON="${BASELINE_LOG_DIR}/suite_summary.json"
if [[ ! -f "$BASELINE_SUMMARY_JSON" ]]; then
  echo "[error] missing baseline summary: ${BASELINE_SUMMARY_JSON}" >&2
  exit 1
fi

BASELINE_ELAPSED_SECONDS="$("$PYTHON_BIN" - "$BASELINE_SUMMARY_JSON" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("suite_elapsed_seconds", 0))
PY
)"
BASELINE_TIMELINE_CSV="${BASELINE_LOG_DIR}/artifacts/coverage_timeline.csv"
BASELINE_INPUTS="$BASELINE_COV_DIR"
if [[ -f "$BASELINE_TIMELINE_CSV" ]]; then
  BASELINE_INPUTS="$("$PYTHON_BIN" - "$BASELINE_TIMELINE_CSV" "$BASELINE_COV_DIR" <<'PY'
import csv
import sys
from pathlib import Path

timeline = Path(sys.argv[1])
fallback = Path(sys.argv[2])
rows = list(csv.DictReader(timeline.open("r", encoding="utf-8", newline="")))
for row in reversed(rows):
    cov_inputs = (row.get("cov_inputs") or "").strip()
    if cov_inputs:
        print(cov_inputs)
        break
else:
    print(fallback)
PY
)"
fi

NOW_TS="$(date +%s)"
ELAPSED_SO_FAR="$((NOW_TS - SUITE_START_TS))"
RANDOM_BUDGET_SECONDS="$((TOTAL_BUDGET_SECONDS - ELAPSED_SO_FAR - POSTPROCESS_RESERVE_SECONDS))"
if (( RANDOM_BUDGET_SECONDS < 0 )); then
  RANDOM_BUDGET_SECONDS=0
fi

echo "[stage-b] remaining fuzz budget=${RANDOM_BUDGET_SECONDS}s"
if (( RANDOM_BUDGET_SECONDS > 0 )); then
  set +e
  "$PYTHON_BIN" "$ROOT_DIR/run_weaviate_cov_suite.py" \
    --suite "$RANDOM_SUITE" \
    --seed-strategy derived \
    --suite-seed "$RANDOM_SUITE_SEED" \
    --time-budget-seconds "$RANDOM_BUDGET_SECONDS" \
    --budget-schedule "$RANDOM_SCHEDULE" \
    --coverage-timeline \
    --coverage-timeline-interval-cases "$TIMELINE_INTERVAL_CASES" \
    --launcher "$LAUNCHER" \
    --python-bin "$PYTHON_BIN" \
    --go-bin "$GO_BIN" \
    --host "$HOST" \
    --port "$HTTP_PORT" \
    --grpc-port "$GRPC_PORT" \
    --run-id "$RANDOM_RUN_ID" \
    --log-root "$LOG_ROOT" \
    --cov-dir "$RANDOM_COV_DIR" \
    --data-dir "$RANDOM_DATA_DIR"
  RANDOM_EXIT=$?
  set -e
  echo "[stage-b] exit=${RANDOM_EXIT}"
else
  RANDOM_EXIT=0
  echo "[stage-b] skipped because no budget remained"
fi

MERGED_INPUTS="$BASELINE_INPUTS"
if [[ -d "$RANDOM_COV_DIR" ]]; then
  while IFS= read -r cov_dir; do
    MERGED_INPUTS="${MERGED_INPUTS},${cov_dir}"
  done < <(find "$RANDOM_COV_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi

echo "[stage-c] merged coverage report"
"$PYTHON_BIN" "$ROOT_DIR/run_weaviate_cov_suite.py" \
  --coverage-report-only \
  --launcher "$LAUNCHER" \
  --python-bin "$PYTHON_BIN" \
  --go-bin "$GO_BIN" \
  --run-id "$MERGED_RUN_ID" \
  --log-root "$LOG_ROOT" \
  --coverage-inputs "$MERGED_INPUTS"

RANDOM_TIMELINE_CSV="${RANDOM_LOG_DIR}/artifacts/coverage_timeline.csv"
if [[ -f "$RANDOM_TIMELINE_CSV" ]]; then
  echo "[stage-d] merged timeline"
  BASELINE_TIMELINE_ARGS=()
  if [[ -f "$BASELINE_TIMELINE_CSV" ]]; then
    BASELINE_TIMELINE_ARGS+=(--baseline-timeline-csv "$BASELINE_TIMELINE_CSV")
  fi
  "$PYTHON_BIN" "$ROOT_DIR/tools/merge_weaviate_timeline_with_baseline.py" \
    --timeline-csv "$RANDOM_TIMELINE_CSV" \
    --baseline-cov-inputs "$BASELINE_INPUTS" \
    --baseline-summary-json "$BASELINE_SUMMARY_JSON" \
    --baseline-elapsed-seconds "$BASELINE_ELAPSED_SECONDS" \
    --output-dir "$MERGED_TIMELINE_DIR" \
    --go-bin "$GO_BIN" \
    "${BASELINE_TIMELINE_ARGS[@]}"

  MERGED_TIMELINE_CSV="${MERGED_TIMELINE_DIR}/coverage_timeline_merged.csv"
  if [[ -f "$MERGED_TIMELINE_CSV" ]]; then
    echo "[stage-e] figures"
    "$PYTHON_BIN" "$ROOT_DIR/tools/gen_weaviate_timeline_figure.py" \
      --timeline-csv "$MERGED_TIMELINE_CSV" \
      --metric-group scalar \
      --output-dir "$FIGURE_DIR" \
      --basename "coverage_timeline_scalar_dual_axis" \
      --title "Weaviate Scalar Target Coverage Growth"
    "$PYTHON_BIN" "$ROOT_DIR/tools/gen_weaviate_timeline_figure.py" \
      --timeline-csv "$MERGED_TIMELINE_CSV" \
      --metric-group overall \
      --output-dir "$FIGURE_DIR" \
      --basename "coverage_timeline_overall_dual_axis" \
      --title "Weaviate Overall Coverage Growth"
  fi
else
  echo "[stage-d] skipped merged timeline because random timeline csv is missing"
fi

echo "[done] baseline summary: ${BASELINE_SUMMARY_JSON}"
echo "[done] merged summary:   ${MERGED_LOG_DIR}/suite_summary.json"
echo "[done] merged markdown:  ${MERGED_LOG_DIR}/suite_summary.md"
echo "[done] merged timeline:  ${MERGED_TIMELINE_DIR}/coverage_timeline_merged.csv"
echo "[done] scalar figure:    ${FIGURE_DIR}/coverage_timeline_scalar_dual_axis.png"
echo "[done] overall figure:   ${FIGURE_DIR}/coverage_timeline_overall_dual_axis.png"
