#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run_id>" >&2
  exit 1
fi

RUN_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${QDRANT_SRC:-}" ]]; then
  echo "QDRANT_SRC is required and should point to a local qdrant source checkout" >&2
  exit 1
fi

QDRANT_SRC="$(cd "${QDRANT_SRC}" && pwd)"
if [[ ! -d "${QDRANT_SRC}" ]]; then
  echo "QDRANT_SRC does not exist: ${QDRANT_SRC}" >&2
  exit 1
fi

if [[ ! -f "${QDRANT_SRC}/Cargo.toml" ]]; then
  echo "QDRANT_SRC does not look like a Rust workspace: ${QDRANT_SRC}" >&2
  exit 1
fi

append_flag() {
  local var_name="$1"
  local flag="$2"
  local current="${!var_name:-}"
  if [[ -n "${current}" ]]; then
    printf -v "${var_name}" '%s %s' "${current}" "${flag}"
  else
    printf -v "${var_name}" '%s' "${flag}"
  fi
  export "${var_name}"
}

prefer_local_rust_toolchain() {
  local candidate_roots=()
  if [[ -n "${QDRANT_EXPERIMENT_ROOT:-}" ]]; then
    candidate_roots+=("${QDRANT_EXPERIMENT_ROOT}")
  fi
  candidate_roots+=("${ROOT_DIR}")
  if [[ -n "${HOME:-}" ]]; then
    candidate_roots+=("${HOME}/qdrant_artifacts/qdrant_log")
    candidate_roots+=("${HOME}")
  fi

  local root=""
  local cargo_bin=""
  local rustup_home=""
  local cargo_home=""
  for root in "${candidate_roots[@]}"; do
    [[ -n "${root}" ]] || continue
    rustup_home="${root}/.rustup-local"
    cargo_home="${root}/.cargo-local"
    if compgen -G "${rustup_home}/toolchains/*/bin/cargo" >/dev/null; then
      cargo_bin="$(printf '%s\n' "${rustup_home}"/toolchains/*/bin/cargo | sort | head -n1)"
      export RUSTUP_HOME="${rustup_home}"
      if [[ -d "${cargo_home}" ]]; then
        export CARGO_HOME="${cargo_home}"
        export PATH="${cargo_home}/bin:$(dirname "${cargo_bin}"):${PATH}"
      else
        export PATH="$(dirname "${cargo_bin}"):${PATH}"
      fi
      return 0
    fi
  done
  return 1
}

QDRANT_HOST="${QDRANT_HOST:-127.0.0.1}"
QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
QDRANT_LOG_LEVEL="${QDRANT_LOG_LEVEL:-INFO}"
QDRANT_MAX_WORKERS="${QDRANT_MAX_WORKERS:-8}"
QDRANT_CARGO_PROFILE="${QDRANT_CARGO_PROFILE:-debug}"
QDRANT_EXPERIMENT_ROOT="${QDRANT_EXPERIMENT_ROOT:-${HOME}/qdrant_artifacts}"
QDRANT_STORAGE_DIR="${QDRANT_STORAGE_DIR:-${QDRANT_EXPERIMENT_ROOT}/data/qdrant/${RUN_ID}}"
QDRANT_COV_DIR="${QDRANT_COV_DIR:-${QDRANT_EXPERIMENT_ROOT}/.cov/qdrant/${RUN_ID}}"
QDRANT_TARGET_DIR="${QDRANT_TARGET_DIR:-${QDRANT_EXPERIMENT_ROOT}/.cov/qdrant/target}"
QDRANT_SNAPSHOTS_DIR="${QDRANT_SNAPSHOTS_DIR:-${QDRANT_STORAGE_DIR}/snapshots}"
QDRANT_TEMP_DIR="${QDRANT_TEMP_DIR:-${QDRANT_STORAGE_DIR}/tmp}"
QDRANT_COV_CLEAN="${QDRANT_COV_CLEAN:-0}"
QDRANT_SKIP_BUILD="${QDRANT_SKIP_BUILD:-0}"
QDRANT_LOCKED="${QDRANT_LOCKED:-1}"
QDRANT_FEATURES="${QDRANT_FEATURES:-service_debug data-consistency-check}"
QDRANT_BUILD_ARGS="${QDRANT_BUILD_ARGS:-}"
QDRANT_EXTRA_ARGS="${QDRANT_EXTRA_ARGS:-}"
QDRANT_CONFIG_PATH="${QDRANT_CONFIG_PATH:-}"

prefer_local_rust_toolchain || true

mkdir -p "${QDRANT_STORAGE_DIR}" "${QDRANT_COV_DIR}" "${QDRANT_TARGET_DIR}" "${QDRANT_SNAPSHOTS_DIR}" "${QDRANT_TEMP_DIR}"

if [[ -z "${QDRANT_CONFIG_PATH}" && -f "${QDRANT_SRC}/config/config.yaml" ]]; then
  QDRANT_CONFIG_PATH="${QDRANT_SRC}/config/config.yaml"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found in PATH" >&2
  exit 1
fi

if ! command -v rustc >/dev/null 2>&1; then
  echo "rustc not found in PATH" >&2
  exit 1
fi

export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR="${QDRANT_TARGET_DIR}"
export LLVM_PROFILE_FILE="${QDRANT_COV_DIR}/qdrant-%p-%m.profraw"
append_flag RUSTFLAGS "-C instrument-coverage"
append_flag RUSTDOCFLAGS "-C instrument-coverage"

export QDRANT__SERVICE__HOST="${QDRANT_HOST}"
export QDRANT__SERVICE__HTTP_PORT="${QDRANT_HTTP_PORT}"
export QDRANT__SERVICE__GRPC_PORT="${QDRANT_GRPC_PORT}"
export QDRANT__SERVICE__MAX_WORKERS="${QDRANT_MAX_WORKERS}"
export QDRANT__STORAGE__STORAGE_PATH="${QDRANT_STORAGE_DIR}"
export QDRANT__STORAGE__SNAPSHOTS_PATH="${QDRANT_SNAPSHOTS_DIR}"
export QDRANT__STORAGE__TEMP_PATH="${QDRANT_TEMP_DIR}"
export QDRANT__LOG_LEVEL="${QDRANT_LOG_LEVEL}"

build_args=(build --bin qdrant)
if [[ "${QDRANT_CARGO_PROFILE}" == "release" ]]; then
  build_args+=(--release)
fi
if [[ "${QDRANT_LOCKED}" == "1" ]]; then
  build_args+=(--locked)
fi
if [[ -n "${QDRANT_FEATURES}" ]]; then
  build_args+=(--features "${QDRANT_FEATURES}")
fi

if [[ -n "${QDRANT_BUILD_ARGS}" ]]; then
  read -r -a build_extra <<< "${QDRANT_BUILD_ARGS}"
  build_args+=("${build_extra[@]}")
fi

if [[ "${QDRANT_COV_CLEAN}" == "1" ]]; then
  echo "[qdrant-cov] cleaning instrumented target dir: ${QDRANT_TARGET_DIR}" >&2
  cargo clean --target-dir "${QDRANT_TARGET_DIR}" >/dev/null 2>&1 || true
fi

cd "${QDRANT_SRC}"

if [[ "${QDRANT_SKIP_BUILD}" != "1" ]]; then
  echo "[qdrant-cov] building instrumented qdrant (${QDRANT_CARGO_PROFILE}) ..." >&2
  cargo "${build_args[@]}"
fi

QDRANT_BIN="${QDRANT_TARGET_DIR}/${QDRANT_CARGO_PROFILE}/qdrant"
if [[ ! -x "${QDRANT_BIN}" ]]; then
  echo "instrumented qdrant binary not found: ${QDRANT_BIN}" >&2
  exit 1
fi

runtime_args=()
if [[ -n "${QDRANT_CONFIG_PATH}" ]]; then
  runtime_args+=(--config-path "${QDRANT_CONFIG_PATH}")
fi
if [[ -n "${QDRANT_EXTRA_ARGS}" ]]; then
  read -r -a runtime_extra <<< "${QDRANT_EXTRA_ARGS}"
  runtime_args+=("${runtime_extra[@]}")
fi

echo "[qdrant-cov] run_id=${RUN_ID}" >&2
echo "[qdrant-cov] experiment_root=${QDRANT_EXPERIMENT_ROOT}" >&2
echo "[qdrant-cov] source=${QDRANT_SRC}" >&2
echo "[qdrant-cov] cov_dir=${QDRANT_COV_DIR}" >&2
echo "[qdrant-cov] storage_dir=${QDRANT_STORAGE_DIR}" >&2
echo "[qdrant-cov] max_workers=${QDRANT_MAX_WORKERS}" >&2
echo "[qdrant-cov] snapshots_dir=${QDRANT_SNAPSHOTS_DIR}" >&2
echo "[qdrant-cov] temp_dir=${QDRANT_TEMP_DIR}" >&2
echo "[qdrant-cov] binary=${QDRANT_BIN}" >&2
echo "[qdrant-cov] cargo=$(command -v cargo)" >&2
echo "[qdrant-cov] rustc=$(command -v rustc)" >&2

cd "${QDRANT_EXPERIMENT_ROOT}"
exec "${QDRANT_BIN}" "${runtime_args[@]}"
