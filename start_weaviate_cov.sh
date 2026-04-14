#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run_id>" >&2
  exit 1
fi

RUN_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/data/$RUN_ID"
COV_DIR="$ROOT_DIR/.cov/$RUN_ID"

mkdir -p "$DATA_DIR" "$COV_DIR"

export ASYNC_INDEXING=true
export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
export PERSISTENCE_DATA_PATH="$DATA_DIR"
export DEFAULT_VECTORIZER_MODULE=none
export ENABLE_MODULES=
export QUERY_DEFAULTS_LIMIT=25
export QUERY_MAXIMUM_RESULTS=60000
export CLUSTER_HOSTNAME=node1
export DISABLE_TELEMETRY=true
export GOCOVERDIR="$COV_DIR"
export LIMIT_RESOURCES=true

exec "$ROOT_DIR/weaviate-cov" --host 0.0.0.0 --port 8080 --scheme http
