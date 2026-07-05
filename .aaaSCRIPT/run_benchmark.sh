#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

DATA_DIR="${HMS_DATA_DIR:-$ROOT_DIR/.aaaDATA}"
LOG_DIR="${HMS_LOG_DIR:-$ROOT_DIR/.aaaLOG}"
RESULT_DIR="${HMS_RESULT_DIR:-$ROOT_DIR/.aaaRESULT}"

mkdir -p "$DATA_DIR" "$LOG_DIR" "$RESULT_DIR"

BENCHMARK="${HMS_BENCHMARK:-longmemeval}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${HMS_BENCHMARK_LOG:-$LOG_DIR/${BENCHMARK}_${TIMESTAMP}.log}"
RESULTS_FILENAME="${HMS_RESULTS_FILENAME:-${BENCHMARK}_${TIMESTAMP}.json}"

COMMON_ARGS=()
PYTHON_BIN="${HMS_PYTHON_BIN:-}"
if [ -n "$PYTHON_BIN" ]; then
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "HMS_PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 2
  fi
  export PYTHONPATH="$ROOT_DIR/core/dataplane:$ROOT_DIR/lab/evaluation${PYTHONPATH:+:$PYTHONPATH}"
fi

# Retrieval-only is the default development mode. It avoids re-running retain
# extraction and reuses memories already stored in the configured database.
if [ "${HMS_RETRIEVAL_ONLY:-1}" = "1" ]; then
  COMMON_ARGS+=(--skip-ingestion)
fi

cd "$ROOT_DIR"

case "$BENCHMARK" in
  longmemeval)
    LONGMEMEVAL_ARGS=(
      --results-dir "$RESULT_DIR"
      --results-filename "$RESULTS_FILENAME"
      "${COMMON_ARGS[@]}"
    )

    if [ -n "${HMS_MAX_INSTANCES:-}" ]; then
      LONGMEMEVAL_ARGS+=(--max-instances "$HMS_MAX_INSTANCES")
    fi

    if [ -n "${HMS_MAX_QUESTIONS:-}" ]; then
      LONGMEMEVAL_ARGS+=(--max-questions "$HMS_MAX_QUESTIONS")
    fi

    if [ -n "${HMS_DATASET_PATH:-}" ]; then
      LONGMEMEVAL_ARGS+=(--dataset-path "$HMS_DATASET_PATH")
    fi

    if [ "${HMS_ENABLE_QUERY_EXPANSION:-0}" = "1" ]; then
      LONGMEMEVAL_ARGS+=(--enable-query-expansion)
      LONGMEMEVAL_ARGS+=(--query-rewriting-strategy "${HMS_QUERY_REWRITING_STRATEGY:-llm_driven}")
    fi

    if [ -n "${HMS_SESSION_EXPANSION_WEIGHT:-}" ]; then
      LONGMEMEVAL_ARGS+=(--session-expansion-weight "$HMS_SESSION_EXPANSION_WEIGHT")
    fi

    case "${HMS_PIPELINE:-}" in
      ledger)
        LONGMEMEVAL_ARGS+=(--oracle-planner-v26)
        ;;
      self_evolution)
        LONGMEMEVAL_ARGS+=(--oracle-planner-v220)
        ;;
      "")
        if [ "${HMS_ORACLE_PLANNER_V26:-0}" = "1" ]; then
          LONGMEMEVAL_ARGS+=(--oracle-planner-v26)
        fi

        if [ "${HMS_ORACLE_PLANNER_V220:-0}" = "1" ]; then
          LONGMEMEVAL_ARGS+=(--oracle-planner-v220)
        fi
        ;;
      *)
        echo "Unsupported HMS_PIPELINE: $HMS_PIPELINE" >&2
        echo "Supported values: ledger, self_evolution" >&2
        exit 2
        ;;
    esac

    if [ -n "$PYTHON_BIN" ]; then
      CMD=("$PYTHON_BIN" -m benchmarks.longmemeval.longmemeval_benchmark "${LONGMEMEVAL_ARGS[@]}" "$@")
    else
      CMD=(uv run --directory lab/evaluation python -m benchmarks.longmemeval.longmemeval_benchmark "${LONGMEMEVAL_ARGS[@]}" "$@")
    fi
    ;;
  *)
    echo "Unsupported HMS_BENCHMARK: $BENCHMARK" >&2
    echo "Supported values: longmemeval" >&2
    exit 2
    ;;
esac

{
  echo "[$(date --iso-8601=seconds)] Running benchmark: ${CMD[*]}"
  "${CMD[@]}"
} 2>&1 | tee "$LOG_FILE"
