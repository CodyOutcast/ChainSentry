#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="${SESSION_NAME:-chainsentry-train}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda-envs/chainsentry/bin/python}"
ARTIFACT_PATH="${ARTIFACT_PATH:-$ROOT_DIR/backend/app/ml/artifacts/graph-model.pt}"
METRICS_PATH="${METRICS_PATH:-$ROOT_DIR/backend/app/ml/artifacts/graph-model-metrics.json}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/backend/app/ml/artifacts/graph-model-metrics-training.log}"
JSONL_LOG_PATH="${JSONL_LOG_PATH:-$ROOT_DIR/backend/app/ml/artifacts/graph-model-metrics-training.jsonl}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/backend/app/ml/artifacts/graph-model-metrics-report}"
TMUX_STDOUT_LOG="${TMUX_STDOUT_LOG:-$ROOT_DIR/backend/app/ml/artifacts/graph-model-tmux-stdout.log}"
SIZE_PROFILE="${SIZE_PROFILE:-max}"
EPOCHS="${EPOCHS:-18}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-chainsentry}"
SWANLAB_RUN_NAME="${SWANLAB_RUN_NAME:-chainsentry-${SIZE_PROFILE}-$(date +%Y%m%d-%H%M%S)}"
ENABLE_SWANLAB="${ENABLE_SWANLAB:-0}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$(dirname "$ARTIFACT_PATH")"
: > "$TMUX_STDOUT_LOG"

SWANLAB_FLAG=""
if [[ "$ENABLE_SWANLAB" == "1" || "$ENABLE_SWANLAB" == "true" ]]; then
  SWANLAB_FLAG="--enable-swanlab"
elif [[ "$ENABLE_SWANLAB" == "auto" && -n "${SWANLAB_API_KEY:-}" ]]; then
  SWANLAB_FLAG="--enable-swanlab"
fi

TRAIN_SAMPLES_FLAG=""
if [[ -n "$TRAIN_SAMPLES_PER_EPOCH" ]]; then
  TRAIN_SAMPLES_FLAG="--train-samples-per-epoch $TRAIN_SAMPLES_PER_EPOCH"
fi

COMMAND=$(cat <<EOF
cd "$ROOT_DIR" && \
PYTHONPATH=backend "$PYTHON_BIN" -m app.ml.training.train_multidataset_model \
  --artifact-path "$ARTIFACT_PATH" \
  --metrics-path "$METRICS_PATH" \
  --log-path "$LOG_PATH" \
  --jsonl-log-path "$JSONL_LOG_PATH" \
  --report-dir "$REPORT_DIR" \
  --size-profile "$SIZE_PROFILE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --swanlab-project "$SWANLAB_PROJECT" \
  --swanlab-run-name "$SWANLAB_RUN_NAME" \
  $SWANLAB_FLAG \
  $TRAIN_SAMPLES_FLAG \
  2>&1 | tee "$TMUX_STDOUT_LOG"
EOF
)

tmux new-session -d -s "$SESSION_NAME" "$COMMAND"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Text log: $LOG_PATH"
echo "JSONL log: $JSONL_LOG_PATH"
echo "Report dir: $REPORT_DIR"
echo "tmux stdout log: $TMUX_STDOUT_LOG"
