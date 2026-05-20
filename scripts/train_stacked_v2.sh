#!/usr/bin/env bash
# Train the v2 stacked Poker44 model.
#
# Usage:
#   ./scripts/train_stacked_v2.sh                       # train with defaults
#   OUTPUT=/path/to/model.joblib ./scripts/train_stacked_v2.sh
#   TARGET_FPR=0.03 HUMAN_WEIGHT=2.5 ./scripts/train_stacked_v2.sh
#
# After training, point the miner at the new artifact:
#   POKER44_MODEL_PATH=/path/to/model.joblib pm2 restart poker44_miner

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT="${OUTPUT:-models/poker44_stacked_v2.joblib}"
HOLDOUT_LATEST_DAYS="${HOLDOUT_LATEST_DAYS:-2}"
HOLDOUT_SOURCE_DATES="${HOLDOUT_SOURCE_DATES:-}"
TARGET_FPR="${TARGET_FPR:-0.04}"
HUMAN_WEIGHT="${HUMAN_WEIGHT:-2.0}"
META_C="${META_C:-1.0}"
N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-42}"
MAX_FEATURES="${MAX_FEATURES:-0}"
BIAS_GRID="${BIAS_GRID:--1.5,-1.0,-0.6,-0.3,0.0,0.3,0.6}"
TEMP_GRID="${TEMP_GRID:-0.6,0.8,1.0,1.2}"

EXTRA_ARGS=()
if [[ -n "$HOLDOUT_SOURCE_DATES" ]]; then
  EXTRA_ARGS+=(--holdout-source-dates "$HOLDOUT_SOURCE_DATES")
fi
if [[ -n "${EXCLUDE_TRAIN_SOURCE_DATES:-}" ]]; then
  EXTRA_ARGS+=(--exclude-train-source-dates "$EXCLUDE_TRAIN_SOURCE_DATES")
fi
if [[ "${PER_SOURCE_DATE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--per-source-date)
fi
if [[ "${DISABLE_LIGHTGBM:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-lightgbm)
fi
if [[ "${DISABLE_XGBOOST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-xgboost)
fi
if [[ "${DISABLE_CATBOOST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-catboost)
fi
if [[ "${ENABLE_GPU_TREES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-gpu-trees)
fi
if [[ "${ENABLE_SEQUENCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-sequence)
  EXTRA_ARGS+=(--sequence-epochs "${SEQUENCE_EPOCHS:-8}")
  EXTRA_ARGS+=(--sequence-batch-size "${SEQUENCE_BATCH_SIZE:-32}")
  EXTRA_ARGS+=(--sequence-learning-rate "${SEQUENCE_LEARNING_RATE:-1e-3}")
  EXTRA_ARGS+=(--sequence-d-model "${SEQUENCE_D_MODEL:-64}")
  EXTRA_ARGS+=(--sequence-heads "${SEQUENCE_HEADS:-4}")
  EXTRA_ARGS+=(--sequence-action-layers "${SEQUENCE_ACTION_LAYERS:-2}")
  EXTRA_ARGS+=(--sequence-hand-layers "${SEQUENCE_HAND_LAYERS:-1}")
  EXTRA_ARGS+=(--sequence-dropout "${SEQUENCE_DROPOUT:-0.1}")
  EXTRA_ARGS+=(--sequence-device "${SEQUENCE_DEVICE:-cpu}")
fi

mkdir -p "$(dirname "$OUTPUT")"

python -m training.train_model_v2 \
  --output "$OUTPUT" \
  --holdout-latest-days "$HOLDOUT_LATEST_DAYS" \
  --target-fpr "$TARGET_FPR" \
  --human-weight-multiplier "$HUMAN_WEIGHT" \
  --meta-c "$META_C" \
  --n-folds "$N_FOLDS" \
  --seed "$SEED" \
  --max-features "$MAX_FEATURES" \
  --score-logit-bias-grid="$BIAS_GRID" \
  --score-logit-temperature-grid="$TEMP_GRID" \
  "${EXTRA_ARGS[@]}"
