#!/usr/bin/env bash
# Train the v2 stacked Poker44 model for live validator reward.
#
# Usage:
#   ./scripts/train_stacked_v2.sh
#   OUTPUT=models/poker44_stacked_live.joblib ./scripts/train_stacked_v2.sh
#   HOLDOUT_SOURCE_DATES=2026-05-08 EXCLUDE_TRAIN_SOURCE_DATES=2026-05-07 ./scripts/train_stacked_v2.sh
#
# After training, deploy only if diagnose_live_scores shows no CRITICAL flags:
#   POKER44_MODEL_PATH=$(pwd)/models/poker44_stacked_live.joblib \
#   POKER44_LOG_SCORE_COMPONENTS=1 pm2 restart wolf_miner_5 --update-env
#   sleep 360
#   python -m training.diagnose_live_scores --log ~/.pm2/logs/wolf-miner-5-out.log --last 1

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT="${OUTPUT:-models/poker44_stacked_live.joblib}"
HOLDOUT_SOURCE_DATES="${HOLDOUT_SOURCE_DATES:-2026-05-08}"
EXCLUDE_TRAIN_SOURCE_DATES="${EXCLUDE_TRAIN_SOURCE_DATES:-2026-05-07}"
TARGET_FPR="${TARGET_FPR:-0.04}"
MAX_VALIDATOR_FPR="${MAX_VALIDATOR_FPR:-0.09}"
HUMAN_WEIGHT="${HUMAN_WEIGHT:-2.0}"
META_C="${META_C:-1.0}"
N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-42}"
MAX_FEATURES="${MAX_FEATURES:-0}"
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.25}"

EXTRA_ARGS=()
if [[ -n "$HOLDOUT_SOURCE_DATES" ]]; then
  EXTRA_ARGS+=(--holdout-source-dates "$HOLDOUT_SOURCE_DATES")
fi
if [[ -n "$EXCLUDE_TRAIN_SOURCE_DATES" ]]; then
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
if [[ "${NO_MINER_VISIBLE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-miner-visible-payload)
fi

mkdir -p "$(dirname "$OUTPUT")" logs

python -m training.train_model_v2 \
  --output "$OUTPUT" \
  --holdout-source-dates "$HOLDOUT_SOURCE_DATES" \
  --exclude-train-source-dates "$EXCLUDE_TRAIN_SOURCE_DATES" \
  --target-fpr "$TARGET_FPR" \
  --max-validator-fpr "$MAX_VALIDATOR_FPR" \
  --calibration-fraction "$CALIBRATION_FRACTION" \
  --human-weight-multiplier "$HUMAN_WEIGHT" \
  --meta-c "$META_C" \
  --n-folds "$N_FOLDS" \
  --seed "$SEED" \
  --max-features "$MAX_FEATURES" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "logs/train_$(basename "$OUTPUT" .joblib)_$(date +%Y%m%d_%H%M%S).log"
