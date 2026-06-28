# CLAUDE.md — Poker44 miner (ensemble-max-blend, 336-feature)

Working notes for this repo. This is a **Bittensor Poker44 (subnet 126) miner**: it scores chunks of poker hands as bot vs human. Last updated 2026-06-28.

---

## ⚠️ Critical gotchas (read first)

- **Editable install points to a DIFFERENT repo.** `pip show poker44` (`poker44 0.1.1`) resolves `poker44`/`poker44_ml`/`training` to the sibling folder **`Poker44_v1`**, NOT this `..._336_fea` folder. Running **from this repo dir** uses local code (cwd wins). Anything that imports these packages **from another directory** (a script in `/tmp`, a notebook, pytest, a service launched elsewhere) silently uses **`Poker44_v1`'s** code. When testing in standalone scripts, `sys.path.insert(0, "<this repo>")` to force local code. Consider `pip install -e .` from here to fix.
- **This repo tracks subnet version `0.1.12`. The live subnet (`../Poker44-subnet`) is `0.1.32`** (as of 2026-06-27) and evolving fast. Always `git fetch` Poker44-subnet to check the live reward/eval rules before optimizing.

---

## 🎯 The live reward is RANK-FIRST (subnet ≥ 0.1.25) — this drives everything

`poker44/score/scoring.py` reward (current live):
```
ap_score        = average_precision_score(y_true, y_pred)
bot_recall, fpr = _recall_at_fpr(y_pred, y_true, max_fpr=0.05)   # best recall at any threshold with FPR ≤ 5%
reward          = 0.75 * ap_score + 0.25 * bot_recall            # no penalty multiplier
```
- **No fixed-0.5 threshold** (validator sweeps it), **no FPR cliff**. Both terms are **rank-based**.
- ⇒ **Calibration, score_remap, score_logit, isotonic blend, threshold placement, and the top-k "bot budget" are ALL MOOT for reward** — they're monotonic and don't change ranking. The commit "Harden validator eval chunk scoring" explicitly killed top-k guessing.
- **The ONLY lever is model ranking (AP) on the live distribution.** Maximize AP; ignore threshold/calibration tricks.
- (Obsolete OLD reward ≤0.1.23: `0.65·AP + 0.35·recall@0.5 · (1−fpr)²`, fixed 0.5, FPR cliff at 0.10. Much of the calibration/remap machinery in this repo was built for that and is now dead weight.)

---

## 🔬 Distribution shift — the core blocker (verified)

Data files in `hands_generator/evaluation_datas/`:
| file | dates | size | schema |
|---|---|---|---|
| `training_benchmark.txt` | 2026-04-30 → 05-08 (9) | ~4280 chunks | OLD (~v1.1) |
| `training_benchmark_v1_12_only.txt` | 2026-05-26 → 06-28 (34) | 486 chunks (balanced 243/243) | current (v1.12) |
| `training_benchmark_v1_12_total.txt` | union (STALE: 04-30→**06-22** only, 4684 chunks) | rebuild to refresh | mixed |

- **Train on `training_benchmark.txt` → test on `_only` = ROC-AUC ≈ 0.50 (random).** Domain classifier separating the two files = **AUC 1.000**. They are *different populations*; old data does NOT generalize to new.
- In-distribution AP on either file is high (~0.9–1.0) but **meaningless** for live (e.g. holdouts of 05-02/05-08 are the easy old regime → inflated AP).
- On live, the deployed model collapses to **near-constant raw output** (~0.006) → AP ≈ 0.5 → reward floor ~0.375. Feature *code* works (offline 227/293 features vary); `payload_view.py` is identical+deterministic (no train/serve skew) — so the collapse is genuine distribution shift on live data.
- Aggregate-feature signal on current data (RandomForest 5-fold on `_only`): AP ≈ 0.87, ROC-AUC ≈ 0.85. **Top tells = action-pattern repetition/diversity** (`schema_*_signature_unique_share`, `schema_street_entropy_std`).
- **FIX: train on current-distribution (v1.12) data only**; the old data is harmful (it dominates `_total`). Accumulate more recent dated chunks via `api.poker44.net/api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD`.

---

## Model architecture (`poker44_ml/`)

`Poker44Model.predict_chunk_scores` (inference.py) = `blend(base models) → _apply_calibrator → _apply_score_remap (threshold_logit_v1) → _apply_score_logit`. All post-blend stages are monotone (irrelevant to AP/reward).
- `stacked.py` `StackedEnsemble`: logistic-regression **meta-learner** stacking base learners (LightGBM/XGBoost/CatBoost/ExtraTrees/RandomForest **+ sequence**).
- `sequence_model.py` `ChunkSetTransformer`: 2-level hierarchical **Set Transformer** (action→hand attention-pool, hand→chunk attention-pool). AdamW (wd 1e-4), grad-clip 1.0, weighted BCE, **best checkpoint by VAL LOSS**, early-stop patience 3.
- `calibration.py`: **`BlendedIsotonicCalibrator`** is the active stack calibrator; `BlendedQuantileCalibrator` kept only as an unpickle shim for old artifacts.

---

## How to train (`scripts/train_stacked_v2.sh`, env-driven)

- **Ensemble (recommended):** `SEQUENCE_ONLY=0 ENABLE_SEQUENCE=1` (trees + sequence stacked). Trees-only: `ENABLE_SEQUENCE=0`. Sequence-only (avoid — overfits small data): `SEQUENCE_ONLY=1` (disables all trees).
- **For AP (the reward metric): only base learners + ensemble + features + data matter.** Do NOT tune `CALIBRATION_OBJECTIVE`/`ISOTONIC_CALIBRATION_BLEND`/`NO_SCORE_REMAP`/`NO_SCORE_LOGIT_TUNE`/bot-budget — they don't change AP.
- `HOLDOUT_SOURCE_DATES` is a **global** split (excluded from all learners + used for the final honest eval), NOT sequence-only. Hold out **recent** dates; date-split is leakage-safe.
- Match the live format: chunks are now **100 hands**, so set `SEQUENCE_MAX_HANDS=100`.
- LR schedule: `SEQUENCE_LEARNING_RATE_SCHEDULE="lr:epochs,cosine:lr_max:lr_min:epochs"` (segment epochs must sum to total). For d_model=72: `"4e-4:1,cosine:1.2e-3:1.5e-4:N"`. `SEQUENCE_HEADS`: minor effect (2 slightly more stable than 4 on small data); don't grow d_model on small data.
- Individual **tree hyperparameters are hardcoded** in `train_model_v2._make_base_models` (~lines 585–670). `--meta-hard-bot-weight/-gamma` are CLI-only (not env-exposed).

---

## Code changes made this session (this repo)

- **Reward now matches the LIVE subnet (0.1.32 rank-first).** `poker44/score/scoring.py` `reward()` rewritten to `0.75·AP + 0.25·recall@(FPR≤0.05)`, penalty=1.0 (was the obsolete `0.65·AP + 0.35·recall@0.5 · (1−fpr)²` with the 0.10 cliff). It's the single source of truth — `train_model_v2._validator_metrics`, `train_model`, `evaluate_model` (via `reward_eval`), and `forward.py` all route through it, so **training/eval `validator_reward` now reports the real live number** (e.g. model `poker44_stacked_robust_F_1` on 06-25→28: validator_reward 0.231 OLD → **0.710** live; AP 0.816, recall@FPR≤0.05 0.393). Old formula kept as `legacy_reward` for comparison only. `reward_eval` live/base/soft modes are now equivalent (no penalty to vary). NOTE: `train_model_v2`'s `_select_score_*_for_validator_reward` tuners become no-ops under rank-first (monotone params don't change AP/recall).
- **Reward-breakdown logging.** Added `scoring.format_reward_breakdown(ap, recall, fpr, reward)` → one-line decomposition `reward = 0.75·AP + 0.25·recall@FPR≤0.05` with each weighted term + per-term *headroom* (`weight·(1−metric)`) and a `-> push X` pointer to the limiting term. Printed in `train_model_v2` holdout + round-trip logs and in `evaluate_model` metric blocks. (e.g. on 06-25→28: `AP=0.6117 + recall=0.0982 -> push recall@FPR<=0.05`, i.e. the recall term is the limiter.)
- Removed the **quantile** stack calibrator; default stack calibrator → **isotonic** (`BlendedIsotonicCalibrator`, env `ISOTONIC_CALIBRATION_BLEND` default 0.5). Quantile class kept as compat shim. (Note: moot under rank-first reward, but isotonic is a fine default.)
- `inference.debug_score_components`: `raw_scores` now = **pre-calibration** (meta only) via single-pass `_raw_model_score_stages`; added `calibrated_scores`. **Submitted `predict_chunk_scores` is byte-identical** (verified vs git HEAD).
- `train_model_v2` holdout/round-trip logs + `neurons/miner.py` logs updated for the pre-cal `raw`/`calibrated` convention.
- **Bot budget** (top-k positive cap) was added to `neurons/miner.py` then **removed** (pointless under rank-first reward). Miner now submits model scores directly.

---

## Live subnet snapshot (`../Poker44-subnet`, 0.1.32, 2026-06-27)

- **Burn = 0** → 100% of emissions distributed to miners by score (was 0.97 → 0.70 → 0.00 over Jun 23–24). Economics maximally favorable.
- Competition epoch **120h (5-day)**; daily 24h eval windows → dashboard/reward settle slowly (give changes ~5 days).
- Eval workload **backend-controlled "full snapshots"** (no local chunk_count/reward_window); **100-hand chunks**; **miner query timeout 120s** (confirm the miner handles large batches in time).
- **Encrypted audit lane** (`poker44/validator/audit.py`) records each miner's manifest (repo, commit, implementation hash/files, data attestations), AES-256-GCM + RSA-OAEP to a Poker44 key, optional Verathos LLM review. Keep the manifest honest/consistent.
- Miner-eligibility: validator-permit UIDs with **stake < 17000** are now scored as miners (`POKER44_MIN_VALIDATOR_STAKE`).

### Leaderboard manifest (why fields show N/A)
`build_local_model_manifest` always computes `implementation_sha256` + `implementation_files`. `training_data_statement` / `private_data_attestation` come from env (`POKER44_MODEL_TRAINING_DATA_STATEMENT`, `POKER44_MODEL_PRIVATE_DATA_ATTESTATION`) or defaults — **set them or they show N/A** (empty fields are dropped). Run **your** miner (loads `POKER44_MODEL_PATH`), not the subnet's reference miner (heuristic — ignores the model path, says "🤖 Heuristic Poker44 Miner started").

---

## Bottom line

Under the rank-first reward, success = **a model that ranks live bots above live humans (high AP on the current v1.12 distribution)**. Everything else (calibration, remap, logit, threshold, bot budget) is irrelevant. The current models are trained on old data and collapse on live → the single highest-leverage action is **retraining the ensemble on current-format data** (100-hand chunks, `SEQUENCE_MAX_HANDS=100`, recent dates).
