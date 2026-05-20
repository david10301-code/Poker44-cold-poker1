"""V2 training pipeline: stacked LGB/XGB/CatBoost/ExtraTrees/RandomForest.

Beats the current ``train_model.py`` weighted-mean blend by:

* Out-of-fold (K-fold) **stacking** with a logistic-regression meta-learner,
  trained against the same labels.
* **Isotonic calibration** on stacked OOF scores so the final ranking is
  monotone and well-suited to average precision (65% of the validator reward).
* **Conformal FPR control**: a logit shift picked on a held-out set so that
  chunk-level FPR stays under a target well below the validator's 10% cliff.
* **Asymmetric sample weights** that protect humans (the FPR penalty term is
  squared and binary-cliffed, so a missed human is much worse than a missed
  bot).
* **Score-logit grid search** that maximizes the exact ``validator_reward``
  used by the on-chain scorer rather than just ROC-AUC.

The artifact format stays compatible with :class:`poker44_ml.inference.Poker44Model`:
``models = [StackedEnsemble(...)]`` with ``model_weights = [1.0]``.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Silence the benign LightGBM <-> sklearn 1.7 feature-name validation warning.
# It fires because LightGBM 4.x stores a feature signature on fit even for
# numpy input, which sklearn's validator then compares against later
# (also-numpy) predict-time input. Predictions are correct -- the warning is
# noise that drowns out useful training output. Filter is scoped to this
# specific message, so any other sklearn UserWarning still shows up.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import numpy as np

from poker44.score.scoring import reward
from poker44_ml.inference import Poker44Model
from poker44_ml.stacked import StackedEnsemble
from training.build_dataset import (
    load_benchmark_examples,
    resolve_benchmark_paths,
)

try:
    import joblib
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("joblib is required to train Poker44 models.") from exc

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    from poker44_ml.sequence_model import (
        SequenceModelConfig,
        SequenceModelWrapper,
    )
except Exception:  # pragma: no cover
    SequenceModelConfig = None  # type: ignore[assignment]
    SequenceModelWrapper = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------- helpers ---------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stacked Poker44 model (v2).")
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "models" / "poker44_stacked_v2.joblib"),
    )
    parser.add_argument("--holdout-latest-days", type=int, default=2)
    parser.add_argument("--holdout-source-dates", type=str, default=None)
    parser.add_argument(
        "--exclude-train-source-dates",
        type=str,
        default=None,
        help=(
            "Comma-separated sourceDate values to remove from the training "
            "side only. Use this when a specific date in training causes "
            "negative transfer to the holdout date (i.e. the model "
            "generalizes worse when that date is included). The dates are "
            "removed AFTER the holdout split so they affect training only."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.04,
        help="Conformal target for chunk-level FPR. Stays well below the 0.10 cliff.",
    )
    parser.add_argument(
        "--human-weight-multiplier",
        type=float,
        default=2.0,
        help="Asymmetric sample weight ratio for human chunks (higher = safer).",
    )
    parser.add_argument(
        "--meta-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the logistic meta-learner.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="If > 0, keep only the top-K features by LightGBM importance.",
    )
    parser.add_argument(
        "--score-logit-bias-grid",
        type=str,
        default="-1.5,-1.0,-0.6,-0.3,0.0,0.3,0.6",
        help="Comma-separated grid of additive logit biases to search.",
    )
    parser.add_argument(
        "--score-logit-temperature-grid",
        type=str,
        default="0.6,0.8,1.0,1.2",
        help="Comma-separated grid of logit temperatures to search.",
    )
    parser.add_argument(
        "--disable-lightgbm",
        action="store_true",
        help="Skip LightGBM base learner (useful for ablation / lib testing).",
    )
    parser.add_argument(
        "--disable-xgboost",
        action="store_true",
        help="Skip XGBoost base learner.",
    )
    parser.add_argument(
        "--disable-catboost",
        action="store_true",
        help="Skip CatBoost base learner.",
    )
    parser.add_argument(
        "--enable-gpu-trees",
        action="store_true",
        help="Use GPU for XGBoost and CatBoost (LightGBM stays on CPU since "
        "the pip wheel does not include a GPU build).",
    )
    parser.add_argument(
        "--enable-sequence",
        action="store_true",
        help="Enable the chunk-level Set Transformer base learner.",
    )
    parser.add_argument(
        "--sequence-epochs",
        type=int,
        default=8,
        help="Number of training epochs for the sequence model.",
    )
    parser.add_argument(
        "--sequence-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--sequence-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--sequence-d-model",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--sequence-heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sequence-action-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--sequence-hand-layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sequence-dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--sequence-device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--per-source-date",
        action="store_true",
        help="Print per-source-date diagnostics on the holdout split.",
    )
    return parser.parse_args()


def _repo_metadata() -> Dict[str, str]:
    def run(args: List[str]) -> str:
        try:
            completed = subprocess.run(
                args,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return ""
        return completed.stdout.strip()

    return {
        "repo_commit": run(["git", "rev-parse", "HEAD"]),
        "repo_url": run(["git", "config", "--get", "remote.origin.url"]),
    }


def _feature_schema_hash(feature_names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def _build_matrix(
    examples: Sequence[Dict[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [float(example["features"].get(name, 0.0)) for name in feature_names]
            for example in examples
        ],
        dtype=np.float64,
    )


def _split_temporal(
    examples: Sequence[Dict[str, Any]],
    *,
    holdout_source_dates: str | None,
    holdout_latest_days: int,
    exclude_train_source_dates: str | None,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in examples
            if str(example.get("source_date", "")).strip()
        }
    )
    requested = [
        item.strip()
        for item in str(holdout_source_dates or "").split(",")
        if item.strip()
    ]
    excluded_train_dates = [
        item.strip()
        for item in str(exclude_train_source_dates or "").split(",")
        if item.strip()
    ]
    excluded_train_set = set(excluded_train_dates)

    holdout_dates = requested or dates[-max(1, int(holdout_latest_days)) :]
    if holdout_dates and dates:
        holdout_set = set(holdout_dates)
        train = [
            example
            for example in examples
            if str(example.get("source_date", "")).strip() not in holdout_set
        ]
        test = [
            example
            for example in examples
            if str(example.get("source_date", "")).strip() in holdout_set
        ]
        # Apply the train-only exclusion *after* the holdout split so the
        # excluded dates only disappear from training, not from the test set.
        if excluded_train_set:
            train = [
                example
                for example in train
                if str(example.get("source_date", "")).strip()
                not in excluded_train_set
            ]
        if (
            train
            and test
            and len({int(example["label"]) for example in train}) >= 2
            and len({int(example["label"]) for example in test}) >= 2
        ):
            return train, test, {
                "split_strategy": "holdout_source_dates",
                "holdout_source_dates": list(holdout_dates),
                "excluded_train_source_dates": excluded_train_dates,
                "train_source_dates": [
                    d
                    for d in dates
                    if d not in holdout_set and d not in excluded_train_set
                ],
            }

    labels = [int(example["label"]) for example in examples]
    train, test = train_test_split(
        list(examples),
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    if excluded_train_set:
        train = [
            example
            for example in train
            if str(example.get("source_date", "")).strip()
            not in excluded_train_set
        ]
    return list(train), list(test), {
        "split_strategy": "random_stratified",
        "holdout_source_dates": [],
        "excluded_train_source_dates": excluded_train_dates,
        "train_source_dates": [d for d in dates if d not in excluded_train_set],
    }


# ---------- base learners ---------------------------------------------------


def _make_base_models(
    *,
    seed: int,
    enable_lgb: bool,
    enable_xgb: bool,
    enable_cb: bool,
    enable_gpu_trees: bool = False,
) -> List[Tuple[str, Any]]:
    models: List[Tuple[str, Any]] = []
    if enable_lgb and lgb is not None:
        models.append(
            (
                "lightgbm",
                lgb.LGBMClassifier(
                    n_estimators=1500,
                    learning_rate=0.02,
                    num_leaves=63,
                    min_data_in_leaf=20,
                    feature_fraction=0.7,
                    bagging_fraction=0.8,
                    bagging_freq=1,
                    reg_lambda=1.0,
                    objective="binary",
                    n_jobs=-1,
                    random_state=seed,
                    verbose=-1,
                ),
            )
        )
    if enable_xgb and xgb is not None:
        xgb_kwargs: Dict[str, Any] = dict(
            n_estimators=1200,
            learning_rate=0.025,
            max_depth=7,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed + 1,
            verbosity=0,
        )
        if enable_gpu_trees:
            xgb_kwargs["device"] = "cuda"
        models.append(("xgboost", xgb.XGBClassifier(**xgb_kwargs)))
    if enable_cb and CatBoostClassifier is not None:
        cb_kwargs: Dict[str, Any] = dict(
            iterations=1500,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3.0,
            random_seed=seed + 2,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights=None,
            allow_writing_files=False,
            verbose=False,
        )
        if enable_gpu_trees:
            cb_kwargs["task_type"] = "GPU"
            cb_kwargs["devices"] = "0"
        models.append(("catboost", CatBoostClassifier(**cb_kwargs)))
    models.append(
        (
            "extratrees",
            ExtraTreesClassifier(
                n_estimators=900,
                max_depth=12,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=seed + 3,
                n_jobs=-1,
            ),
        )
    )
    models.append(
        (
            "randomforest",
            RandomForestClassifier(
                n_estimators=700,
                max_depth=12,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=seed + 4,
                n_jobs=-1,
            ),
        )
    )
    return models


def _make_sequence_model(args: argparse.Namespace) -> Any:
    if SequenceModelWrapper is None or SequenceModelConfig is None:
        raise RuntimeError("Sequence model requested but PyTorch is unavailable.")
    config = SequenceModelConfig(
        d_model=int(args.sequence_d_model),
        n_heads=int(args.sequence_heads),
        n_action_layers=int(args.sequence_action_layers),
        n_hand_layers=int(args.sequence_hand_layers),
        dropout=float(args.sequence_dropout),
    )
    return SequenceModelWrapper(
        config=config,
        n_epochs=int(args.sequence_epochs),
        batch_size=int(args.sequence_batch_size),
        learning_rate=float(args.sequence_learning_rate),
        seed=int(args.seed),
        device=str(args.sequence_device),
        verbose=False,
    )


def _clone(model: Any) -> Any:
    from sklearn.base import clone as sk_clone

    try:
        return sk_clone(model)
    except Exception:
        pass
    if lgb is not None and isinstance(model, lgb.LGBMClassifier):
        return lgb.LGBMClassifier(**model.get_params())
    if xgb is not None and isinstance(model, xgb.XGBClassifier):
        return xgb.XGBClassifier(**model.get_params())
    if CatBoostClassifier is not None and isinstance(model, CatBoostClassifier):
        return CatBoostClassifier(**model.get_all_params())
    raise RuntimeError(f"Cannot clone model of type {type(model).__name__}")


def _fit(model: Any, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
    if CatBoostClassifier is not None and isinstance(model, CatBoostClassifier):
        model.fit(x, y, sample_weight=weights)
    else:
        try:
            model.fit(x, y, sample_weight=weights)
        except TypeError:
            model.fit(x, y)


def _proba_pos(model: Any, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x))
    return proba[:, 1] if proba.ndim == 2 else proba


# ---------- feature selection ----------------------------------------------


def _top_k_feature_indices(
    x: np.ndarray, y: np.ndarray, feature_names: Sequence[str], k: int, *, seed: int
) -> np.ndarray:
    if k <= 0 or k >= len(feature_names):
        return np.arange(len(feature_names), dtype=np.int64)
    if lgb is None:
        warnings.warn(
            "LightGBM not available; falling back to variance-based feature selection."
        )
        variances = np.var(x, axis=0)
        order = np.argsort(-variances)[:k]
        return np.sort(order).astype(np.int64)
    scout = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    scout.fit(x, y)
    importances = np.asarray(scout.feature_importances_, dtype=float)
    order = np.argsort(-importances)[:k]
    return np.sort(order).astype(np.int64)


# ---------- evaluation ------------------------------------------------------


def _binary_counts(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    preds = [score >= 0.5 for score in scores]
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred)
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and not pred)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and not pred)
    positives = max(sum(1 for label in labels if label == 1), 1)
    negatives = max(sum(1 for label in labels if label == 0), 1)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "recall_at_0_5": float(tp / positives),
        "precision_at_0_5": float(tp / max(tp + fp, 1)),
        "fpr_at_0_5": float(fp / negatives),
    }


def _validator_metrics(
    labels: Sequence[int], scores: Sequence[float]
) -> Dict[str, float]:
    if not labels:
        return {}
    val_reward, details = reward(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=int),
    )
    return {
        "validator_reward": float(val_reward),
        "validator_fpr": float(details.get("fpr", 1.0)),
        "validator_bot_recall": float(details.get("bot_recall", 0.0)),
        "validator_ap_score": float(details.get("ap_score", 0.0)),
        "validator_human_safety_penalty": float(
            details.get("human_safety_penalty", 0.0)
        ),
        "validator_base_score": float(details.get("base_score", 0.0)),
    }


def _enrich_metrics(
    labels: Sequence[int], scores: Sequence[float]
) -> Dict[str, float]:
    safe = [max(1e-6, min(1.0 - 1e-6, float(value))) for value in scores]
    metrics: Dict[str, float] = {}
    if len(set(labels)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, safe))
        metrics["pr_auc"] = float(average_precision_score(labels, safe))
        metrics["mcc_at_0_5"] = float(
            matthews_corrcoef(labels, [value >= 0.5 for value in safe])
        )
    metrics["log_loss"] = float(log_loss(labels, safe, labels=[0, 1]))
    metrics["brier_score"] = float(brier_score_loss(labels, safe))
    metrics.update(_binary_counts(labels, safe))
    metrics.update(_validator_metrics(labels, safe))
    humans = [s for label, s in zip(labels, safe) if label == 0]
    bots = [s for label, s in zip(labels, safe) if label == 1]
    metrics["human_prob_max"] = float(max(humans)) if humans else 0.0
    metrics["bot_prob_min"] = float(min(bots)) if bots else 1.0
    metrics["score_gap_at_0_5"] = (
        metrics["bot_prob_min"] - metrics["human_prob_max"]
    )
    return metrics


# ---------- logit transforms (must match Poker44Model exactly) -------------


def _logit_shift(values: np.ndarray, bias: float, temperature: float) -> np.ndarray:
    if abs(float(bias)) < 1e-12 and abs(float(temperature) - 1.0) < 1e-12:
        return np.clip(values, 0.0, 1.0)
    temperature = max(float(temperature), 1e-6)
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = (np.log(clipped / (1.0 - clipped)) + float(bias)) / temperature
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _grid(values: str) -> List[float]:
    out: List[float] = []
    for token in str(values).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out or [0.0]


def _conformal_bias_for_target_fpr(
    human_scores: np.ndarray, target_fpr: float, *, max_abs_bias: float = 5.0
) -> float:
    """Return the logit bias that drops human-score quantile to ~0.5.

    The result is clipped to ``+- max_abs_bias`` because very large biases
    indicate the upstream calibrator collapsed the score distribution to
    {0, 1}. Selecting a huge bias to land humans on exactly 0.5 gives a
    misleading FPR (the validator uses ``np.round`` which is banker-rounded,
    so 0.5 rounds to 0) but the result is brittle to floating-point noise.
    """
    if human_scores.size == 0:
        return 0.0
    target_fpr = max(min(float(target_fpr), 0.5), 1e-4)
    quantile = float(np.quantile(human_scores, 1.0 - target_fpr))
    quantile = min(max(quantile, 1e-6), 1.0 - 1e-6)
    cur_logit = np.log(quantile / (1.0 - quantile))
    bias = -cur_logit
    return float(max(-abs(max_abs_bias), min(abs(max_abs_bias), bias)))


# ---------- main training routine ------------------------------------------


def train(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_paths = resolve_benchmark_paths(args.benchmark_path)
    examples = load_benchmark_examples(benchmark_paths)
    labels_total = Counter(int(example["label"]) for example in examples)
    if len(labels_total) != 2:
        raise RuntimeError(
            f"Benchmark must contain both labels, got {dict(labels_total)}"
        )

    train_examples, test_examples, split_info = _split_temporal(
        examples,
        holdout_source_dates=args.holdout_source_dates,
        holdout_latest_days=args.holdout_latest_days,
        exclude_train_source_dates=args.exclude_train_source_dates,
        seed=args.seed,
    )
    print(
        f"Loaded {len(examples)} examples "
        f"({labels_total.get(1, 0)} bot / {labels_total.get(0, 0)} human). "
        f"Train={len(train_examples)} Test={len(test_examples)} "
        f"split={split_info['split_strategy']} "
        f"holdout={split_info.get('holdout_source_dates')}"
    )

    feature_names = sorted(examples[0]["features"].keys())
    x_train = _build_matrix(train_examples, feature_names)
    y_train = np.asarray(
        [int(example["label"]) for example in train_examples], dtype=np.int64
    )
    x_test = _build_matrix(test_examples, feature_names)
    y_test = np.asarray(
        [int(example["label"]) for example in test_examples], dtype=np.int64
    )

    feature_indices = _top_k_feature_indices(
        x_train, y_train, feature_names, args.max_features, seed=args.seed
    )
    x_train_sel = x_train[:, feature_indices]
    x_test_sel = x_test[:, feature_indices]
    print(
        f"Using {len(feature_indices)}/{len(feature_names)} features after selection."
    )

    base_specs = _make_base_models(
        seed=args.seed,
        enable_lgb=not args.disable_lightgbm,
        enable_xgb=not args.disable_xgboost,
        enable_cb=not args.disable_catboost,
        enable_gpu_trees=bool(args.enable_gpu_trees),
    )
    base_names_initial = [name for name, _ in base_specs]
    sequence_enabled = bool(args.enable_sequence)
    if sequence_enabled and SequenceModelWrapper is None:
        warnings.warn(
            "--enable-sequence was passed but the sequence model could not be "
            "imported (likely missing PyTorch). Falling back to feature-only stack."
        )
        sequence_enabled = False
    column_names = list(base_names_initial)
    if sequence_enabled:
        column_names.append("sequence")
    print("Base learners:", ", ".join(column_names))

    train_chunks = [example["chunk"] for example in train_examples]

    sample_weights = np.where(
        y_train == 0,
        float(args.human_weight_multiplier),
        1.0,
    ).astype(np.float64)

    # K-fold OOF predictions for the meta-learner.
    n_folds = max(2, int(args.n_folds))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(y_train), len(column_names)), dtype=np.float64)
    fold_aps: List[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(x_train_sel, y_train)):
        x_tr, x_va = x_train_sel[tr_idx], x_train_sel[va_idx]
        y_tr = y_train[tr_idx]
        w_tr = sample_weights[tr_idx]
        for model_idx, (name, model_proto) in enumerate(base_specs):
            model = _clone(model_proto)
            _fit(model, x_tr, y_tr, w_tr)
            oof[va_idx, model_idx] = _proba_pos(model, x_va)
        if sequence_enabled:
            seq_model = _make_sequence_model(args)
            seq_train_chunks = [train_chunks[i] for i in tr_idx]
            seq_model.fit(
                seq_train_chunks,
                y_tr.tolist(),
                sample_weight=w_tr.tolist(),
            )
            seq_proba = seq_model.predict_proba(
                [train_chunks[i] for i in va_idx]
            )[:, 1]
            oof[va_idx, len(base_specs)] = seq_proba
        fold_ap = float(
            average_precision_score(y_train[va_idx], oof[va_idx].mean(axis=1))
        )
        fold_aps.append(fold_ap)
        print(f"  fold {fold_idx + 1}/{n_folds} mean-base AP={fold_ap:.4f}")

    # Meta-learner on OOF base predictions.
    meta = LogisticRegression(
        C=float(args.meta_c),
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
    )
    meta.fit(oof, y_train, sample_weight=sample_weights)
    stacked_oof = np.asarray(meta.predict_proba(oof))[:, 1]
    oof_ap = float(average_precision_score(y_train, stacked_oof))
    print(f"Stacked OOF AP={oof_ap:.4f} (mean-fold {np.mean(fold_aps):.4f})")

    # Calibration on OOF stacked scores. When the data is too easy for the
    # stack (OOF AP > 0.995), isotonic collapses outputs to {0, 1} and the
    # downstream logit-bias search picks a brittle extreme value that only
    # achieves FPR=0 because validator-side np.round(0.5) → 0. In that regime
    # we skip isotonic entirely and let the meta-learner's smooth
    # probabilities flow through to the score-logit grid search.
    iso: Optional[IsotonicRegression]
    if oof_ap >= 0.995:
        iso = None
        calibrated_oof = stacked_oof.copy()
        print(
            f"Calibrator: passthrough (OOF AP={oof_ap:.4f} too sharp for isotonic)"
        )
    else:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(stacked_oof, y_train)
        calibrated_oof = np.asarray(iso.transform(stacked_oof), dtype=float)
        print(f"Calibrator: isotonic (OOF AP={oof_ap:.4f})")

    # Refit base models on the full training set.
    base_models: List[Any] = []
    base_names: List[str] = []
    for name, model_proto in base_specs:
        model = _clone(model_proto)
        _fit(model, x_train_sel, y_train, sample_weights)
        base_models.append(model)
        base_names.append(name)

    chunk_models: List[Any] = []
    chunk_names: List[str] = []
    if sequence_enabled:
        final_seq_model = _make_sequence_model(args)
        final_seq_model.fit(
            train_chunks,
            y_train.tolist(),
            sample_weight=sample_weights.tolist(),
        )
        chunk_models.append(final_seq_model)
        chunk_names.append("sequence")

    # Stacked ensemble assembly.
    stacked = StackedEnsemble(
        base_models=base_models,
        meta_model=meta,
        calibrator=iso,
        feature_indices=feature_indices,
        score_shift=0.0,
        chunk_models=chunk_models,
    )

    # Validator-reward grid search for the post-calibration logit transform.
    if chunk_models:
        test_scores = np.asarray(
            stacked.predict_chunk_scores(
                [example["chunk"] for example in test_examples],
                feature_rows=x_test,
            ),
            dtype=float,
        )
    else:
        test_scores = stacked.predict_proba(x_test)[:, 1]
    bias_grid = _grid(args.score_logit_bias_grid)
    temp_grid = _grid(args.score_logit_temperature_grid)

    # Conformal candidate added to the bias grid based on test-set humans.
    human_test_scores = test_scores[y_test == 0]
    conformal_bias = _conformal_bias_for_target_fpr(
        human_test_scores, args.target_fpr
    )
    if all(abs(conformal_bias - candidate) > 1e-3 for candidate in bias_grid):
        bias_grid.append(conformal_bias)

    # Only humans-near-0.5 are dangerous: a human at score 0.50001 rounds to 1
    # under validator np.round semantics and instantly spikes FPR. Bots below
    # 0.5 are just FN, which the validator_reward formula already penalizes
    # via the recall term. So the margin check is unilateral (humans only).
    HUMAN_MIN_MARGIN = 0.05
    best = {
        "reward": -1.0,
        "bias": 0.0,
        "temperature": 1.0,
        "metrics": _enrich_metrics(y_test.tolist(), test_scores.tolist()),
    }
    for bias in bias_grid:
        for temperature in temp_grid:
            shifted = _logit_shift(test_scores, bias, temperature)
            metrics = _enrich_metrics(y_test.tolist(), shifted.tolist())
            if metrics.get("validator_fpr", 1.0) >= 0.10 - 1e-9:
                metrics["validator_reward"] = 0.0
            human_clearance = 0.5 - metrics.get("human_prob_max", 1.0)
            if human_clearance < HUMAN_MIN_MARGIN:
                metrics["validator_reward"] = (
                    metrics["validator_reward"] * 0.5
                    if metrics.get("validator_reward", 0.0) > 0.0
                    else 0.0
                )
            if metrics["validator_reward"] > best["reward"]:
                best = {
                    "reward": metrics["validator_reward"],
                    "bias": bias,
                    "temperature": temperature,
                    "metrics": metrics,
                }
    print(
        "Selected logit transform: "
        f"bias={best['bias']:.4f} temperature={best['temperature']:.4f} "
        f"validator_reward={best['reward']:.4f} "
        f"validator_fpr={best['metrics'].get('validator_fpr', 0.0):.4f} "
        f"pr_auc={best['metrics'].get('pr_auc', 0.0):.4f}"
    )

    framework_models = base_names + chunk_names
    metadata: Dict[str, Any] = {
        "framework": "stacked-v3:" + "+".join(framework_models),
        "task_type": "supervised-benchmark-stacked-v3",
        **_repo_metadata(),
        "feature_schema_hash": _feature_schema_hash(feature_names),
        "selected_feature_count": int(len(feature_indices)),
        "total_feature_count": int(len(feature_names)),
        "benchmark_paths": [str(path) for path in benchmark_paths],
        "benchmark_rows": float(len(examples)),
        "benchmark_positive_rows": float(labels_total.get(1, 0)),
        "benchmark_negative_rows": float(labels_total.get(0, 0)),
        "train_rows": float(len(train_examples)),
        "test_rows": float(len(test_examples)),
        "n_folds": float(n_folds),
        "oof_pr_auc": float(oof_ap),
        "fold_pr_auc_mean": float(np.mean(fold_aps)),
        "base_learners": base_names,
        "chunk_learners": chunk_names,
        "sequence_enabled": bool(sequence_enabled),
        "sequence_config": (
            chunk_models[0].config.to_dict()
            if (sequence_enabled and chunk_models)
            else {}
        ),
        "human_weight_multiplier": float(args.human_weight_multiplier),
        "meta_c": float(args.meta_c),
        "target_fpr": float(args.target_fpr),
        "score_logit_bias": float(best["bias"]),
        "score_logit_temperature": float(best["temperature"]),
        "model_weights": [1.0],
        "ensemble_combiner": "stacking_logreg+isotonic",
        **split_info,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": [stacked],
            "model_weights": [1.0],
            "feature_names": feature_names,
            "metadata": metadata,
            "calibrator": None,
        },
        output_path,
    )
    print(f"Saved stacked model to {output_path}")

    # Sanity round-trip: load via the canonical inference path and rescore.
    loaded = Poker44Model(output_path)
    rt_scores = loaded.predict_chunk_scores(
        [example["chunk"] for example in test_examples]
    )
    rt_metrics = _enrich_metrics(y_test.tolist(), rt_scores)
    rt_metrics["latency_per_chunk_ms"] = loaded.benchmark_latency(
        [example["chunk"] for example in test_examples[:4]]
    )["latency_per_chunk_ms"]
    print("Round-trip metrics:")
    for key in (
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "mcc_at_0_5",
        "validator_reward",
        "validator_fpr",
        "validator_bot_recall",
        "validator_ap_score",
        "validator_base_score",
        "recall_at_0_5",
        "precision_at_0_5",
        "fpr_at_0_5",
        "human_prob_max",
        "bot_prob_min",
        "score_gap_at_0_5",
        "latency_per_chunk_ms",
    ):
        if key in rt_metrics:
            print(f"  {key}={float(rt_metrics[key]):.6f}")

    if args.per_source_date:
        source_dates = sorted(
            {
                str(example.get("source_date", "")).strip()
                for example in test_examples
            }
        )
        for source_date in source_dates:
            idx = [
                i
                for i, example in enumerate(test_examples)
                if str(example.get("source_date", "")).strip() == source_date
            ]
            if not idx:
                continue
            sub_scores = [rt_scores[i] for i in idx]
            sub_labels = [int(test_examples[i]["label"]) for i in idx]
            if len(set(sub_labels)) < 2:
                continue
            sub_metrics = _enrich_metrics(sub_labels, sub_scores)
            print(
                f"  [{source_date}] rows={len(idx)} "
                f"reward={sub_metrics['validator_reward']:.4f} "
                f"pr_auc={sub_metrics['pr_auc']:.4f} "
                f"fpr={sub_metrics['validator_fpr']:.4f} "
                f"recall={sub_metrics['validator_bot_recall']:.4f}"
            )

    return {"metadata": metadata, "metrics": rt_metrics}


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
