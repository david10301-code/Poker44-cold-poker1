from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from poker44_ml.inference import Poker44Model
from training.build_dataset import (
    build_human_chunk_rows,
    load_benchmark_examples,
    load_json_or_gz,
    resolve_benchmark_paths,
    resolve_human_path,
)

try:
    import joblib
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    joblib = None

try:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    ExtraTreesClassifier = None
    HistGradientBoostingClassifier = None
    average_precision_score = None
    brier_score_loss = None
    log_loss = None
    matthews_corrcoef = None
    roc_auc_score = None
    train_test_split = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a benchmark-supervised Poker44 model."
    )
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--enable-aux-human", action="store_true")
    parser.add_argument("--aux-human-weight", type=float, default=0.35)
    parser.add_argument("--max-aux-human-chunks", type=int, default=160)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--holdout-source-dates", type=str, default=None)
    parser.add_argument("--holdout-latest-days", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--calibration-size", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "models" / "poker44_benchmark_supervised.joblib"),
    )
    return parser.parse_args()


def _build_feature_matrix(
    rows: list[dict[str, float]],
    feature_names: list[str],
) -> list[list[float]]:
    return [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]


def _clip_prob(value: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def _repo_metadata() -> dict[str, str]:
    return {
        "repo_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "repo_url": _git_output(["git", "config", "--get", "remote.origin.url"]),
    }


def _feature_schema_hash(feature_names: list[str]) -> str:
    joined = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _average_probabilities(models: list[object], rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    per_model_scores: list[list[float]] = []
    for model in models:
        probabilities = model.predict_proba(rows)
        per_model_scores.append([float(row[1]) for row in probabilities])
    averaged: list[float] = []
    for index in range(len(rows)):
        averaged.append(
            sum(scores[index] for scores in per_model_scores) / max(len(per_model_scores), 1)
        )
    return averaged


def _derive_score_remap(
    labels: list[int],
    probabilities: list[float],
) -> dict[str, float]:
    negatives = sorted(
        float(prob) for prob, label in zip(probabilities, labels) if int(label) == 0
    )
    positives = sorted(
        float(prob) for prob, label in zip(probabilities, labels) if int(label) == 1
    )
    if not negatives or not positives:
        return {}

    human_upper = negatives[-1]
    bot_lower_candidates = [value for value in positives if value > human_upper]
    if bot_lower_candidates:
        bot_lower = bot_lower_candidates[0]
        gap = max(bot_lower - human_upper, 1e-6)
        threshold = human_upper + 0.15 * gap
        threshold = min(
            max(threshold, human_upper + 1e-6),
            bot_lower - 1e-6,
        )
    else:
        bot_lower = positives[0]
        threshold = 0.5

    threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
    return {
        "kind": "threshold_centering_v1",
        "threshold": float(threshold),
        "human_upper": float(human_upper),
        "bot_lower": float(bot_lower),
    }


def _apply_score_remap(probabilities: list[float], remap: dict[str, Any] | None) -> list[float]:
    if not probabilities or not remap:
        return [_clamp01(value) for value in probabilities]

    threshold = float(remap.get("threshold", 0.5))
    threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
    adjusted: list[float] = []
    for value in probabilities:
        score = _clamp01(value)
        if score <= threshold:
            mapped = 0.5 * score / threshold
        else:
            mapped = 0.5 + 0.5 * (score - threshold) / (1.0 - threshold)
        adjusted.append(round(_clamp01(mapped), 6))
    return adjusted


def _binary_metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    predictions = [prob >= 0.5 for prob in probabilities]
    tp = sum(1 for truth, pred in zip(labels, predictions) if truth == 1 and pred)
    fp = sum(1 for truth, pred in zip(labels, predictions) if truth == 0 and pred)
    tn = sum(1 for truth, pred in zip(labels, predictions) if truth == 0 and not pred)
    fn = sum(1 for truth, pred in zip(labels, predictions) if truth == 1 and not pred)

    positives = max(sum(1 for value in labels if value == 1), 1)
    negatives = max(sum(1 for value in labels if value == 0), 1)
    predicted_positive = max(tp + fp, 1)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "recall_at_0_5": tp / positives,
        "precision_at_0_5": tp / predicted_positive,
        "fpr_at_0_5": fp / negatives,
    }


def _overfit_risk_warnings(
    *,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    test_rows = int(float(metadata.get("test_rows", 0.0) or 0.0))
    holdout_dates = list(metadata.get("holdout_source_dates") or [])
    benchmark_files = int(float(metadata.get("benchmark_file_count", 0.0) or 0.0))
    perfect_at_threshold = (
        float(metrics.get("recall_at_0_5", 0.0)) >= 0.999999
        and float(metrics.get("precision_at_0_5", 0.0)) >= 0.999999
        and float(metrics.get("fpr_at_0_5", 1.0)) <= 0.000001
    )
    near_perfect_ranking = (
        float(metrics.get("roc_auc", 0.0)) >= 0.999999
        and float(metrics.get("pr_auc", 0.0)) >= 0.999999
    )

    if perfect_at_threshold and near_perfect_ranking and test_rows <= 500:
        warnings.append(
            "Evaluation is effectively perfect on a small holdout. This may be optimistic; validate on more released days."
        )
    if perfect_at_threshold and len(holdout_dates) <= 1:
        warnings.append(
            "Only one held-out sourceDate was used. Add more held-out dates to reduce day-specific overfit risk."
        )
    if perfect_at_threshold and benchmark_files <= 1:
        warnings.append(
            "Training/evaluation used only one benchmark file. Use multiple released benchmark files for a stronger anti-overfit check."
        )
    raw_human_max = float(metrics.get("raw_human_prob_max", 0.0))
    raw_bot_min = float(metrics.get("raw_bot_prob_min", 1.0))
    if perfect_at_threshold and raw_human_max < 0.15 and raw_bot_min > 0.7:
        warnings.append(
            "Raw class separation is extremely large on the holdout. Recheck on unseen release days and live validator traffic."
        )
    return warnings


def _split_benchmark_examples(
    examples: list[dict[str, Any]],
    *,
    test_size: float,
    seed: int,
    holdout_source_dates: str | None,
    holdout_latest_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    available_dates = sorted(
        {str(example.get("source_date", "")).strip() for example in examples if str(example.get("source_date", "")).strip()}
    )

    if holdout_source_dates:
        requested = [
            item.strip()
            for item in holdout_source_dates.split(",")
            if item.strip()
        ]
        holdout_dates = [date for date in requested if date in available_dates]
        if not holdout_dates:
            raise RuntimeError(
                f"Requested holdout dates {requested} not found in available source dates {available_dates}."
            )
    elif len(available_dates) > 1:
        holdout_count = max(1, min(int(holdout_latest_days), len(available_dates) - 1))
        holdout_dates = available_dates[-holdout_count:]
    else:
        holdout_dates = []

    if holdout_dates:
        holdout_set = set(holdout_dates)
        train_examples = [
            example for example in examples
            if str(example.get("source_date", "")).strip() not in holdout_set
        ]
        test_examples = [
            example for example in examples
            if str(example.get("source_date", "")).strip() in holdout_set
        ]
        if not train_examples or not test_examples:
            raise RuntimeError(
                f"Source-date holdout produced an empty split. holdout_dates={holdout_dates}"
            )
        return train_examples, test_examples, {
            "split_strategy": "holdout_source_dates",
            "holdout_source_dates": list(holdout_dates),
            "train_source_dates": [
                date for date in available_dates if date not in holdout_set
            ],
        }

    labels = [int(example["label"]) for example in examples]
    train_examples, test_examples = train_test_split(
        examples,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_examples, test_examples, {
        "split_strategy": "random_chunk_split",
        "holdout_source_dates": [],
        "train_source_dates": list(available_dates),
    }


def train_model(args: argparse.Namespace) -> tuple[list[object], list[str], dict[str, float], dict[str, Any]]:
    if joblib is None:
        raise RuntimeError("joblib is required to save the benchmark model.")
    if (
        ExtraTreesClassifier is None
        or HistGradientBoostingClassifier is None
        or average_precision_score is None
        or brier_score_loss is None
        or log_loss is None
        or matthews_corrcoef is None
        or roc_auc_score is None
        or train_test_split is None
    ):
        raise RuntimeError("scikit-learn is required to train the benchmark model.")

    benchmark_paths = resolve_benchmark_paths(args.benchmark_path)
    benchmark_examples = load_benchmark_examples(benchmark_paths)
    label_counts = Counter(example["label"] for example in benchmark_examples)
    if len(label_counts) < 2:
        raise RuntimeError("Benchmark dataset must contain both human and bot labels.")
    if min(label_counts.values()) < 2:
        raise RuntimeError("Benchmark dataset is too small to create a stratified split.")

    train_examples, test_examples, split_info = _split_benchmark_examples(
        benchmark_examples,
        test_size=args.test_size,
        seed=args.seed,
        holdout_source_dates=args.holdout_source_dates,
        holdout_latest_days=args.holdout_latest_days,
    )
    benchmark_label_sum = sum(int(example["label"]) for example in benchmark_examples)

    benchmark_rows = [example["features"] for example in benchmark_examples]
    feature_names = sorted(benchmark_rows[0].keys())

    X_test = _build_feature_matrix(
        [example["features"] for example in test_examples], feature_names
    )
    y_test = [int(example["label"]) for example in test_examples]
    chunks_test = [example["chunk"] for example in test_examples]

    calibration_examples: list[dict[str, Any]] = []
    fit_examples = list(train_examples)
    fit_labels = [int(example["label"]) for example in fit_examples]
    if (
        len(fit_examples) >= 20
        and len({*fit_labels}) >= 2
        and args.calibration_size > 0.0
    ):
        calib_size = min(max(float(args.calibration_size), 0.05), 0.4)
        fit_examples, calibration_examples = train_test_split(
            fit_examples,
            test_size=calib_size,
            random_state=args.seed,
            stratify=fit_labels,
        )

    X_train_base = _build_feature_matrix(
        [example["features"] for example in fit_examples], feature_names
    )
    y_train_base = [int(example["label"]) for example in fit_examples]
    X_calibration = _build_feature_matrix(
        [example["features"] for example in calibration_examples], feature_names
    )
    y_calibration = [int(example["label"]) for example in calibration_examples]

    X_train = list(X_train_base)
    y_train = list(y_train_base)
    sample_weight = [1.0 for _ in X_train]
    aux_human_rows_added = 0

    if args.enable_aux_human:
        human_path = resolve_human_path(args.human_path)
        human_hands = load_json_or_gz(human_path)
        aux_rows = build_human_chunk_rows(
            human_hands=human_hands,
            chunk_size=args.chunk_size,
            min_chunk_size=args.min_chunk_size,
            stride=args.stride,
            repeats=args.repeats,
            seed=args.seed,
            shuffle=False,
        )
        if args.max_aux_human_chunks > 0 and len(aux_rows) > args.max_aux_human_chunks:
            picker = list(aux_rows)
            rng = __import__("random").Random(args.seed)
            aux_rows = rng.sample(picker, args.max_aux_human_chunks)
        X_aux = _build_feature_matrix(aux_rows, feature_names)
        X_train.extend(X_aux)
        y_train.extend([0 for _ in X_aux])
        sample_weight.extend([float(args.aux_human_weight) for _ in X_aux])
        aux_human_rows_added = len(X_aux)

    extra_trees = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=1,
    )
    hist_gradient = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_iter=args.n_estimators,
        min_samples_leaf=5,
        random_state=args.seed,
    )

    extra_trees.fit(X_train, y_train, sample_weight=sample_weight)
    hist_gradient.fit(X_train, y_train, sample_weight=sample_weight)
    models: list[object] = [extra_trees, hist_gradient]

    calibration_raw_probabilities = _average_probabilities(models, X_calibration)
    score_remap = _derive_score_remap(y_calibration, calibration_raw_probabilities)

    raw_probabilities = _average_probabilities(models, X_test)
    probabilities = _apply_score_remap(raw_probabilities, score_remap)
    clipped = [_clip_prob(value) for value in probabilities]
    metrics = _binary_metrics(y_test, probabilities)
    metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
    metrics["pr_auc"] = average_precision_score(y_test, probabilities)
    metrics["log_loss"] = log_loss(y_test, clipped)
    metrics["brier_score"] = brier_score_loss(y_test, probabilities)
    metrics["mcc_at_0_5"] = matthews_corrcoef(
        y_test, [1 if prob >= 0.5 else 0 for prob in probabilities]
    )
    metrics["prob_min"] = min(probabilities) if probabilities else 0.0
    metrics["prob_max"] = max(probabilities) if probabilities else 0.0
    human_probs = [prob for prob, label in zip(probabilities, y_test) if label == 0]
    bot_probs = [prob for prob, label in zip(probabilities, y_test) if label == 1]
    metrics["human_prob_max"] = max(human_probs) if human_probs else 0.0
    metrics["bot_prob_min"] = min(bot_probs) if bot_probs else 0.0
    raw_human_probs = [
        prob for prob, label in zip(raw_probabilities, y_test) if label == 0
    ]
    raw_bot_probs = [
        prob for prob, label in zip(raw_probabilities, y_test) if label == 1
    ]
    metrics["raw_human_prob_max"] = max(raw_human_probs) if raw_human_probs else 0.0
    metrics["raw_bot_prob_min"] = min(raw_bot_probs) if raw_bot_probs else 0.0

    metadata = {
        "framework": "sklearn.ExtraTreesClassifier+sklearn.HistGradientBoostingClassifier",
        "task_type": "supervised-benchmark",
        **_repo_metadata(),
        "feature_schema_hash": _feature_schema_hash(feature_names),
        "score_remap": score_remap,
        **split_info,
        "benchmark_paths": [str(path) for path in benchmark_paths],
        "benchmark_file_count": float(len(benchmark_paths)),
        "benchmark_rows": float(len(benchmark_examples)),
        "benchmark_positive_rows": float(benchmark_label_sum),
        "benchmark_negative_rows": float(len(benchmark_examples) - benchmark_label_sum),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
        "calibration_rows": float(len(X_calibration)),
        "aux_human_rows": float(aux_human_rows_added),
        "aux_human_weight": float(args.aux_human_weight),
        "n_estimators": float(args.n_estimators),
        "max_depth": float(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "calibration_size": float(args.calibration_size),
        "chunk_size": float(args.chunk_size),
        "min_chunk_size": float(args.min_chunk_size),
        "stride": float(args.stride),
        "repeats": float(args.repeats),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "feature_names": feature_names,
            "metadata": metadata,
        },
        output_path,
    )

    loaded = Poker44Model(output_path)
    latency_chunks = list(chunks_test[: min(4, len(chunks_test))])
    latency = loaded.benchmark_latency(latency_chunks or benchmark_chunks[:2])
    metrics["latency_per_chunk_ms"] = latency["latency_per_chunk_ms"]
    metrics["prob_mean"] = sum(probabilities) / max(len(probabilities), 1)
    metadata["overfit_warnings"] = _overfit_risk_warnings(
        metrics=metrics,
        metadata=metadata,
    )
    return models, feature_names, metrics, metadata


def main() -> None:
    args = parse_args()
    _, feature_names, metrics, metadata = train_model(args)
    print(f"Saved model to {args.output}")
    print(f"Feature count: {len(feature_names)}")
    print(
        "Selected config: "
        f"framework={metadata.get('framework')} "
        f"split_strategy={metadata.get('split_strategy')} "
        f"holdout_dates={metadata.get('holdout_source_dates')} "
        f"benchmark_files={metadata.get('benchmark_file_count')} "
        f"benchmark_rows={metadata.get('benchmark_rows')} "
        f"aux_human_rows={metadata.get('aux_human_rows')} "
        f"n_estimators={metadata.get('n_estimators')} "
        f"max_depth={metadata.get('max_depth')} "
        f"learning_rate={metadata.get('learning_rate')}"
    )
    for key in (
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "mcc_at_0_5",
        "recall_at_0_5",
        "precision_at_0_5",
        "fpr_at_0_5",
        "tp",
        "fp",
        "tn",
        "fn",
        "prob_min",
        "prob_max",
        "human_prob_max",
        "bot_prob_min",
        "prob_mean",
        "raw_human_prob_max",
        "raw_bot_prob_min",
        "latency_per_chunk_ms",
    ):
        print(f"{key}={float(metrics.get(key, 0.0)):.6f}")
    for warning in metadata.get("overfit_warnings", []):
        print(f"overfit_warning={warning}")


if __name__ == "__main__":
    main()
