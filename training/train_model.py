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
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    ExtraTreesClassifier = None
    HistGradientBoostingClassifier = None
    average_precision_score = None
    log_loss = None
    roc_auc_score = None
    train_test_split = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a benchmark-supervised Poker44 model."
    )
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--disable-aux-human", action="store_true")
    parser.add_argument("--aux-human-weight", type=float, default=0.35)
    parser.add_argument("--max-aux-human-chunks", type=int, default=160)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
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


def train_model(args: argparse.Namespace) -> tuple[list[object], list[str], dict[str, float], dict[str, Any]]:
    if joblib is None:
        raise RuntimeError("joblib is required to save the benchmark model.")
    if (
        ExtraTreesClassifier is None
        or HistGradientBoostingClassifier is None
        or average_precision_score is None
        or log_loss is None
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

    benchmark_rows = [example["features"] for example in benchmark_examples]
    benchmark_labels = [int(example["label"]) for example in benchmark_examples]
    benchmark_chunks = [example["chunk"] for example in benchmark_examples]

    feature_names = sorted(benchmark_rows[0].keys())
    X_benchmark = _build_feature_matrix(benchmark_rows, feature_names)

    (
        X_train_base,
        X_test,
        y_train_base,
        y_test,
        _chunks_train,
        chunks_test,
    ) = train_test_split(
        X_benchmark,
        benchmark_labels,
        benchmark_chunks,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=benchmark_labels,
    )

    X_train = list(X_train_base)
    y_train = list(y_train_base)
    sample_weight = [1.0 for _ in X_train]
    aux_human_rows_added = 0

    if not args.disable_aux_human:
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

    probabilities = _average_probabilities(models, X_test)
    clipped = [_clip_prob(value) for value in probabilities]
    metrics = _binary_metrics(y_test, probabilities)
    metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
    metrics["pr_auc"] = average_precision_score(y_test, probabilities)
    metrics["log_loss"] = log_loss(y_test, clipped)
    metrics["prob_min"] = min(probabilities) if probabilities else 0.0
    metrics["prob_max"] = max(probabilities) if probabilities else 0.0
    human_probs = [prob for prob, label in zip(probabilities, y_test) if label == 0]
    bot_probs = [prob for prob, label in zip(probabilities, y_test) if label == 1]
    metrics["human_prob_max"] = max(human_probs) if human_probs else 0.0
    metrics["bot_prob_min"] = min(bot_probs) if bot_probs else 0.0

    metadata = {
        "framework": "extra-trees+hist-gradient-boosting",
        "task_type": "supervised-benchmark",
        **_repo_metadata(),
        "feature_schema_hash": _feature_schema_hash(feature_names),
        "benchmark_paths": [str(path) for path in benchmark_paths],
        "benchmark_file_count": float(len(benchmark_paths)),
        "benchmark_rows": float(len(benchmark_examples)),
        "benchmark_positive_rows": float(sum(benchmark_labels)),
        "benchmark_negative_rows": float(len(benchmark_labels) - sum(benchmark_labels)),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
        "aux_human_rows": float(aux_human_rows_added),
        "aux_human_weight": float(args.aux_human_weight),
        "n_estimators": float(args.n_estimators),
        "max_depth": float(args.max_depth),
        "learning_rate": float(args.learning_rate),
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
    return models, feature_names, metrics, metadata


def main() -> None:
    args = parse_args()
    _, feature_names, metrics, metadata = train_model(args)
    print(f"Saved model to {args.output}")
    print(f"Feature count: {len(feature_names)}")
    print(
        "Selected config: "
        f"framework={metadata.get('framework')} "
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
        "latency_per_chunk_ms",
    ):
        print(f"{key}={float(metrics.get(key, 0.0)):.6f}")


if __name__ == "__main__":
    main()
