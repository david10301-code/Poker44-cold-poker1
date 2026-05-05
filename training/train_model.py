from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from poker44_ml.inference import HumanBaselineModel
from training.build_dataset import (
    build_human_chunk_rows,
    load_json_or_gz,
    resolve_human_path,
)

try:
    import joblib
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    joblib = None

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    IsolationForest = None
    train_test_split = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a human-only Poker44 baseline model."
    )
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-samples", type=float, default=0.75)
    parser.add_argument("--max-features", type=float, default=0.9)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "models" / "poker44_human_baseline.joblib"),
    )
    return parser.parse_args()


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = max(0.0, min(1.0, q))
    index = q * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def validation_metrics(risks: list[float]) -> dict[str, float]:
    sorted_risks = sorted(float(value) for value in risks)
    false_positive_at_0_5 = sum(1 for value in sorted_risks if value >= 0.5) / max(len(sorted_risks), 1)
    false_positive_at_0_75 = sum(1 for value in sorted_risks if value >= 0.75) / max(len(sorted_risks), 1)
    return {
        "human_fpr_at_0_5": false_positive_at_0_5,
        "human_fpr_at_0_75": false_positive_at_0_75,
        "risk_mean": sum(sorted_risks) / max(len(sorted_risks), 1),
        "risk_q50": quantile(sorted_risks, 0.50),
        "risk_q90": quantile(sorted_risks, 0.90),
        "risk_q95": quantile(sorted_risks, 0.95),
        "risk_q99": quantile(sorted_risks, 0.99),
        "risk_max": max(sorted_risks) if sorted_risks else 0.0,
    }


def train_model(args: argparse.Namespace) -> tuple[object, list[str], dict[str, float], dict[str, Any]]:
    if joblib is None:
        raise RuntimeError("joblib is required to save the human baseline model.")
    if IsolationForest is None or train_test_split is None:
        raise RuntimeError("scikit-learn is required to train the human baseline model.")

    human_path = resolve_human_path(args.human_path)
    human_hands = load_json_or_gz(human_path)
    rows = build_human_chunk_rows(
        human_hands=human_hands,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        stride=args.stride,
        repeats=args.repeats,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError("No human chunks were generated from the corpus.")

    feature_names = sorted(rows[0].keys())
    X = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    X_train, X_test = train_test_split(
        X,
        test_size=args.test_size,
        random_state=args.seed,
    )

    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        contamination="auto",
        random_state=args.seed,
        n_jobs=1,
    )
    model.fit(X_train)

    training_anomaly_scores = sorted(-float(value) for value in model.score_samples(X_train))
    metadata = {
        "framework": "isolation-forest-human-baseline",
        "human_path": str(human_path),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
        "chunk_size": float(args.chunk_size),
        "min_chunk_size": float(args.min_chunk_size),
        "stride": float(args.stride),
        "repeats": float(args.repeats),
        "n_estimators": float(args.n_estimators),
        "max_samples": float(args.max_samples),
        "max_features": float(args.max_features),
        "bootstrap": bool(args.bootstrap),
        "score_quantiles": {
            "q50": quantile(training_anomaly_scores, 0.50),
            "q90": quantile(training_anomaly_scores, 0.90),
            "q95": quantile(training_anomaly_scores, 0.95),
            "q99": quantile(training_anomaly_scores, 0.99),
            "q995": quantile(training_anomaly_scores, 0.995),
            "q999": quantile(training_anomaly_scores, 0.999),
            "q9995": quantile(training_anomaly_scores, 0.9995),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "metadata": metadata,
        },
        output_path,
    )

    loaded = HumanBaselineModel(output_path)
    validation_risks = [-float(value) for value in loaded.model.score_samples(X_test)]
    validation_human_risks = [
        loaded._risk_from_anomaly(anomaly_score) for anomaly_score in validation_risks
    ]
    metrics = validation_metrics(validation_human_risks)
    latency = loaded.benchmark_latency(
        [human_hands[: args.chunk_size], human_hands[args.chunk_size : args.chunk_size * 2]]
    )
    metrics["latency_per_chunk_ms"] = latency["latency_per_chunk_ms"]
    metrics["raw_validation_score_mean"] = (
        sum(validation_risks) / max(len(validation_risks), 1)
    )
    return model, feature_names, metrics, metadata


def main() -> None:
    args = parse_args()
    _, feature_names, metrics, metadata = train_model(args)
    print(f"Saved model to {args.output}")
    print(f"Feature count: {len(feature_names)}")
    print(
        "Selected config: "
        f"framework={metadata.get('framework')} "
        f"n_estimators={metadata.get('n_estimators')} "
        f"max_samples={metadata.get('max_samples')} "
        f"max_features={metadata.get('max_features')} "
        f"bootstrap={metadata.get('bootstrap')}"
    )
    for key in (
        "human_fpr_at_0_5",
        "human_fpr_at_0_75",
        "risk_mean",
        "risk_q50",
        "risk_q90",
        "risk_q95",
        "risk_q99",
        "risk_max",
        "latency_per_chunk_ms",
    ):
        print(f"{key}={float(metrics.get(key, 0.0)):.6f}")


if __name__ == "__main__":
    main()
