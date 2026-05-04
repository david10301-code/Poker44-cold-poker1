from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from poker44.miner.config import repository_paths
from poker44_ml.inference import Poker44Model
from training.build_dataset import (
    DEFAULT_BOT_PATHS,
    DEFAULT_HUMAN_PATHS,
    build_training_dataframe,
    load_json_or_gz,
    resolve_existing_path,
)
from training.evaluate import evaluate_predictions, format_metrics

try:
    import joblib
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    joblib = None

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        VotingClassifier,
    )
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    CalibratedClassifierCV = None
    ExtraTreesClassifier = None
    HistGradientBoostingClassifier = None
    VotingClassifier = None
    train_test_split = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    XGBClassifier = None


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = repository_paths(REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fast chunk-level Poker44 miner model."
    )
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--bot-path", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument(
        "--calibration",
        choices=("auto", "isotonic", "sigmoid", "none"),
        default="auto",
    )
    parser.add_argument("--recall-target", type=float, default=0.9)
    parser.add_argument(
        "--min-threshold-at-recall",
        type=float,
        default=0.0,
        help="Minimum acceptable threshold_at_recall during model selection.",
    )
    parser.add_argument(
        "--max-fpr-at-threshold-0-5",
        type=float,
        default=0.0,
        help="Constraint used during model selection.",
    )
    parser.add_argument(
        "--max-fpr-at-recall",
        type=float,
        default=0.003,
        help="Constraint used during model selection.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Evaluate multiple candidate configurations and save the best one.",
    )
    parser.add_argument(
        "--search-budget",
        type=int,
        default=6,
        help="Maximum number of base parameter configurations to evaluate in search mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PATHS.default_model_path),
    )
    return parser.parse_args()


def choose_calibration_options(method: str, train_size: int) -> list[str | None]:
    if method == "none":
        return [None]
    if method == "sigmoid":
        return ["sigmoid"]
    if method == "isotonic":
        return ["isotonic"] if train_size >= 800 else ["sigmoid"]
    options: list[str | None] = ["sigmoid", None]
    if train_size >= 800:
        options.insert(1, "isotonic")
    return options


def build_base_model(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    seed: int,
) -> tuple[object, str]:
    if XGBClassifier is not None:
        booster = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
        forest = ExtraTreesClassifier(
            n_estimators=max(200, n_estimators),
            max_depth=max_depth + 2,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        return (
            VotingClassifier(
                estimators=[("xgb", booster), ("et", forest)],
                voting="soft",
                weights=[2, 1],
            ),
            "xgboost+extra-trees+sklearn-calibration",
        )

    booster = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=n_estimators,
        random_state=seed,
    )
    forest = ExtraTreesClassifier(
        n_estimators=max(300, n_estimators),
        max_depth=max_depth + 2,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    return (
        VotingClassifier(
            estimators=[("hgb", booster), ("et", forest)],
            voting="soft",
            weights=[2, 1],
        ),
        "sklearn-hist-gradient-boosting+extra-trees+calibration",
    )


def build_search_configs(args: argparse.Namespace) -> list[dict[str, float | int]]:
    base = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    candidates = [
        base,
        {
            **base,
            "n_estimators": max(args.n_estimators, 500),
            "max_depth": min(args.max_depth, 4),
            "learning_rate": min(args.learning_rate, 0.03),
            "subsample": max(args.subsample, 0.95),
            "colsample_bytree": max(args.colsample_bytree, 0.95),
        },
        {
            **base,
            "n_estimators": max(args.n_estimators, 700),
            "max_depth": min(args.max_depth, 4),
            "learning_rate": min(args.learning_rate, 0.02),
            "subsample": max(args.subsample, 0.95),
            "colsample_bytree": max(args.colsample_bytree, 0.95),
        },
        {
            **base,
            "n_estimators": max(args.n_estimators, 600),
            "max_depth": 5,
            "learning_rate": min(args.learning_rate, 0.03),
        },
        {
            **base,
            "n_estimators": max(args.n_estimators, 900),
            "max_depth": 3,
            "learning_rate": min(args.learning_rate, 0.02),
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        {
            **base,
            "n_estimators": max(args.n_estimators, 800),
            "max_depth": 4,
            "learning_rate": min(args.learning_rate, 0.025),
            "subsample": max(args.subsample, 0.95),
            "colsample_bytree": max(args.colsample_bytree, 0.95),
        },
    ]
    return candidates[: max(1, args.search_budget)]


def constraint_status(
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> tuple[bool, float]:
    violation = 0.0
    if metrics["threshold_at_recall"] < args.min_threshold_at_recall:
        violation += (
            args.min_threshold_at_recall - metrics["threshold_at_recall"]
        ) * 10.0
    if metrics["fpr_at_threshold_0_5"] > args.max_fpr_at_threshold_0_5:
        violation += (
            metrics["fpr_at_threshold_0_5"] - args.max_fpr_at_threshold_0_5
        ) * 1000.0
    if metrics["fpr_at_recall"] > args.max_fpr_at_recall:
        violation += metrics["fpr_at_recall"] - args.max_fpr_at_recall
    return violation == 0.0, violation


def candidate_rank(
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> tuple[float, ...]:
    satisfies_constraints, violation = constraint_status(metrics, args)
    if satisfies_constraints:
        return (
            1.0,
            metrics["achieved_recall"],
            -metrics["fpr_at_recall"],
            -metrics["fpr_at_threshold_0_5"],
            metrics["pr_auc"],
            metrics["roc_auc"],
            -metrics["log_loss"],
        )
    return (
        0.0,
        -violation,
        -metrics["fpr_at_threshold_0_5"],
        -metrics["fpr_at_recall"],
        metrics["achieved_recall"],
        metrics["pr_auc"],
        metrics["roc_auc"],
    )


def fit_candidate(
    *,
    config: dict[str, float | int],
    calibration_method: str | None,
    X_train: list[list[float]],
    X_test: list[list[float]],
    y_train: list[int],
    y_test: list[int],
    args: argparse.Namespace,
) -> tuple[object, dict[str, float], dict[str, Any]]:
    base_model, framework_name = build_base_model(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        seed=args.seed,
    )

    model = base_model
    if calibration_method is not None:
        min_class_count = min(Counter(y_train).values())
        cv = min(3, min_class_count)
        if cv >= 2:
            model = CalibratedClassifierCV(
                base_model,
                method=calibration_method,
                cv=cv,
            )

    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)

    metrics = evaluate_predictions(
        y_true=y_test,
        y_prob=probs,
        recall_target=args.recall_target,
    )
    metadata = {
        **config,
        "framework": framework_name,
        "calibration": calibration_method or "none",
    }
    return model, metrics, metadata


def train_model(args: argparse.Namespace) -> tuple[object, list[str], dict[str, float], dict[str, Any]]:
    if joblib is None:
        raise RuntimeError("Training dependencies are missing. Install joblib first.")
    if any(
        dependency is None
        for dependency in (
            CalibratedClassifierCV,
            ExtraTreesClassifier,
            HistGradientBoostingClassifier,
            VotingClassifier,
            train_test_split,
        )
    ):
        raise RuntimeError("scikit-learn is required to train and calibrate the miner model.")

    human_path = resolve_existing_path(args.human_path, DEFAULT_HUMAN_PATHS)
    bot_path = resolve_existing_path(args.bot_path, DEFAULT_BOT_PATHS)
    human_hands = load_json_or_gz(human_path)
    bot_hands = load_json_or_gz(bot_path)

    rows = build_training_dataframe(
        human_hands=human_hands,
        bot_hands=bot_hands,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError(
            "Training dataframe is empty. Verify your human/bot hand sources."
        )

    feature_names = sorted(key for key in rows[0].keys() if key != "label")
    X = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    y = [int(row["label"]) for row in rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    configs = build_search_configs(args) if args.search else [
        {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }
    ]
    calibration_options = choose_calibration_options(args.calibration, len(X_train))

    best_model = None
    best_metrics: dict[str, float] | None = None
    best_metadata: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None

    candidate_index = 0
    for config in configs:
        for calibration_method in calibration_options:
            candidate_index += 1
            model, metrics, metadata = fit_candidate(
                config=config,
                calibration_method=calibration_method,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                args=args,
            )
            rank = candidate_rank(metrics, args)
            satisfies_constraints, violation = constraint_status(metrics, args)
            print(
                f"candidate={candidate_index} "
                f"constraints_ok={int(satisfies_constraints)} "
                f"constraint_violation={violation:.6f} "
                f"config={metadata} "
                f"{format_metrics(metrics)}"
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_model = model
                best_metrics = metrics
                best_metadata = metadata

    if best_model is None or best_metrics is None or best_metadata is None:
        raise RuntimeError("No model candidate was trained.")

    artifact_meta = {
        "chunk_size": float(args.chunk_size),
        "min_chunk_size": float(args.min_chunk_size),
        "human_path": str(human_path),
        "bot_path": str(bot_path),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
        "recall_target": float(args.recall_target),
        "min_threshold_at_recall": float(args.min_threshold_at_recall),
        "max_fpr_at_threshold_0_5": float(args.max_fpr_at_threshold_0_5),
        "max_fpr_at_recall": float(args.max_fpr_at_recall),
        **{
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in best_metadata.items()
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_names": feature_names,
            "metadata": artifact_meta,
        },
        output_path,
    )

    loaded = Poker44Model(output_path)
    latency = loaded.benchmark_latency(
        [human_hands[: args.chunk_size], bot_hands[: args.chunk_size]]
    )
    best_metrics["latency_per_chunk_ms"] = latency["latency_per_chunk_ms"]
    return best_model, feature_names, best_metrics, artifact_meta


def main() -> None:
    args = parse_args()
    _, feature_names, metrics, metadata = train_model(args)
    print(f"Saved model to {args.output}")
    print(f"Feature count: {len(feature_names)}")
    print(
        "Selected config: "
        f"framework={metadata.get('framework')} "
        f"calibration={metadata.get('calibration')} "
        f"n_estimators={metadata.get('n_estimators')} "
        f"max_depth={metadata.get('max_depth')} "
        f"learning_rate={metadata.get('learning_rate')}"
    )
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
