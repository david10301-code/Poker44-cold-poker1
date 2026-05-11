from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from poker44_ml.inference import Poker44Model
from training.build_dataset import load_benchmark_examples, resolve_benchmark_paths
from training.train_model import _enrich_probability_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Poker44 model on released benchmark chunks."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(REPO_ROOT / "models" / "poker44_benchmark_supervised.joblib"),
    )
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument(
        "--source-dates",
        type=str,
        default=None,
        help="Optional comma-separated sourceDate filter.",
    )
    parser.add_argument(
        "--per-source-date",
        action="store_true",
        help="Print a compact metric summary for each sourceDate.",
    )
    return parser.parse_args()


def _evaluate_examples(
    model: Poker44Model,
    examples: list[dict[str, Any]],
) -> dict[str, float]:
    chunks = [list(example["chunk"]) for example in examples]
    labels = [int(example["label"]) for example in examples]
    probabilities = model.predict_chunk_scores(chunks)
    return _enrich_probability_metrics(labels, probabilities)


def _filter_examples(
    examples: list[dict[str, Any]],
    requested_dates: str | None,
) -> list[dict[str, Any]]:
    if not requested_dates:
        return examples
    allowed = {item.strip() for item in requested_dates.split(",") if item.strip()}
    return [
        example
        for example in examples
        if str(example.get("source_date", "")).strip() in allowed
    ]


def _print_metric_block(title: str, metrics: dict[str, float], rows: int) -> None:
    print(title)
    print(f"rows={rows}")
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
        "validator_human_safety_penalty",
        "validator_base_score",
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
        "human_clearance_to_0_5",
        "bot_clearance_to_0_5",
        "score_gap_at_0_5",
        "threshold_margin_at_0_5",
        "prob_mean",
    ):
        if key in metrics:
            print(f"{key}={float(metrics[key]):.6f}")


def main() -> None:
    args = parse_args()
    benchmark_paths = resolve_benchmark_paths(args.benchmark_path)
    examples = load_benchmark_examples(benchmark_paths)
    examples = _filter_examples(examples, args.source_dates)
    if not examples:
        raise RuntimeError("No benchmark examples matched the requested filters.")

    label_counts = Counter(int(example["label"]) for example in examples)
    source_dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in examples
            if str(example.get("source_date", "")).strip()
        }
    )

    model = Poker44Model(args.model_path)
    metrics = _evaluate_examples(model, examples)

    print(f"Model path: {args.model_path}")
    print(f"Benchmark files: {len(benchmark_paths)}")
    print(f"Source dates: {source_dates}")
    print(
        f"Label counts: human={label_counts.get(0, 0)} bot={label_counts.get(1, 0)}"
    )
    _print_metric_block("Overall metrics", metrics, len(examples))

    if args.per_source_date:
        for source_date in source_dates:
            date_examples = [
                example
                for example in examples
                if str(example.get("source_date", "")).strip() == source_date
            ]
            if not date_examples:
                continue
            date_metrics = _evaluate_examples(model, date_examples)
            _print_metric_block(f"Per-source-date metrics | {source_date}", date_metrics, len(date_examples))


if __name__ == "__main__":
    main()
