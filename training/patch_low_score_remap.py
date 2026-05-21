"""Patch an existing stacked model with low-score remap for live payloads.

Use when live raw scores sit in ~0.004-0.01 but holdout tuning used higher bands.

Example:
    python -m training.patch_low_score_remap \\
        --model models/poker44_stacked_robust.joblib \\
        --threshold 0.0062 --temperature 0.002 --score-logit-bias 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from poker44_ml.inference import Poker44Model
from training.train_model_v2 import _apply_score_remap_np, _logit_shift


def _simulate_live(
    raw: list[float],
    *,
    threshold: float,
    temperature: float,
    bias: float,
    temp_logit: float,
) -> list[float]:
    remap = {
        "kind": "threshold_logit_v1",
        "mode": "low_score",
        "threshold": threshold,
        "temperature": temperature,
    }
    mid = _apply_score_remap_np(np.asarray(raw, dtype=float), remap)
    out = _logit_shift(mid, bias, temp_logit)
    return out.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0062)
    parser.add_argument("--temperature", type=float, default=0.002)
    parser.add_argument("--score-logit-bias", type=float, default=0.0)
    parser.add_argument("--score-logit-temperature", type=float, default=1.0)
    parser.add_argument(
        "--live-raw",
        type=str,
        default="0.0045,0.0050,0.0057,0.0068,0.0094",
        help="Comma-separated sample live raw scores to print a quick simulation.",
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    metadata: dict[str, Any] = dict(artifact.get("metadata") or {})
    metadata["low_score_remap"] = True
    metadata["live_anchor_human"] = 0.006
    metadata["live_anchor_bot"] = 0.012
    metadata["score_remap"] = {
        "kind": "threshold_logit_v1",
        "mode": "low_score",
        "threshold": float(args.threshold),
        "temperature": float(args.temperature),
        "patched": True,
    }
    metadata["score_logit_bias"] = float(args.score_logit_bias)
    metadata["score_logit_temperature"] = float(args.score_logit_temperature)
    artifact["metadata"] = metadata
    joblib.dump(artifact, args.model)
    print(f"Patched {args.model}")

    raw = [float(value) for value in args.live_raw.split(",") if value.strip()]
    final = _simulate_live(
        raw,
        threshold=args.threshold,
        temperature=args.temperature,
        bias=args.score_logit_bias,
        temp_logit=args.score_logit_temperature,
    )
    print("Live raw simulation:")
    for value, score in zip(raw, final):
        print(f"  raw={value:.4f} -> final={score:.4f} flag={score >= 0.5}")
    print(
        f"predicted bot rate: {100.0 * sum(1 for s in final if s >= 0.5) / len(final):.1f}%"
    )

    loaded = Poker44Model(args.model)
    # Poker44Model does not expose raw-only; verify metadata round-trip.
    print(
        f"Loaded remap threshold={loaded.score_remap.get('threshold')} "
        f"temperature={loaded.score_remap.get('temperature')} "
        f"logit_bias={loaded.score_logit_bias}"
    )


if __name__ == "__main__":
    main()
