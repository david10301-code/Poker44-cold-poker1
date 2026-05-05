from __future__ import annotations

from collections import Counter
from typing import Any

import math
import statistics


MEANINGFUL_ACTIONS = ("call", "check", "bet", "raise", "fold")
AGGRESSIVE_ACTIONS = ("bet", "raise")
PASSIVE_ACTIONS = ("call", "check")


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(statistics.fmean(values)),
        f"{prefix}_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
    }


def coefficient_of_variation(values: list[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    if not positives:
        return 0.0
    mean = float(statistics.fmean(positives))
    return safe_div(
        float(statistics.pstdev(positives)) if len(positives) > 1 else 0.0,
        mean,
    )


def _normalized_entropy(values: list[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    total = sum(positives)
    if total <= 0.0 or len(positives) <= 1:
        return 0.0
    probs = [value / total for value in positives]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    return safe_div(entropy, math.log(len(probs)))


def _action_counts(actions: list[dict[str, Any]]) -> Counter[str]:
    return Counter((action.get("action_type") or "").lower() for action in actions)


def _hand_features(hand: dict[str, Any]) -> dict[str, float]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}

    action_counts = _action_counts(actions)
    meaningful_actions = max(
        1,
        sum(action_counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS),
    )
    aggressive_actions = sum(action_counts.get(kind, 0) for kind in AGGRESSIVE_ACTIONS)
    passive_actions = sum(action_counts.get(kind, 0) for kind in PASSIVE_ACTIONS)
    total_actions = max(len(actions), 1)

    call_ratio = action_counts.get("call", 0) / meaningful_actions
    check_ratio = action_counts.get("check", 0) / meaningful_actions
    fold_ratio = action_counts.get("fold", 0) / meaningful_actions
    raise_ratio = action_counts.get("raise", 0) / meaningful_actions
    bet_ratio = action_counts.get("bet", 0) / meaningful_actions
    aggression_ratio = safe_div(aggressive_actions, aggressive_actions + passive_actions)
    action_diversity = len([kind for kind in MEANINGFUL_ACTIONS if action_counts.get(kind, 0)]) / len(
        MEANINGFUL_ACTIONS
    )
    action_entropy = _normalized_entropy(
        [action_counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS]
    )
    player_count = float(len(players))
    player_count_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0
    street_depth = len(streets) / 4.0
    showdown_flag = 1.0 if outcome.get("showdown") else 0.0

    return {
        "call_ratio": call_ratio,
        "check_ratio": check_ratio,
        "fold_ratio": fold_ratio,
        "raise_ratio": raise_ratio,
        "bet_ratio": bet_ratio,
        "aggression_ratio": aggression_ratio,
        "action_diversity": action_diversity,
        "action_entropy": action_entropy,
        "street_depth": street_depth,
        "showdown_flag": showdown_flag,
        "player_count": player_count,
        "player_count_signal": player_count_signal,
        "total_actions": float(total_actions),
        "aggressive_action_share": aggressive_actions / total_actions,
        "passive_action_share": passive_actions / total_actions,
    }


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    if not chunk:
        return {"hand_count": 0.0}

    per_hand = [_hand_features(hand) for hand in chunk]
    feature_names = sorted(per_hand[0].keys())
    output: dict[str, float] = {"hand_count": float(len(chunk))}

    for feature_name in feature_names:
        values = [features[feature_name] for features in per_hand]
        output.update(summarize(values, feature_name))
        output[f"{feature_name}_cv"] = coefficient_of_variation(values)

    output["showdown_rate"] = safe_div(
        sum(features["showdown_flag"] for features in per_hand),
        len(per_hand),
    )
    output["deep_street_rate"] = safe_div(
        sum(1 for features in per_hand if features["street_depth"] >= 0.75),
        len(per_hand),
    )
    output["passive_style_rate"] = safe_div(
        sum(1 for features in per_hand if features["passive_action_share"] >= 0.55),
        len(per_hand),
    )
    output["aggressive_style_rate"] = safe_div(
        sum(1 for features in per_hand if features["aggressive_action_share"] >= 0.35),
        len(per_hand),
    )
    return output
