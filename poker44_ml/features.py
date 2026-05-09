from __future__ import annotations

from typing import Any

import math


MEANINGFUL_ACTIONS = ("call", "check", "bet", "raise", "fold")
FEATURE_NAMES = (
    "call_ratio",
    "check_ratio",
    "fold_ratio",
    "raise_ratio",
    "bet_ratio",
    "aggression_ratio",
    "action_diversity",
    "action_entropy",
    "street_depth",
    "showdown_flag",
    "player_count",
    "player_count_signal",
    "total_actions",
    "aggressive_action_share",
    "passive_action_share",
)


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


def _normalized_entropy(values: list[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    total = sum(positives)
    if total <= 0.0 or len(positives) <= 1:
        return 0.0
    probs = [value / total for value in positives]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    return safe_div(entropy, math.log(len(probs)))


def _hand_feature_values(hand: dict[str, Any]) -> tuple[float, ...]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}

    call_count = 0
    check_count = 0
    bet_count = 0
    raise_count = 0
    fold_count = 0
    for action in actions:
        action_type = (action.get("action_type") or "").lower()
        if action_type == "call":
            call_count += 1
        elif action_type == "check":
            check_count += 1
        elif action_type == "bet":
            bet_count += 1
        elif action_type == "raise":
            raise_count += 1
        elif action_type == "fold":
            fold_count += 1

    meaningful_actions = max(1, call_count + check_count + bet_count + raise_count + fold_count)
    aggressive_actions = bet_count + raise_count
    passive_actions = call_count + check_count
    total_actions = max(len(actions), 1)

    call_ratio = call_count / meaningful_actions
    check_ratio = check_count / meaningful_actions
    fold_ratio = fold_count / meaningful_actions
    raise_ratio = raise_count / meaningful_actions
    bet_ratio = bet_count / meaningful_actions
    aggression_ratio = safe_div(aggressive_actions, aggressive_actions + passive_actions)
    active_kinds = (
        int(call_count > 0)
        + int(check_count > 0)
        + int(bet_count > 0)
        + int(raise_count > 0)
        + int(fold_count > 0)
    )
    action_diversity = active_kinds / len(MEANINGFUL_ACTIONS)
    action_entropy = _normalized_entropy(
        [call_count, check_count, bet_count, raise_count, fold_count]
    )
    player_count = float(len(players))
    player_count_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0
    street_depth = len(streets) / 4.0
    showdown_flag = 1.0 if outcome.get("showdown") else 0.0

    return (
        call_ratio,
        check_ratio,
        fold_ratio,
        raise_ratio,
        bet_ratio,
        aggression_ratio,
        action_diversity,
        action_entropy,
        street_depth,
        showdown_flag,
        player_count,
        player_count_signal,
        float(total_actions),
        aggressive_actions / total_actions,
        passive_actions / total_actions,
    )


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    if not chunk:
        return {"hand_count": 0.0}

    output: dict[str, float] = {"hand_count": float(len(chunk))}
    feature_count = len(FEATURE_NAMES)
    sums = [0.0] * feature_count
    sums_sq = [0.0] * feature_count
    mins = [float("inf")] * feature_count
    maxs = [float("-inf")] * feature_count
    positive_sums = [0.0] * feature_count
    positive_sums_sq = [0.0] * feature_count
    positive_counts = [0] * feature_count
    showdown_total = 0.0
    deep_street_count = 0
    passive_style_count = 0
    aggressive_style_count = 0

    for hand in chunk:
        values = _hand_feature_values(hand)
        street_depth = values[8]
        showdown_flag = values[9]
        aggressive_share = values[13]
        passive_share = values[14]
        showdown_total += showdown_flag
        deep_street_count += int(street_depth >= 0.75)
        passive_style_count += int(passive_share >= 0.55)
        aggressive_style_count += int(aggressive_share >= 0.35)

        for idx, value in enumerate(values):
            sums[idx] += value
            sums_sq[idx] += value * value
            if value < mins[idx]:
                mins[idx] = value
            if value > maxs[idx]:
                maxs[idx] = value
            if value > 0.0:
                positive_sums[idx] += value
                positive_sums_sq[idx] += value * value
                positive_counts[idx] += 1

    hand_total = float(len(chunk))
    for idx, feature_name in enumerate(FEATURE_NAMES):
        mean = sums[idx] / hand_total
        variance = max(0.0, (sums_sq[idx] / hand_total) - (mean * mean))
        output[f"{feature_name}_mean"] = mean
        output[f"{feature_name}_std"] = math.sqrt(variance)
        output[f"{feature_name}_min"] = mins[idx]
        output[f"{feature_name}_max"] = maxs[idx]

        positive_count = positive_counts[idx]
        if positive_count <= 0 or positive_sums[idx] <= 0.0:
            output[f"{feature_name}_cv"] = 0.0
        else:
            positive_mean = positive_sums[idx] / positive_count
            positive_variance = max(
                0.0,
                (positive_sums_sq[idx] / positive_count) - (positive_mean * positive_mean),
            )
            output[f"{feature_name}_cv"] = safe_div(math.sqrt(positive_variance), positive_mean)

    output["showdown_rate"] = showdown_total / hand_total
    output["deep_street_rate"] = deep_street_count / hand_total
    output["passive_style_rate"] = passive_style_count / hand_total
    output["aggressive_style_rate"] = aggressive_style_count / hand_total
    return output
