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
    "other_ratio",
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
    "preflop_action_share",
    "later_street_action_share",
    "unique_actor_share",
    "repeated_actor_share",
    "hero_action_share",
    "button_action_share",
    "zero_amount_action_share",
    "normalized_amount_mean_bb",
    "normalized_amount_max_bb",
    "normalized_amount_std_bb",
    "pot_growth_bb",
    "pot_growth_per_action_bb",
    "raise_to_max_bb",
    "call_to_max_bb",
    "starting_stack_mean_bb",
    "starting_stack_std_bb",
    "showed_hand_share",
    "winner_count_signal",
    "positive_payout_count_signal",
    "rake_to_pot_ratio",
    "hero_is_button",
    "hero_position_signal",
    "players_to_flop_signal",
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
    metadata = hand.get("metadata") or {}

    call_count = 0
    check_count = 0
    bet_count = 0
    raise_count = 0
    fold_count = 0
    other_count = 0
    zero_amount_count = 0
    preflop_count = 0
    later_street_count = 0
    action_actor_sequence: list[int] = []
    hero_action_count = 0
    button_action_count = 0
    amount_values_bb: list[float] = []
    raise_to_values_bb: list[float] = []
    call_to_values_bb: list[float] = []
    pot_before_values: list[float] = []
    pot_after_values: list[float] = []
    hero_seat = int(metadata.get("hero_seat") or 0)
    button_seat = int(metadata.get("button_seat") or 0)
    max_seats = max(1, int(metadata.get("max_seats") or len(players) or 1))
    bb_value = safe_float(metadata.get("bb"), 0.0)
    for action in actions:
        action_type = (action.get("action_type") or "").lower()
        action_actor = int(action.get("actor_seat") or 0)
        if action_actor:
            action_actor_sequence.append(action_actor)
            hero_action_count += int(action_actor == hero_seat)
            button_action_count += int(action_actor == button_seat)
        street_name = (action.get("street") or "").lower()
        preflop_count += int(street_name == "preflop")
        later_street_count += int(street_name not in {"", "preflop"})
        amount_bb = safe_float(action.get("normalized_amount_bb"), 0.0)
        amount_values_bb.append(amount_bb)
        raise_to_value = safe_float(action.get("raise_to"), 0.0)
        call_to_value = safe_float(action.get("call_to"), 0.0)
        if bb_value > 0.0:
            raise_to_value /= bb_value
            call_to_value /= bb_value
        raise_to_values_bb.append(raise_to_value)
        call_to_values_bb.append(call_to_value)
        pot_before_values.append(safe_float(action.get("pot_before"), 0.0))
        pot_after_values.append(safe_float(action.get("pot_after"), 0.0))
        if amount_bb <= 0.0:
            zero_amount_count += 1
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
        else:
            other_count += 1

    meaningful_actions = max(
        1, call_count + check_count + bet_count + raise_count + fold_count
    )
    aggressive_actions = bet_count + raise_count
    passive_actions = call_count + check_count
    total_actions = max(len(actions), 1)

    call_ratio = call_count / meaningful_actions
    check_ratio = check_count / meaningful_actions
    fold_ratio = fold_count / meaningful_actions
    raise_ratio = raise_count / meaningful_actions
    bet_ratio = bet_count / meaningful_actions
    other_ratio = other_count / total_actions
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
    unique_actor_share = safe_div(len(set(action_actor_sequence)), max(1.0, player_count))
    repeated_actor_share = safe_div(
        sum(
            1
            for prev, curr in zip(action_actor_sequence, action_actor_sequence[1:])
            if prev == curr
        ),
        max(len(action_actor_sequence) - 1, 1),
    )
    hero_action_share = safe_div(hero_action_count, total_actions)
    button_action_share = safe_div(button_action_count, total_actions)
    preflop_action_share = safe_div(preflop_count, total_actions)
    later_street_action_share = safe_div(later_street_count, total_actions)
    zero_amount_action_share = safe_div(zero_amount_count, total_actions)
    normalized_amount_mean_bb = safe_div(sum(amount_values_bb), len(amount_values_bb))
    normalized_amount_max_bb = max(amount_values_bb) if amount_values_bb else 0.0
    normalized_amount_std_bb = math.sqrt(
        max(
            0.0,
            safe_div(sum(value * value for value in amount_values_bb), len(amount_values_bb))
            - normalized_amount_mean_bb * normalized_amount_mean_bb,
        )
    )
    pot_growth = (
        max(pot_after_values) - min(pot_before_values)
        if pot_after_values and pot_before_values
        else 0.0
    )
    if bb_value > 0.0:
        pot_growth /= bb_value
    pot_growth_bb = max(0.0, pot_growth)
    pot_growth_per_action_bb = safe_div(pot_growth_bb, total_actions)
    raise_to_max_bb = max(raise_to_values_bb) if raise_to_values_bb else 0.0
    call_to_max_bb = max(call_to_values_bb) if call_to_values_bb else 0.0
    starting_stacks_bb = [
        safe_div(safe_float(player.get("starting_stack"), 0.0), bb_value)
        if bb_value > 0.0
        else safe_float(player.get("starting_stack"), 0.0)
        for player in players
    ]
    starting_stack_mean_bb = safe_div(sum(starting_stacks_bb), len(starting_stacks_bb))
    starting_stack_std_bb = math.sqrt(
        max(
            0.0,
            safe_div(
                sum(value * value for value in starting_stacks_bb), len(starting_stacks_bb)
            ) - starting_stack_mean_bb * starting_stack_mean_bb,
        )
    )
    showed_hand_share = safe_div(
        sum(1 for player in players if player.get("showed_hand")),
        max(1.0, player_count),
    )
    winners = outcome.get("winners") or []
    payouts = outcome.get("payouts") or {}
    positive_payout_count = sum(
        1 for value in payouts.values() if safe_float(value, 0.0) > 0.0
    )
    total_pot = safe_float(outcome.get("total_pot"), 0.0)
    rake = safe_float(outcome.get("rake"), 0.0)
    winner_count_signal = safe_div(len(winners), max(1.0, player_count))
    positive_payout_count_signal = safe_div(positive_payout_count, max(1.0, player_count))
    rake_to_pot_ratio = safe_div(rake, total_pot)
    hero_is_button = 1.0 if hero_seat and hero_seat == button_seat else 0.0
    hero_position_signal = 0.0
    if hero_seat and button_seat:
        hero_position_signal = ((hero_seat - button_seat) % max_seats) / max(1, max_seats - 1)
    players_to_flop_signal = safe_div(len(streets), max(1.0, player_count))

    return (
        call_ratio,
        check_ratio,
        fold_ratio,
        raise_ratio,
        bet_ratio,
        other_ratio,
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
        preflop_action_share,
        later_street_action_share,
        unique_actor_share,
        repeated_actor_share,
        hero_action_share,
        button_action_share,
        zero_amount_action_share,
        normalized_amount_mean_bb,
        normalized_amount_max_bb,
        normalized_amount_std_bb,
        pot_growth_bb,
        pot_growth_per_action_bb,
        raise_to_max_bb,
        call_to_max_bb,
        starting_stack_mean_bb,
        starting_stack_std_bb,
        showed_hand_share,
        winner_count_signal,
        positive_payout_count_signal,
        rake_to_pot_ratio,
        hero_is_button,
        hero_position_signal,
        players_to_flop_signal,
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
    low_amount_style_count = 0
    hero_button_count = 0
    low_actor_diversity_count = 0

    for hand in chunk:
        values = _hand_feature_values(hand)
        street_depth = values[9]
        showdown_flag = values[10]
        aggressive_share = values[14]
        passive_share = values[15]
        action_diversity = values[7]
        normalized_amount_mean_bb = values[22]
        hero_is_button = values[35]
        showdown_total += showdown_flag
        deep_street_count += int(street_depth >= 0.75)
        passive_style_count += int(passive_share >= 0.55)
        aggressive_style_count += int(aggressive_share >= 0.35)
        low_amount_style_count += int(normalized_amount_mean_bb <= 0.5)
        hero_button_count += int(hero_is_button >= 1.0)
        low_actor_diversity_count += int(action_diversity <= 0.4)

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
    output["low_amount_style_rate"] = low_amount_style_count / hand_total
    output["hero_button_rate"] = hero_button_count / hand_total
    output["low_actor_diversity_rate"] = low_actor_diversity_count / hand_total
    return output
