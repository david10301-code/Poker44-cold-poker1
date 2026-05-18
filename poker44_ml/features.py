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


def _categorical_entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return _normalized_entropy([float(count) for count in counts.values()])


def _unique_share(values: list[Any]) -> float:
    return safe_div(len(set(values)), len(values)) if values else 0.0


def _top_share(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return safe_div(max(counts.values()), len(values))


def _max_run_share(values: list[Any]) -> float:
    if not values:
        return 0.0
    longest = 1
    current = 1
    for prev, value in zip(values, values[1:]):
        if value == prev:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return safe_div(longest, len(values))


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(fraction), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _iqr(values: list[float]) -> float:
    return _quantile(values, 0.75) - _quantile(values, 0.25)


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
    action_signatures: list[tuple[str, ...]] = []
    action_bigram_signatures: list[tuple[tuple[str, str], ...]] = []
    actor_signatures: list[tuple[int, ...]] = []
    street_signatures: list[tuple[str, ...]] = []
    amount_signatures: list[tuple[float, ...]] = []
    action_counts: list[float] = []
    actor_entropies: list[float] = []
    action_entropies: list[float] = []
    amount_entropies: list[float] = []
    amount_unique_shares: list[float] = []
    max_actor_run_shares: list[float] = []
    max_action_run_shares: list[float] = []
    actor_switch_rates: list[float] = []
    action_transition_entropies: list[float] = []
    zero_amount_noncheck_shares: list[float] = []
    repeated_amount_shares: list[float] = []
    hero_seats: list[int] = []
    button_seats: list[int] = []
    player_counts: list[int] = []
    uniform_starting_stack_count = 0

    for hand in chunk:
        values = _hand_feature_values(hand)
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        metadata = hand.get("metadata") or {}
        action_types = [
            str(action.get("action_type") or "").lower()
            for action in actions
            if isinstance(action, dict)
        ]
        actor_sequence = [
            int(action.get("actor_seat") or 0)
            for action in actions
            if isinstance(action, dict) and int(action.get("actor_seat") or 0)
        ]
        street_sequence = [
            str(action.get("street") or "").lower()
            for action in actions
            if isinstance(action, dict)
        ]
        amount_sequence = [
            round(safe_float(action.get("normalized_amount_bb"), 0.0), 3)
            for action in actions
            if isinstance(action, dict)
        ]
        bigrams = tuple(zip(action_types, action_types[1:]))
        action_signatures.append(tuple(action_types))
        action_bigram_signatures.append(bigrams)
        actor_signatures.append(tuple(actor_sequence))
        street_signatures.append(tuple(street_sequence))
        amount_signatures.append(tuple(amount_sequence))
        action_counts.append(float(len(actions)))
        actor_entropies.append(_categorical_entropy(actor_sequence))
        action_entropies.append(_categorical_entropy(action_types))
        amount_entropies.append(_categorical_entropy(amount_sequence))
        amount_unique_shares.append(_unique_share(amount_sequence))
        max_actor_run_shares.append(_max_run_share(actor_sequence))
        max_action_run_shares.append(_max_run_share(action_types))
        actor_switch_rates.append(
            safe_div(
                sum(1 for prev, curr in zip(actor_sequence, actor_sequence[1:]) if prev != curr),
                max(len(actor_sequence) - 1, 1),
            )
        )
        action_transition_entropies.append(_categorical_entropy(list(bigrams)))
        zero_amount_noncheck_shares.append(
            safe_div(
                sum(
                    1
                    for action_type, amount in zip(action_types, amount_sequence)
                    if amount <= 0.0 and action_type not in {"check", "fold"}
                ),
                max(len(action_types), 1),
            )
        )
        repeated_amount_shares.append(max(0.0, 1.0 - _unique_share(amount_sequence)))
        hero_seats.append(int(metadata.get("hero_seat") or 0))
        button_seats.append(int(metadata.get("button_seat") or 0))
        player_counts.append(len(players))
        starting_stacks = [
            round(safe_float(player.get("starting_stack"), 0.0), 6)
            for player in players
            if isinstance(player, dict)
        ]
        uniform_starting_stack_count += int(bool(starting_stacks) and len(set(starting_stacks)) == 1)

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
    output["top_action_signature_share"] = _top_share(action_signatures)
    output["unique_action_signature_share"] = _unique_share(action_signatures)
    output["top_action_bigram_signature_share"] = _top_share(action_bigram_signatures)
    output["unique_action_bigram_signature_share"] = _unique_share(action_bigram_signatures)
    output["top_actor_signature_share"] = _top_share(actor_signatures)
    output["unique_actor_signature_share"] = _unique_share(actor_signatures)
    output["top_street_signature_share"] = _top_share(street_signatures)
    output["unique_street_signature_share"] = _unique_share(street_signatures)
    output["top_amount_signature_share"] = _top_share(amount_signatures)
    output["unique_amount_signature_share"] = _unique_share(amount_signatures)
    output["action_count_iqr"] = _iqr(action_counts)
    output["action_count_unique_share"] = _unique_share(action_counts)
    output["actor_entropy_mean"] = safe_div(sum(actor_entropies), hand_total)
    output["actor_entropy_iqr"] = _iqr(actor_entropies)
    output["action_entropy_chunk_mean"] = safe_div(sum(action_entropies), hand_total)
    output["amount_entropy_mean"] = safe_div(sum(amount_entropies), hand_total)
    output["amount_unique_share_mean"] = safe_div(sum(amount_unique_shares), hand_total)
    output["amount_unique_share_iqr"] = _iqr(amount_unique_shares)
    output["max_actor_run_share_mean"] = safe_div(sum(max_actor_run_shares), hand_total)
    output["max_actor_run_share_max"] = max(max_actor_run_shares) if max_actor_run_shares else 0.0
    output["max_action_run_share_mean"] = safe_div(sum(max_action_run_shares), hand_total)
    output["max_action_run_share_max"] = max(max_action_run_shares) if max_action_run_shares else 0.0
    output["actor_switch_rate_mean"] = safe_div(sum(actor_switch_rates), hand_total)
    output["action_transition_entropy_mean"] = safe_div(sum(action_transition_entropies), hand_total)
    output["zero_amount_noncheck_share_mean"] = safe_div(sum(zero_amount_noncheck_shares), hand_total)
    output["repeated_amount_share_mean"] = safe_div(sum(repeated_amount_shares), hand_total)
    output["hero_seat_entropy"] = _categorical_entropy(hero_seats)
    output["button_seat_entropy"] = _categorical_entropy(button_seats)
    output["player_count_unique_share"] = _unique_share(player_counts)
    output["uniform_starting_stack_rate"] = uniform_starting_stack_count / hand_total
    return output
