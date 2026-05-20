"""Feature allowlist for validator-generalized Poker44 training.

Keeps action-mix, signature, and entropy features that stay meaningful on
miner-visible validator payloads. Drops outcome/position fields cleared by
``prepare_hand_for_miner`` and absolute BB aggregates that drift across eval API
vs static benchmark.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Substrings that indicate a column is fragile or empty after sanitization.
_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "other_ratio",
    "showdown",
    "winner",
    "positive_payout",
    "rake",
    "hero_",
    "button_",
    "hero_position",
    "hero_is_button",
    "hero_button",
    "pot_mismatch",
    "pot_decrease",
    "pot_nonmonotonic",
    "pot_growth",
    "raise_to_max_bb",
    "call_to_max_bb",
    "starting_stack",
    "normalized_amount",
    "showed_hand",
    "players_to_flop",
    "low_amount_style",
)

# At least one must appear in the feature name.
_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "hand_count",
    "signature",
    "aggression_ratio",
    "call_ratio",
    "check_ratio",
    "fold_ratio",
    "raise_ratio",
    "bet_ratio",
    "action_diversity",
    "action_entropy",
    "passive_style",
    "aggressive_style",
    "low_actor_diversity",
    "deep_street",
    "preflop_raise",
    "fold_after_aggression",
    "first_action_aggressive",
    "last_action_aggressive",
    "actor_entropy",
    "action_transition_entropy",
    "actor_switch",
    "zero_amount_noncheck",
    "repeated_amount",
    "amount_entropy",
    "amount_unique_share",
    "action_count_iqr",
    "action_count_unique",
    "postflop_aggression",
    "preflop_action_share",
    "later_street_action",
    "preflop_zero_check",
    "short_hand_rate",
    "long_hand_rate",
    "very_long_hand",
    "low_action_entropy",
    "high_action_entropy",
    "low_actor_entropy",
    "high_actor_switch",
    "high_repeated_amount",
    "zero_amount_noncheck_rate",
    "unique_actor_share",
    "repeated_actor_share",
    "aggressive_action_share",
    "passive_action_share",
    "street_depth",
    "player_count_unique",
    "uniform_starting_stack",
    "max_actor_run_share",
    "max_action_run_share",
    "unique_action",
    "top_action",
    "top_actor",
    "top_preflop",
    "top_street",
    "unique_preflop",
    "unique_street",
    "unique_actor_action",
    "action_amount_signature",
    "actor_action_signature",
)


def is_robust_feature_name(name: str) -> bool:
    """Return True if ``name`` is safe for live-validator generalization."""
    lowered = str(name).strip().lower()
    if not lowered:
        return False
    if any(token in lowered for token in _EXCLUDE_SUBSTRINGS):
        return False
    return any(token in lowered for token in _INCLUDE_SUBSTRINGS)


def filter_robust_feature_names(names: Sequence[str]) -> list[str]:
    """Stable sorted allowlist intersected with available columns."""
    return sorted(name for name in names if is_robust_feature_name(name))


def summarize_robust_filter(
    all_names: Sequence[str],
    kept: Sequence[str],
) -> dict[str, int | list[str]]:
    dropped = [name for name in all_names if name not in set(kept)]
    return {
        "total": len(all_names),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_sample": dropped[:12],
    }
