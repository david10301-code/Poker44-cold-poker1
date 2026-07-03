"""Feature allowlist for validator-generalized Poker44 training.

Keeps action-mix, signature, and entropy features that stay meaningful on
miner-visible validator payloads. Drops outcome/position fields cleared by
``prepare_hand_for_miner`` and absolute BB aggregates that drift across eval API
vs static benchmark.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Substrings that indicate a column is fragile or empty after sanitization.
# Verified 2026-07-01 (benchmark vs real live-format sample):
#   * button_action_share / hero_button_same: button_seat is always 0 on BOTH
#     benchmark and live, so poker44_ml/features.py hard-zeros these (dead const).
#   * *_bb absolute magnitudes (amount_*_bb, pot_*_bb, starting_stack_*_bb):
#     2-11 sigma OOD on the sanitized live feed (live pots/bets ~half benchmark
#     size) -> trees split on benchmark-scale thresholds that collapse on live.
_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "button_action_share",
    "hero_button_same",
    "_bb",
)

# At least one must appear in the feature name.
_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "hand_count",
    "schema_",
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
