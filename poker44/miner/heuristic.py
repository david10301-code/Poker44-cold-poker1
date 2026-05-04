"""Deterministic heuristic scorer used as a fallback or baseline miner."""

from __future__ import annotations

from collections import Counter


class HeuristicChunkScorer:
    @staticmethod
    def clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def score_hand(self, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * self.clamp01(call_ratio / 0.35)
        score += 0.12 * self.clamp01(check_ratio / 0.30)
        score += 0.08 * self.clamp01(player_count_signal)
        score -= 0.18 * self.clamp01(fold_ratio / 0.55)
        score -= 0.10 * self.clamp01(raise_ratio / 0.20)
        return self.clamp01(score)

    def score_chunk(self, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5
        hand_scores = [self.score_hand(hand) for hand in chunk]
        return round(self.clamp01(sum(hand_scores) / len(hand_scores)), 6)

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        return [self.score_chunk(chunk) for chunk in chunks]
