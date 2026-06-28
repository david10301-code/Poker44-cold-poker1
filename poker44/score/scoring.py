"""Reward and scoring utilities for Poker44 poker bot detection.

This mirrors the **live subnet reward** (Poker44-subnet >= 0.1.25, current
deploy 0.1.32): a *rank-first* reward that protects humans without rewarding
top-k guessing.

    ap_score        = average_precision_score(y_true, y_pred)
    bot_recall, fpr = _recall_at_fpr(y_pred, y_true, max_fpr=0.05)
    reward          = 0.75 * ap_score + 0.25 * bot_recall      # penalty = 1.0

Both terms are pure ranking metrics, so any monotonic post-processing
(calibration, score_remap, score_logit, threshold placement, top-k "bot
budget") leaves the reward unchanged — only the model's *ranking* of the
current live distribution matters. The pre-0.1.25 formula (fixed-0.5 threshold,
``(1-fpr)**2`` penalty with a 0.10 cliff) is kept as :func:`legacy_reward`
purely for before/after comparison and is not used anywhere by default.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def _recall_at_fpr(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> tuple[float, float]:
    """Best bot recall reachable while keeping human false-positive rate bounded.

    Sweeps every threshold and returns the highest recall whose false-positive
    rate stays <= ``max_fpr`` (and the FPR at that operating point). This is the
    25% term of the live reward; it rewards a *clean top of the ranking*
    (catching bots before humans start being flagged), independent of where any
    fixed threshold sits.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count <= 0 or negative_count <= 0 or scores.size == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(positive_count, 1)
    fpr = fp / max(negative_count, 1)

    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0

    allowed_indices = np.flatnonzero(allowed)
    best_local = int(allowed_indices[np.argmax(recall[allowed])])
    return float(recall[best_local]), float(fpr[best_local])


def reward(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    """Live rank-first reward (subnet >= 0.1.25). Matches Poker44-subnet exactly.

    Returns ``(reward, details)`` with the same dict keys the old formula used
    (``fpr``, ``bot_recall``, ``ap_score``, ``human_safety_penalty``,
    ``base_score``, ``reward``) so every caller keeps working — but the values
    now reflect the live reward:

    * ``ap_score``   — average precision (the 75% term).
    * ``bot_recall`` — recall at FPR <= 0.05 (the 25% term), NOT recall@0.5.
    * ``fpr``        — the FPR at that recall operating point.
    * ``human_safety_penalty`` — always 1.0 (no penalty under rank-first).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    if y_pred.size and np.any(y_true == 1):
        ap_score = float(average_precision_score(y_true, y_pred))
    else:
        ap_score = 0.0

    bot_recall, fpr = _recall_at_fpr(y_pred, y_true, max_fpr=0.05)
    human_safety_penalty = 1.0

    base_score = 0.75 * ap_score + 0.25 * bot_recall
    rew = base_score * human_safety_penalty

    res = {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "base_score": base_score,
        "reward": rew,
    }
    return rew, res


def reward_eval(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    mode: str = "live",
) -> tuple[float, dict]:
    """Evaluation wrapper around :func:`reward`.

    Under the rank-first live reward there is no FPR penalty to vary, so the
    historical ``live`` / ``base`` / ``soft`` modes all return the **same**
    rank-first reward. ``mode`` is retained for CLI back-compat and only tags
    ``reward_mode`` in the returned details.
    """
    if mode not in ("live", "base", "soft"):
        raise ValueError(f"Unknown reward eval mode: {mode!r}")
    rew, details = reward(y_pred, y_true)
    return rew, {**details, "reward_mode": mode}


def format_reward_breakdown(
    ap_score: float,
    bot_recall: float,
    *,
    fpr: float = 0.0,
    reward: float | None = None,
) -> str:
    """One-line decomposition of the live rank-first reward into its two terms.

    ``reward = 0.75*AP + 0.25*recall@(FPR<=0.05)``. Shows each weighted
    contribution plus the per-term *headroom* (``weight * (1 - metric)``) so it
    is obvious which term to push for the biggest reward gain.
    """
    ap = float(ap_score)
    recall = float(bot_recall)
    rew = (0.75 * ap + 0.25 * recall) if reward is None else float(reward)
    ap_term, recall_term = 0.75 * ap, 0.25 * recall
    ap_headroom, recall_headroom = 0.75 * (1.0 - ap), 0.25 * (1.0 - recall)
    push = "AP" if ap_headroom >= recall_headroom else "recall@FPR<=0.05"
    return (
        f"reward={rew:.4f} = 0.75*AP({ap:.4f})={ap_term:.4f} "
        f"+ 0.25*recall@FPR<=0.05({recall:.4f}, fpr={fpr:.4f})={recall_term:.4f} | "
        f"headroom AP=+{ap_headroom:.4f} recall=+{recall_headroom:.4f} -> push {push}"
    )


def legacy_reward(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    """Obsolete pre-0.1.25 reward (fixed-0.5 threshold + FPR-cliff penalty).

    Kept ONLY so you can compare old-vs-new on the same scores. Not wired into
    training or eval. ``reward = (0.65*AP + 0.35*recall@0.5) * (1-fpr)**2`` with
    a hard 0 below FPR 0.10.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    preds = np.round(y_pred).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    negative_count = max(tn + fp, 1)
    positive_count = max(tp + fn, 1)

    fpr = fp / negative_count
    bot_recall = tp / positive_count

    if y_pred.size and np.any(y_true == 1):
        ap_score = float(average_precision_score(y_true, y_pred))
    else:
        ap_score = 0.0

    human_safety_penalty = max(0.0, 1.0 - fpr) ** 2
    if fpr >= 0.10:
        human_safety_penalty = 0.0

    base_score = 0.65 * ap_score + 0.35 * bot_recall
    rew = base_score * human_safety_penalty

    res = {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "base_score": base_score,
        "reward": rew,
    }
    return rew, res
