from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

from poker44_ml.features import chunk_features


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN_PATH = (
    REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
)


def load_json_or_gz(path: str | Path) -> Any:
    file_path = Path(path)
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_human_path(preferred: str | Path | None) -> Path:
    candidates = [Path(preferred)] if preferred else []
    candidates.append(DEFAULT_HUMAN_PATH)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of the candidate human corpus paths exist: {joined}")


def build_human_chunk_rows(
    human_hands: list[dict[str, Any]],
    *,
    chunk_size: int = 80,
    min_chunk_size: int | None = None,
    stride: int | None = None,
    repeats: int = 3,
    seed: int = 42,
) -> list[dict[str, float]]:
    minimum = min_chunk_size if min_chunk_size is not None else max(20, chunk_size // 2)
    stride = stride if stride is not None else max(1, chunk_size // 2)
    rows: list[dict[str, float]] = []

    for repeat_index in range(max(1, repeats)):
        shuffled = list(human_hands)
        random.Random(seed + repeat_index).shuffle(shuffled)
        for start_index in range(0, len(shuffled), stride):
            chunk = shuffled[start_index : start_index + chunk_size]
            if len(chunk) < minimum:
                continue
            features = chunk_features(chunk)
            features["hand_count"] = float(len(chunk))
            rows.append(features)
    return rows
