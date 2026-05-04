from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Any

from poker44.miner.config import repository_paths
from poker44_ml.features import chunk_features


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = repository_paths(REPO_ROOT)
DEFAULT_HUMAN_PATHS = (PATHS.human_hands_path,)
DEFAULT_BOT_PATHS = (PATHS.generated_bot_hands_path,)


def load_json_or_gz(path: str | Path) -> Any:
    file_path = Path(path)
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_existing_path(preferred: str | Path | None, fallbacks: tuple[Path, ...]) -> Path:
    candidates = [Path(preferred)] if preferred else []
    candidates.extend(fallbacks)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of the candidate paths exist: {joined}")


def build_chunks(
    hands: list[dict[str, Any]],
    label: int,
    chunk_size: int = 80,
    min_chunk_size: int | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    shuffled = list(hands)
    rng.shuffle(shuffled)

    minimum = min_chunk_size if min_chunk_size is not None else max(20, chunk_size // 2)
    rows: list[dict[str, float]] = []
    for index in range(0, len(shuffled), chunk_size):
        chunk = shuffled[index : index + chunk_size]
        if len(chunk) < minimum:
            continue
        features = chunk_features(chunk)
        features["label"] = float(label)
        features["hand_count"] = float(len(chunk))
        rows.append(features)
    return rows


def build_training_dataframe(
    human_hands: list[dict[str, Any]],
    bot_hands: list[dict[str, Any]],
    chunk_size: int = 80,
    min_chunk_size: int | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    rows.extend(
        build_chunks(
            human_hands,
            label=0,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            seed=seed,
        )
    )
    rows.extend(
        build_chunks(
            bot_hands,
            label=1,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            seed=seed + 1,
        )
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a chunk-level Poker44 training dataset.")
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--bot-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(PATHS.data_dir / "training_chunks.csv"))
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_path = resolve_existing_path(args.human_path, DEFAULT_HUMAN_PATHS)
    bot_path = resolve_existing_path(args.bot_path, DEFAULT_BOT_PATHS)

    human_hands = load_json_or_gz(human_path)
    bot_hands = load_json_or_gz(bot_path)
    df = build_training_dataframe(
        human_hands=human_hands,
        bot_hands=bot_hands,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(df, handle)
    else:
        if not df:
            raise RuntimeError("No rows were generated for dataset export.")
        fieldnames = sorted(df[0].keys())
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(df)

    print(f"Saved {len(df)} chunk rows to {output_path}")
    print(f"Human source: {human_path}")
    print(f"Bot source:   {bot_path}")


if __name__ == "__main__":
    main()
