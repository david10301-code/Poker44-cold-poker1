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
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "hands_generator" / "evaluation_datas"
DEFAULT_BENCHMARK_PATTERN = "training_benchmark_*.txt"


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


def resolve_benchmark_paths(preferred: str | Path | None) -> list[Path]:
    if preferred:
        candidate = Path(preferred)
        if candidate.is_dir():
            paths = sorted(candidate.glob(DEFAULT_BENCHMARK_PATTERN))
        else:
            paths = [candidate] if candidate.exists() else []
    else:
        paths = sorted(DEFAULT_BENCHMARK_DIR.glob(DEFAULT_BENCHMARK_PATTERN))

    existing = [path for path in paths if path.exists()]
    if existing:
        return existing

    target = str(preferred) if preferred else str(DEFAULT_BENCHMARK_DIR / DEFAULT_BENCHMARK_PATTERN)
    raise FileNotFoundError(f"No benchmark files found for: {target}")


def _chunk_row(chunk: list[dict[str, Any]]) -> dict[str, float]:
    features = chunk_features(chunk)
    features["hand_count"] = float(len(chunk))
    return features


def _benchmark_examples_from_file(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json_or_gz(path)
    root = payload.get("data") if isinstance(payload, dict) else None
    root = root if isinstance(root, dict) else payload
    groups = root.get("chunks") if isinstance(root, dict) else None
    if not isinstance(groups, list):
        raise RuntimeError(f"Benchmark payload is missing a top-level chunks list: {path}")

    source_path = str(Path(path))
    examples: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        chunk_list = group.get("chunks") or []
        labels = group.get("groundTruth") or []
        if len(chunk_list) != len(labels):
            raise RuntimeError(
                f"Benchmark group {group.get('chunkId', group_index)} has mismatched "
                f"chunks ({len(chunk_list)}) and groundTruth ({len(labels)}) in {path}."
            )
        source_date = str(group.get("sourceDate") or root.get("sourceDate") or "")
        group_id = str(group.get("chunkId") or f"group_{group_index}")
        group_hash = str(group.get("chunkHash") or "")
        for item_index, (chunk, label) in enumerate(zip(chunk_list, labels)):
            if not isinstance(chunk, list):
                continue
            examples.append(
                {
                    "chunk": list(chunk),
                    "label": int(label),
                    "source_date": source_date,
                    "group_id": group_id,
                    "group_hash": group_hash,
                    "item_index": item_index,
                    "source_path": source_path,
                    "features": _chunk_row(list(chunk)),
                }
            )
    return examples


def load_benchmark_examples(paths: str | Path | list[str | Path]) -> list[dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(path) for path in paths]

    examples: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for path in path_list:
        for example in _benchmark_examples_from_file(path):
            dedupe_key = "|".join(
                [
                    example.get("group_hash", ""),
                    example.get("group_id", ""),
                    str(example.get("item_index", "")),
                ]
            )
            if dedupe_key in {"||", ""}:
                dedupe_key = f"{example.get('source_path', '')}|{example.get('group_id', '')}|{example.get('item_index', '')}"
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            examples.append(example)
    if not examples:
        raise RuntimeError("No benchmark chunk examples were found in the selected files.")
    return examples


def build_human_chunk_rows(
    human_hands: list[dict[str, Any]],
    *,
    chunk_size: int = 80,
    min_chunk_size: int | None = None,
    stride: int | None = None,
    repeats: int = 1,
    seed: int = 42,
    shuffle: bool = False,
) -> list[dict[str, float]]:
    minimum = min_chunk_size if min_chunk_size is not None else max(20, chunk_size // 2)
    stride = stride if stride is not None else max(1, chunk_size // 2)
    rows: list[dict[str, float]] = []
    hands = list(human_hands)

    if shuffle:
        random.Random(seed).shuffle(hands)

    phase_count = max(1, repeats)
    phase_step = max(1, stride // phase_count)
    seen_offsets: set[int] = set()

    for repeat_index in range(phase_count):
        if shuffle and repeat_index:
            random.Random(seed + repeat_index).shuffle(hands)
            start_offset = 0
        else:
            start_offset = min(stride - 1, repeat_index * phase_step)
        if start_offset in seen_offsets and not shuffle:
            continue
        seen_offsets.add(start_offset)
        for start_index in range(start_offset, len(hands), stride):
            chunk = hands[start_index : start_index + chunk_size]
            if len(chunk) < minimum:
                continue
            rows.append(_chunk_row(chunk))
    return rows
