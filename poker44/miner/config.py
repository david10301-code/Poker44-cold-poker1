"""Shared path and runtime configuration for Poker44 miners."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RepositoryPaths:
    repo_root: Path
    data_dir: Path
    models_dir: Path
    docs_dir: Path
    human_hands_path: Path
    generated_bot_hands_path: Path
    default_model_path: Path


def repository_paths(repo_root: Path) -> RepositoryPaths:
    return RepositoryPaths(
        repo_root=repo_root,
        data_dir=repo_root / "data",
        models_dir=repo_root / "models",
        docs_dir=repo_root / "docs",
        human_hands_path=repo_root
        / "hands_generator"
        / "human_hands"
        / "poker_hands_combined.json.gz",
        generated_bot_hands_path=repo_root / "data" / "generated_bot_hands.json",
        default_model_path=repo_root / "models" / "poker44_xgb_calibrated.joblib",
    )


@dataclass(frozen=True)
class MinerRuntimeConfig:
    repo_root: Path
    model_path: Path
    max_hands_per_chunk_eval: int
    query_log_preview: bool
    enable_local_model: bool

    @classmethod
    def from_env(cls, repo_root: Path) -> "MinerRuntimeConfig":
        paths = repository_paths(repo_root)
        model_path = Path(os.getenv("POKER44_MODEL_PATH", str(paths.default_model_path)))
        return cls(
            repo_root=repo_root,
            model_path=model_path,
            max_hands_per_chunk_eval=max(
                0, int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK_EVAL", "120"))
            ),
            query_log_preview=_parse_bool_env("POKER44_LOG_QUERY_PREVIEW", False),
            enable_local_model=_parse_bool_env("POKER44_ENABLE_LOCAL_MODEL", True),
        )
