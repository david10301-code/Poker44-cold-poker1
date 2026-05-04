"""Unified local-model and heuristic scoring pipeline for miner responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker44.miner.config import MinerRuntimeConfig
from poker44.miner.heuristic import HeuristicChunkScorer

try:
    from poker44_ml.inference import Poker44Model
except ImportError:  # pragma: no cover - optional local-model path.
    Poker44Model = None


@dataclass(frozen=True)
class ChunkScoringResult:
    scores: list[float]
    backend: str
    raw_chunk_sizes: list[int]
    eval_chunk_sizes: list[int]


class ChunkScoringPipeline:
    def __init__(self, config: MinerRuntimeConfig):
        self.config = config
        self.heuristic = HeuristicChunkScorer()
        self.predictor = None
        self.predictor_metadata: dict[str, Any] = {}
        self.backend = "heuristic"
        self.load_error: str | None = None
        self._try_load_local_model()

    def _try_load_local_model(self) -> None:
        if not self.config.enable_local_model:
            self.load_error = "Local model loading disabled via POKER44_ENABLE_LOCAL_MODEL."
            return
        if Poker44Model is None:
            self.load_error = "poker44_ml runtime dependencies are unavailable."
            return
        if not self.config.model_path.exists():
            self.load_error = f"Model artifact not found: {self.config.model_path}"
            return
        try:
            self.predictor = Poker44Model(str(self.config.model_path))
            self.predictor_metadata = dict(getattr(self.predictor, "metadata", {}) or {})
            self.backend = "trained-model"
        except Exception as err:  # pragma: no cover - runtime fallback path.
            self.load_error = str(err)
            self.predictor = None
            self.predictor_metadata = {}
            self.backend = "heuristic"

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _compress_chunk(self, chunk: list[dict]) -> list[dict]:
        limit = self.config.max_hands_per_chunk_eval
        if limit <= 0 or len(chunk) <= limit:
            return chunk
        if limit == 1:
            return [chunk[len(chunk) // 2]]
        last_index = len(chunk) - 1
        slots = limit - 1
        indices = {
            min(last_index, round(idx * last_index / slots))
            for idx in range(limit)
        }
        return [chunk[index] for index in sorted(indices)]

    def _score_with_predictor(self, chunks: list[list[dict]]) -> list[float]:
        assert self.predictor is not None
        return [
            round(self._clamp01(score), 6)
            for score in self.predictor.predict_chunk_scores(chunks)
        ]

    def score_chunks(self, chunks: list[list[dict]]) -> ChunkScoringResult:
        raw_chunk_sizes = [len(chunk or []) for chunk in chunks]
        eval_chunks = [self._compress_chunk(list(chunk or [])) for chunk in chunks]
        eval_chunk_sizes = [len(chunk) for chunk in eval_chunks]

        scores: list[float]
        backend = self.backend
        if self.predictor is not None:
            try:
                scores = self._score_with_predictor(eval_chunks)
            except Exception as err:  # pragma: no cover - runtime fallback path.
                self.load_error = str(err)
                backend = "heuristic-fallback"
                scores = self.heuristic.score_chunks(eval_chunks)
        else:
            scores = self.heuristic.score_chunks(eval_chunks)

        if len(scores) < len(chunks):
            scores.extend([0.5] * (len(chunks) - len(scores)))
        scores = [round(self._clamp01(score), 6) for score in scores[: len(chunks)]]
        return ChunkScoringResult(
            scores=scores,
            backend=backend,
            raw_chunk_sizes=raw_chunk_sizes,
            eval_chunk_sizes=eval_chunk_sizes,
        )
