from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from poker44_ml.features import chunk_features

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


class Poker44Model:
    """Small runtime wrapper for the rebuilt supervised Poker44 artifact."""

    def __init__(self, model_path: str | Path):
        if joblib is None:
            raise RuntimeError("joblib is required to load Poker44 models.")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        artifact = joblib.load(self.model_path)
        self.models = list(artifact.get("models") or [])
        if not self.models and artifact.get("model") is not None:
            self.models = [artifact["model"]]
        if not self.models:
            raise RuntimeError("Model artifact contains no models.")

        self.feature_names = list(artifact.get("feature_names") or [])
        self.metadata = dict(artifact.get("metadata") or {})
        self.calibrator = artifact.get("calibrator")
        self.human_guard = dict(self.metadata.get("human_guard") or {})
        self.model_weights = list(
            artifact.get("model_weights")
            or self.metadata.get("model_weights")
            or [1.0 for _ in self.models]
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    def _aligned_rows(self, chunks: list[list[dict[str, Any]]]) -> list[list[float]]:
        rows: list[list[float]] = []
        for chunk in chunks:
            features = chunk_features(chunk)
            features["hand_count"] = float(len(chunk))
            if not self.feature_names:
                self.feature_names = sorted(features)
            rows.append([float(features.get(name, 0.0)) for name in self.feature_names])
        return rows

    def _raw_model_scores(self, rows: list[list[float]]) -> list[float]:
        per_model: list[list[float]] = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(rows)
                per_model.append([self._clamp01(row[1]) for row in probabilities])
            elif hasattr(model, "decision_function"):
                decisions = model.decision_function(rows)
                per_model.append([self._sigmoid(value) for value in decisions])
            else:
                per_model.append([self._clamp01(value) for value in model.predict(rows)])

        weights = [max(0.0, float(value)) for value in self.model_weights[: len(per_model)]]
        if len(weights) != len(per_model) or sum(weights) <= 0.0:
            weights = [1.0 for _ in per_model]
        total_weight = sum(weights)

        scores: list[float] = []
        for row_index in range(len(rows)):
            value = sum(
                weight * model_scores[row_index]
                for weight, model_scores in zip(weights, per_model)
            ) / total_weight
            scores.append(self._clamp01(value))
        return scores

    def _apply_calibrator(self, scores: list[float]) -> list[float]:
        if not scores or self.calibrator is None:
            return [self._clamp01(value) for value in scores]
        if hasattr(self.calibrator, "predict_proba"):
            calibrated = self.calibrator.predict_proba([[float(value)] for value in scores])
            return [self._clamp01(row[1]) for row in calibrated]
        if hasattr(self.calibrator, "transform"):
            return [self._clamp01(value) for value in self.calibrator.transform(scores)]
        return [self._clamp01(value) for value in scores]

    def _apply_human_guard(self, scores: list[float]) -> list[float]:
        if not scores or not self.human_guard:
            return [self._clamp01(value) for value in scores]
        try:
            anchor = float(self.human_guard.get("anchor", 0.0))
            softness = max(float(self.human_guard.get("softness", 1.0)), 1e-6)
            strength = min(max(float(self.human_guard.get("strength", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            return [self._clamp01(value) for value in scores]
        if strength <= 0.0:
            return [self._clamp01(value) for value in scores]

        guarded: list[float] = []
        for value in scores:
            score = self._clamp01(value)
            human_like = 1.0 / (1.0 + math.exp((score - anchor) / softness))
            guarded.append(self._clamp01(score * (1.0 - strength * human_like)))
        return guarded

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if not chunks:
            return []
        rows = self._aligned_rows(chunks)
        raw_scores = self._raw_model_scores(rows)
        calibrated_scores = self._apply_calibrator(raw_scores)
        guarded_scores = self._apply_human_guard(calibrated_scores)
        return [round(self._clamp01(value), 6) for value in guarded_scores]

    def predict_chunk_score(self, chunk: list[dict[str, Any]]) -> float:
        scores = self.predict_chunk_scores([chunk])
        return scores[0] if scores else 0.5

    def debug_score_components(
        self,
        chunks: list[list[dict[str, Any]]],
    ) -> dict[str, list[float]]:
        if not chunks:
            return {}
        rows = self._aligned_rows(chunks)
        raw_scores = self._raw_model_scores(rows)
        calibrated_scores = self._apply_calibrator(raw_scores)
        final_scores = self._apply_human_guard(calibrated_scores)
        return {
            "raw_scores": [round(value, 6) for value in raw_scores],
            "calibrated_scores": [round(value, 6) for value in calibrated_scores],
            "final_scores": [round(value, 6) for value in final_scores],
        }

    def benchmark_latency(
        self,
        chunks: list[list[dict[str, Any]]],
        repeats: int = 5,
    ) -> dict[str, float]:
        if not chunks:
            return {"latency_per_chunk_ms": 0.0, "total_latency_ms": 0.0}
        repeats = max(1, int(repeats))
        started = time.perf_counter()
        for _ in range(repeats):
            self.predict_chunk_scores(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
        return {
            "latency_per_chunk_ms": elapsed_ms / max(len(chunks), 1),
            "total_latency_ms": elapsed_ms,
        }


HumanBaselineModel = Poker44Model
