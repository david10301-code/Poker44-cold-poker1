from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from poker44_ml.features import chunk_features

try:
    import joblib
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    joblib = None


class Poker44Model:
    """Runtime wrapper for supervised benchmark or legacy anomaly artifacts."""

    def __init__(self, model_path: str | Path):
        if joblib is None:
            raise RuntimeError(
                "joblib is required to load the Poker44 model artifact. "
                "Install runtime dependencies first."
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        artifact = joblib.load(self.model_path)
        model_list = artifact.get("models")
        if model_list:
            self.models = list(model_list)
        elif artifact.get("model") is not None:
            self.models = [artifact["model"]]
        else:
            raise RuntimeError("Model artifact is missing model content.")

        self.feature_names = list(artifact.get("feature_names") or [])
        self.metadata = dict(artifact.get("metadata") or {})
        self.score_quantiles = dict(self.metadata.get("score_quantiles") or {})
        self.score_remap = dict(self.metadata.get("score_remap") or {})
        self.probability_calibrator = artifact.get("probability_calibrator")
        self.task_type = str(
            self.metadata.get(
                "task_type",
                "human-baseline" if self.score_quantiles else "supervised-benchmark",
            )
        )

    def _aligned_rows(self, chunks: list[list[dict[str, Any]]]) -> list[list[float]]:
        rows: list[list[float]] = []
        for chunk in chunks:
            feats = chunk_features(chunk)
            feats["hand_count"] = float(len(chunk))
            if not self.feature_names:
                self.feature_names = sorted(feats)
            rows.append([float(feats.get(name, 0.0)) for name in self.feature_names])
        return rows

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _risk_from_anomaly(self, anomaly_score: float) -> float:
        q50 = float(self.score_quantiles.get("q50", 0.0))
        q90 = float(self.score_quantiles.get("q90", q50 + 1.0))
        q95 = float(self.score_quantiles.get("q95", q50 + 1.0))
        q99 = float(self.score_quantiles.get("q99", q95 + 1.0))
        q995 = float(self.score_quantiles.get("q995", q99 + 1.0))
        q999 = float(self.score_quantiles.get("q999", q995 + 1.0))
        q9995 = float(self.score_quantiles.get("q9995", q999 + 1.0))

        if q90 <= q50:
            q90 = q50 + 1.0
        if q95 <= q90:
            q95 = q90 + 1e-6
        if q99 <= q95:
            q99 = q95 + 1e-6
        if q995 <= q99:
            q995 = q99 + 1e-6
        if q999 <= q995:
            q999 = q995 + 1e-6
        if q9995 <= q999:
            q9995 = q999 + 1e-6

        if anomaly_score <= q50:
            return 0.0
        if anomaly_score <= q90:
            scaled = (anomaly_score - q50) / max(q90 - q50, 1e-9)
            return round(0.10 * self._clamp01(math.sqrt(max(0.0, scaled))), 6)
        if anomaly_score <= q95:
            scaled = (anomaly_score - q90) / max(q95 - q90, 1e-9)
            return round(0.10 + 0.10 * self._clamp01(scaled), 6)
        if anomaly_score <= q99:
            scaled = (anomaly_score - q95) / max(q99 - q95, 1e-9)
            return round(0.20 + 0.08 * self._clamp01(scaled), 6)
        if anomaly_score <= q995:
            scaled = (anomaly_score - q99) / max(q995 - q99, 1e-9)
            return round(0.28 + 0.05 * self._clamp01(scaled), 6)
        if anomaly_score <= q999:
            scaled = (anomaly_score - q995) / max(q999 - q995, 1e-9)
            return round(0.33 + 0.018 * self._clamp01(scaled), 6)
        if anomaly_score <= q9995:
            scaled = (anomaly_score - q999) / max(q9995 - q999, 1e-9)
            return round(0.348 + 0.012 * self._clamp01(scaled), 6)
        scaled = (anomaly_score - q9995) / max(q9995 - q999, 1e-9)
        return round(0.36 + 0.64 * self._clamp01(math.sqrt(max(0.0, scaled))), 6)

    def _predict_supervised_scores(self, rows: list[list[float]]) -> list[float]:
        if not rows:
            return []

        per_model_scores: list[list[float]] = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(rows)
                per_model_scores.append([float(row[1]) for row in probabilities])
                continue
            if hasattr(model, "decision_function"):
                decisions = model.decision_function(rows)
                per_model_scores.append(
                    [self._sigmoid(float(value)) for value in decisions]
                )
                continue
            predictions = model.predict(rows)
            per_model_scores.append([float(value) for value in predictions])

        averaged: list[float] = []
        for index in range(len(rows)):
            averaged.append(
                self._clamp01(
                    sum(scores[index] for scores in per_model_scores)
                    / max(len(per_model_scores), 1)
                )
            )
        if self.probability_calibrator is not None and hasattr(
            self.probability_calibrator, "transform"
        ):
            calibrated = self.probability_calibrator.transform(averaged)
            return [round(self._clamp01(value), 6) for value in calibrated]
        if self.probability_calibrator is not None and hasattr(
            self.probability_calibrator, "predict_proba"
        ):
            calibrated = self.probability_calibrator.predict_proba(
                [[float(value)] for value in averaged]
            )
            return [round(self._clamp01(row[1]), 6) for row in calibrated]
        return [round(value, 6) for value in self._apply_supervised_score_remap(averaged)]

    def _apply_supervised_score_remap(self, probabilities: list[float]) -> list[float]:
        if not probabilities or not self.score_remap:
            return [self._clamp01(value) for value in probabilities]

        threshold = float(self.score_remap.get("threshold", 0.5))
        threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
        adjusted: list[float] = []
        for value in probabilities:
            score = self._clamp01(value)
            if score <= threshold:
                mapped = 0.5 * score / threshold
            else:
                mapped = 0.5 + 0.5 * (score - threshold) / (1.0 - threshold)
            adjusted.append(self._clamp01(mapped))
        return adjusted

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if not chunks:
            return []
        rows = self._aligned_rows(chunks)
        if self.task_type == "human-baseline" and hasattr(self.models[0], "score_samples"):
            raw_scores = self.models[0].score_samples(rows)
            anomalies = [-float(value) for value in raw_scores]
            return [self._risk_from_anomaly(score) for score in anomalies]
        return self._predict_supervised_scores(rows)

    def predict_chunk_score(self, chunk: list[dict[str, Any]]) -> float:
        scores = self.predict_chunk_scores([chunk])
        return scores[0] if scores else 0.5

    def benchmark_latency(
        self,
        chunks: list[list[dict[str, Any]]],
        repeats: int = 5,
    ) -> dict[str, float]:
        if not chunks:
            return {"latency_per_chunk_ms": 0.0, "total_latency_ms": 0.0}

        repeats = max(1, repeats)
        started = time.perf_counter()
        for _ in range(repeats):
            self.predict_chunk_scores(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
        return {
            "latency_per_chunk_ms": elapsed_ms / max(len(chunks), 1),
            "total_latency_ms": elapsed_ms,
        }


HumanBaselineModel = Poker44Model
