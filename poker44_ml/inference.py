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


class HumanBaselineModel:
    """Runtime wrapper for a human-only baseline model."""

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
        self.model = artifact["model"]
        self.feature_names = list(artifact.get("feature_names") or [])
        self.metadata = dict(artifact.get("metadata") or {})
        self.score_quantiles = dict(self.metadata.get("score_quantiles") or {})

    def _aligned_rows(self, chunks: list[list[dict[str, Any]]]) -> list[list[float]]:
        rows: list[list[float]] = []
        for chunk in chunks:
            feats = chunk_features(chunk)
            if not self.feature_names:
                self.feature_names = sorted(feats)
            rows.append([float(feats.get(name, 0.0)) for name in self.feature_names])
        return rows

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

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
            return round(0.20 + 0.14 * self._clamp01(scaled), 6)
        if anomaly_score <= q995:
            scaled = (anomaly_score - q99) / max(q995 - q99, 1e-9)
            return round(0.34 + 0.11 * self._clamp01(scaled), 6)
        if anomaly_score <= q999:
            scaled = (anomaly_score - q995) / max(q999 - q995, 1e-9)
            return round(0.45 + 0.038 * self._clamp01(scaled), 6)
        if anomaly_score <= q9995:
            scaled = (anomaly_score - q999) / max(q9995 - q999, 1e-9)
            return round(0.488 + 0.011 * self._clamp01(scaled), 6)
        scaled = (anomaly_score - q9995) / max(q9995 - q999, 1e-9)
        return round(0.50 + 0.50 * self._clamp01(math.sqrt(max(0.0, scaled))), 6)

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if not chunks:
            return []
        rows = self._aligned_rows(chunks)
        raw_scores = self.model.score_samples(rows)
        anomalies = [-float(value) for value in raw_scores]
        return [self._risk_from_anomaly(score) for score in anomalies]

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
