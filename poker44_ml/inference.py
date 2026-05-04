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
    """Thin runtime wrapper around a pre-trained calibrated classifier."""

    def __init__(self, model_path: str | Path):
        if joblib is None:
            raise RuntimeError(
                "joblib is required to load the Poker44 model artifact. "
                "Install the training/runtime dependencies first."
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if self.model_path.stat().st_size == 0:
            raise RuntimeError(
                f"Model artifact is empty: {self.model_path}. "
                "Generate bot data and retrain the miner model before starting the miner."
            )

        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict):
            self.model = artifact["model"]
            self.feature_names = list(artifact.get("feature_names") or [])
            self.metadata = dict(artifact.get("metadata") or {})
        else:
            self.model = artifact
            self.feature_names = []
            self.metadata = {}

    def _aligned_rows(self, chunks: list[list[dict[str, Any]]]) -> list[list[float]]:
        rows: list[list[float]] = []
        
        for chunk in chunks:
            feats = chunk_features(chunk)
            
            # If no feature names loaded from artifact, infer from first chunk
            if not self.feature_names:
                ordered = sorted(feats)
                self.feature_names = ordered
                if not self.feature_names:
                    raise RuntimeError(
                        f"Failed to extract any features from chunk. "
                        f"Got empty feature dict: {feats}"
                    )
            
            # Validate that all expected features are present in this chunk
            # Missing features indicate data corruption or model-data mismatch
            missing_features = set(self.feature_names) - set(feats.keys())
            if missing_features:
                raise RuntimeError(
                    f"Feature mismatch in chunk inference: "
                    f"expected {len(self.feature_names)} features ({self.feature_names}), "
                    f"but missing {len(missing_features)}: {sorted(missing_features)}. "
                    f"This indicates data formatting or model-training mismatch. "
                    f"Got features: {sorted(feats.keys())}"
                )
            
            # Align features to training feature order
            rows.append([float(feats.get(name, 0.0)) for name in self.feature_names])
        
        return rows

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if not chunks:
            return []

        rows = self._aligned_rows(chunks)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(rows)
            return [float(row[1]) for row in probs]
        if hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(rows)
            return [1.0 / (1.0 + math.exp(-float(value))) for value in raw]
        preds = self.model.predict(rows)
        return [float(value) for value in preds]

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
