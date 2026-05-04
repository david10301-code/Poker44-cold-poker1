"""Manifest defaults for miner runtime backends."""

from __future__ import annotations

from typing import Any

from poker44.miner.pipeline import ChunkScoringPipeline


def manifest_defaults_for_pipeline(
    pipeline: ChunkScoringPipeline,
) -> dict[str, Any]:
    if pipeline.predictor is not None:
        return {
            "model_name": "poker44-xgb-calibrated",
            "model_version": "1",
            "framework": pipeline.predictor_metadata.get(
                "framework", "xgboost+sklearn"
            ),
            "license": "MIT",
            "repo_url": "https://github.com/Poker44/Poker44-subnet",
            "notes": "Chunk-level tabular model with calibrated probabilities.",
            "open_source": True,
            "inference_mode": "remote",
            "training_data_statement": (
                "Trained on public human corpus plus offline-generated bot hands."
            ),
            "training_data_sources": ["public_human_corpus", "generated_bot_hands"],
            "private_data_attestation": (
                "This miner does not train on validator-only evaluation data."
            ),
        }
    return {
        "model_name": "poker44-reference-heuristic",
        "model_version": "1",
        "framework": "python-heuristic",
        "license": "MIT",
        "repo_url": "https://github.com/Poker44/Poker44-subnet",
        "notes": "Reference heuristic miner shipped with the Poker44 subnet.",
        "open_source": True,
        "inference_mode": "remote",
        "training_data_statement": (
            "Reference heuristic miner. No training step. Uses only runtime chunk features."
        ),
        "training_data_sources": ["none"],
        "private_data_attestation": (
            "This miner does not train on validator-only evaluation data."
        ),
    }
