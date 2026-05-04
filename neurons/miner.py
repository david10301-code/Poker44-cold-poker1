"""Poker44 miner entrypoint backed by a shared scoring pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner.config import MinerRuntimeConfig
from poker44.miner.manifest import manifest_defaults_for_pipeline
from poker44.miner.pipeline import ChunkScoringPipeline
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """Miner entrypoint that delegates scoring to the shared runtime pipeline."""

    def __init__(self, config=None):
        super().__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        self.runtime_config = MinerRuntimeConfig.from_env(repo_root)
        self.scoring_pipeline = ChunkScoringPipeline(self.runtime_config)

        bt.logging.info(
            f"🤖 Poker44 miner started with backend={self.scoring_pipeline.backend}"
        )
        if self.scoring_pipeline.load_error:
            bt.logging.warning(
                f"Local model path unavailable, continuing with fallback backend: "
                f"{self.scoring_pipeline.load_error}"
            )

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults=manifest_defaults_for_pipeline(self.scoring_pipeline),
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Inference backend={self.scoring_pipeline.backend} "
            f"model_path={self.runtime_config.model_path}"
        )
        if self.scoring_pipeline.predictor is not None:
            metadata = self.scoring_pipeline.predictor_metadata
            bt.logging.info(
                f"Model metadata: feature_count="
                f"{len(self.scoring_pipeline.predictor.feature_names)} "
                f"calibration={metadata.get('calibration', 'unknown')} "
                f"framework={metadata.get('framework', 'unknown')}"
            )
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = [list(chunk or []) for chunk in (synapse.chunks or [])]
        started = time.perf_counter()
        result = self.scoring_pipeline.score_chunks(chunks)

        synapse.risk_scores = result.scores
        synapse.predictions = [score >= 0.5 for score in result.scores]
        synapse.model_manifest = dict(self.model_manifest)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        message = (
            f"Scored {len(chunks)} chunks with backend={result.backend} "
            f"elapsed_ms={elapsed_ms:.2f} "
            f"chunk_size_range="
            f"{[min(result.raw_chunk_sizes), max(result.raw_chunk_sizes)] if result.raw_chunk_sizes else [0, 0]} "
            f"eval_chunk_size_range="
            f"{[min(result.eval_chunk_sizes), max(result.eval_chunk_sizes)] if result.eval_chunk_sizes else [0, 0]}"
        )
        if self.runtime_config.query_log_preview:
            message += (
                f" score_preview={result.scores[:5]} "
                f"prediction_preview={synapse.predictions[:5]}"
            )
        bt.logging.info(message)
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
