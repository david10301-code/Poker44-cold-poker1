"""Live validator-query capture (operational, local-only, gitignored).

Persists the UNLABELED chunks validators send at inference = the real live
distribution, for unsupervised domain-adaptation / OOD diagnosis of the
benchmark->live gap. Captures INPUTS ONLY (plus this miner's own score); a live
query carries no ground-truth bot/human label, so nothing written here can serve
as a supervised training label.

Safety contract:
  * OFF by default. Enable with env POKER44_CAPTURE=1.
  * Size-capped per file (POKER44_CAPTURE_MAX_BYTES, default 250MB).
  * Thread-safe (append under a lock) and FAIL-SAFE: every path is wrapped so a
    capture error can never affect serving / scoring.
  * Output is gitignored and never leaves the box.

ATTESTATION: while these captures are used only for diagnosis they do NOT change
your training-data statement. The moment you feed them into training (even
unlabeled, for domain adaptation), update POKER44_MODEL_PRIVATE_DATA_ATTESTATION
truthfully.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

_LOCK = threading.Lock()
# Shared capture dir at the GPU_projects/Poker44 root (parents[2] of this file:
# .../Poker44/Poker44_v1/poker44_ml/live_capture.py -> .../Poker44), so every
# co-located miner/repo writes to one place. Override with POKER44_CAPTURE_DIR.
_DIR = Path(
    os.getenv("POKER44_CAPTURE_DIR")
    or Path(__file__).resolve().parents[2] / "live_capture"
)
_MAX_BYTES = int(os.getenv("POKER44_CAPTURE_MAX_BYTES", str(250 * 1024 * 1024)))
# Per-process state: resolved output path + a latch once the size cap is hit.
_state: dict[str, Any] = {"path": None, "full": False}


def enabled() -> bool:
    return os.getenv("POKER44_CAPTURE", "0") == "1"


def capture(
    chunks: Sequence[Sequence[dict]],
    scores: Sequence[float],
    miner_id: Any,
    validator: Any,
) -> None:
    """Append one JSONL record per chunk: {t, v, uid, n, score, chunk}.

    Input-only (no labels). Never raises — capture must not affect serving.
    """
    if not enabled() or _state["full"] or not chunks:
        return
    try:
        _DIR.mkdir(exist_ok=True)
        if _state["path"] is None:
            _state["path"] = _DIR / f"capture_{str(miner_id)[:16]}.jsonl"
        path: Path = _state["path"]
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            _state["full"] = True
            return
        ts = round(time.time(), 2)
        vtag = str(validator or "")[:8]
        uid = str(miner_id)
        lines = []
        for chunk, score in zip(chunks, scores):
            try:
                s = round(float(score), 5)
            except (TypeError, ValueError):
                s = None
            lines.append(
                json.dumps(
                    {"t": ts, "v": vtag, "uid": uid, "n": len(chunk), "score": s, "chunk": chunk},
                    separators=(",", ":"),
                )
            )
        if not lines:
            return
        payload = "\n".join(lines) + "\n"
        with _LOCK:
            with open(path, "a") as handle:
                handle.write(payload)
    except Exception:
        # Capture must NEVER affect serving.
        pass
