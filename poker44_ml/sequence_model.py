"""Chunk-level Set Transformer for Poker44 bot detection.

This module adds a sixth base learner to the stacked ensemble: a small
hierarchical Transformer that consumes the *raw* action tokens of each hand
instead of the aggregated per-chunk features.

Why this exists
---------------
The feature-based learners in ``train_model_v2`` operate on ~390 hand-aggregate
statistics that throw away action **order**. Bots are most distinguishable in
their temporal patterns: sizing rhythms, postflop response sequences, and
preflop priors. A small Transformer over the (action_type, street, actor,
amount-bucket, pot-after) tokens captures that signal cheaply.

Architecture
------------
For each chunk (list of hands):
    1. **Hand encoder**: action tokens are embedded
       (categorical embeddings + linear projection of continuous features +
       learned action-position embedding), then passed through a 2-layer
       Transformer encoder with key-padding masks. The hand embedding is
       obtained by an attention-pool over the action sequence with a learned
       query token (similar to a Set Transformer PMA).
    2. **Chunk encoder**: hand embeddings are passed through a 1-layer
       Transformer encoder with hand-level key-padding masks (permutation
       invariant once positional encodings are turned off, which we do here).
       A second attention-pool produces the chunk embedding.
    3. **Head**: 2-layer MLP outputs one logit per chunk; sigmoid gives
       ``P(bot | chunk)``.

The exposed wrapper :class:`SequenceModelWrapper` is a sklearn-style estimator
with ``fit(chunks, y, sample_weight=None)`` and ``predict_proba(chunks)`` that
plays nicely with the stacked ensemble pipeline. It pickles cleanly via
joblib (state_dict + config) and runs CPU-only by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for the chunk-level sequence model. "
        "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


# --- tokenizer constants ----------------------------------------------------

_ACTION_TYPE_VOCAB: Dict[str, int] = {
    "<pad>": 0,
    "small_blind": 1,
    "big_blind": 2,
    "ante": 3,
    "check": 4,
    "call": 5,
    "bet": 6,
    "raise": 7,
    "fold": 8,
    "all_in": 9,
    "other": 10,
}
_STREET_VOCAB: Dict[str, int] = {
    "<pad>": 0,
    "preflop": 1,
    "flop": 2,
    "turn": 3,
    "river": 4,
    "": 5,
}
_ACTOR_ROLE_PAD = 0
_ACTOR_ROLE_HERO = 1
_ACTOR_ROLE_BUTTON = 2
_ACTOR_ROLE_OTHER = 3


MAX_ACTIONS_PER_HAND = 16
MAX_HANDS_PER_CHUNK = 16
CONT_DIM = 6  # amount_bb, raise_to_bb, call_to_bb, pot_before_bb, pot_after_bb, pot_delta_bb


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _action_type_id(value: Any) -> int:
    raw = str(value or "other").strip().lower()
    if raw in _ACTION_TYPE_VOCAB:
        return _ACTION_TYPE_VOCAB[raw]
    if "raise" in raw:
        return _ACTION_TYPE_VOCAB["raise"]
    if "bet" in raw:
        return _ACTION_TYPE_VOCAB["bet"]
    if "call" in raw:
        return _ACTION_TYPE_VOCAB["call"]
    if "check" in raw:
        return _ACTION_TYPE_VOCAB["check"]
    if "fold" in raw or raw == "muck":
        return _ACTION_TYPE_VOCAB["fold"]
    return _ACTION_TYPE_VOCAB["other"]


def _street_id(value: Any) -> int:
    raw = str(value or "").strip().lower()
    return _STREET_VOCAB.get(raw, _STREET_VOCAB[""])


def _actor_role(actor_seat: int, hero_seat: int, button_seat: int) -> int:
    if actor_seat <= 0:
        return _ACTOR_ROLE_OTHER
    if hero_seat and actor_seat == hero_seat:
        return _ACTOR_ROLE_HERO
    if button_seat and actor_seat == button_seat:
        return _ACTOR_ROLE_BUTTON
    return _ACTOR_ROLE_OTHER


def _to_bb(value: Any, bb: float) -> float:
    if bb <= 0:
        return _safe_float(value, 0.0)
    return _safe_float(value, 0.0) / bb


def encode_hand(hand: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a single hand payload into padded token tensors."""
    metadata = hand.get("metadata") or {}
    actions = hand.get("actions") or []
    hero_seat = int(metadata.get("hero_seat") or 0)
    button_seat = int(metadata.get("button_seat") or 0)
    bb = _safe_float(metadata.get("bb"), 0.02) or 0.02

    action_type = np.zeros(MAX_ACTIONS_PER_HAND, dtype=np.int64)
    street = np.zeros(MAX_ACTIONS_PER_HAND, dtype=np.int64)
    actor_role = np.zeros(MAX_ACTIONS_PER_HAND, dtype=np.int64)
    cont = np.zeros((MAX_ACTIONS_PER_HAND, CONT_DIM), dtype=np.float32)
    mask = np.zeros(MAX_ACTIONS_PER_HAND, dtype=np.bool_)

    n_actions = min(len(actions), MAX_ACTIONS_PER_HAND)
    for idx in range(n_actions):
        action = actions[idx]
        if not isinstance(action, dict):
            continue
        action_type[idx] = _action_type_id(action.get("action_type"))
        street[idx] = _street_id(action.get("street"))
        actor_role[idx] = _actor_role(
            int(action.get("actor_seat") or 0), hero_seat, button_seat
        )
        amount_bb = _safe_float(action.get("normalized_amount_bb"), 0.0)
        if amount_bb == 0.0:
            amount_bb = _to_bb(action.get("amount"), bb)
        raise_to_bb = _to_bb(action.get("raise_to"), bb)
        call_to_bb = _to_bb(action.get("call_to"), bb)
        pot_before_bb = _to_bb(action.get("pot_before"), bb)
        pot_after_bb = _to_bb(action.get("pot_after"), bb)
        cont[idx, 0] = math.log1p(max(amount_bb, 0.0))
        cont[idx, 1] = math.log1p(max(raise_to_bb, 0.0))
        cont[idx, 2] = math.log1p(max(call_to_bb, 0.0))
        cont[idx, 3] = math.log1p(max(pot_before_bb, 0.0))
        cont[idx, 4] = math.log1p(max(pot_after_bb, 0.0))
        cont[idx, 5] = math.log1p(max(pot_after_bb - pot_before_bb, 0.0))
        mask[idx] = True

    return {
        "action_type": action_type,
        "street": street,
        "actor_role": actor_role,
        "cont": cont,
        "mask": mask,
    }


def encode_chunk(chunk: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Pad a chunk into fixed-size hand x action tensors with masks."""
    action_type = np.zeros((MAX_HANDS_PER_CHUNK, MAX_ACTIONS_PER_HAND), dtype=np.int64)
    street = np.zeros((MAX_HANDS_PER_CHUNK, MAX_ACTIONS_PER_HAND), dtype=np.int64)
    actor_role = np.zeros((MAX_HANDS_PER_CHUNK, MAX_ACTIONS_PER_HAND), dtype=np.int64)
    cont = np.zeros(
        (MAX_HANDS_PER_CHUNK, MAX_ACTIONS_PER_HAND, CONT_DIM), dtype=np.float32
    )
    action_mask = np.zeros(
        (MAX_HANDS_PER_CHUNK, MAX_ACTIONS_PER_HAND), dtype=np.bool_
    )
    hand_mask = np.zeros(MAX_HANDS_PER_CHUNK, dtype=np.bool_)

    n_hands = min(len(chunk), MAX_HANDS_PER_CHUNK)
    for hand_idx in range(n_hands):
        hand = chunk[hand_idx]
        if not isinstance(hand, dict):
            continue
        encoded = encode_hand(hand)
        action_type[hand_idx] = encoded["action_type"]
        street[hand_idx] = encoded["street"]
        actor_role[hand_idx] = encoded["actor_role"]
        cont[hand_idx] = encoded["cont"]
        action_mask[hand_idx] = encoded["mask"]
        hand_mask[hand_idx] = bool(encoded["mask"].any())

    return {
        "action_type": action_type,
        "street": street,
        "actor_role": actor_role,
        "cont": cont,
        "action_mask": action_mask,
        "hand_mask": hand_mask,
    }


# --- torch dataset ---------------------------------------------------------


class _ChunkDataset(Dataset):
    def __init__(
        self,
        chunks: Sequence[Sequence[Dict[str, Any]]],
        labels: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.encoded: List[Dict[str, np.ndarray]] = [
            encode_chunk(chunk) for chunk in chunks
        ]
        self.labels = (
            np.asarray(labels, dtype=np.float32) if labels is not None else None
        )
        self.weights = (
            np.asarray(weights, dtype=np.float32) if weights is not None else None
        )

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], float, float]:
        item = self.encoded[idx]
        label = float(self.labels[idx]) if self.labels is not None else 0.0
        weight = float(self.weights[idx]) if self.weights is not None else 1.0
        return item, label, weight


def _collate(
    batch: List[Tuple[Dict[str, np.ndarray], float, float]]
) -> Dict[str, torch.Tensor]:
    keys = ("action_type", "street", "actor_role", "cont", "action_mask", "hand_mask")
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = np.stack([item[0][key] for item in batch], axis=0)
        if key in ("cont",):
            out[key] = torch.from_numpy(stacked).float()
        elif key in ("action_mask", "hand_mask"):
            out[key] = torch.from_numpy(stacked).bool()
        else:
            out[key] = torch.from_numpy(stacked).long()
    out["label"] = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    out["weight"] = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return out


# --- model ------------------------------------------------------------------


@dataclass
class SequenceModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_action_layers: int = 2
    n_hand_layers: int = 1
    dropout: float = 0.1
    ff_mult: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "n_action_layers": int(self.n_action_layers),
            "n_hand_layers": int(self.n_hand_layers),
            "dropout": float(self.dropout),
            "ff_mult": int(self.ff_mult),
        }


class _AttentionPool(nn.Module):
    """Single-query attention pool (Set Transformer PMA with k=1)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch = x.size(0)
        query = self.query.expand(batch, 1, -1)
        all_padded = (
            key_padding_mask.all(dim=1) if key_padding_mask is not None else None
        )
        safe_mask = key_padding_mask
        if safe_mask is not None and all_padded is not None and all_padded.any():
            safe_mask = safe_mask.clone()
            safe_mask[all_padded, 0] = False
        attn_out, _ = self.attn(
            query=query,
            key=x,
            value=x,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        pooled = self.norm(attn_out.squeeze(1))
        if all_padded is not None and all_padded.any():
            pooled = pooled.masked_fill(all_padded.unsqueeze(-1), 0.0)
        return pooled


class ChunkSetTransformer(nn.Module):
    """Hierarchical action → hand → chunk Transformer."""

    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.action_type_emb = nn.Embedding(
            len(_ACTION_TYPE_VOCAB), d_model, padding_idx=0
        )
        self.street_emb = nn.Embedding(len(_STREET_VOCAB), d_model, padding_idx=0)
        self.actor_emb = nn.Embedding(4, d_model, padding_idx=_ACTOR_ROLE_PAD)
        self.action_pos_emb = nn.Embedding(MAX_ACTIONS_PER_HAND, d_model)
        self.cont_proj = nn.Linear(CONT_DIM, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        action_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=d_model * config.ff_mult,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(
            action_layer, num_layers=config.n_action_layers
        )
        self.action_pool = _AttentionPool(d_model, config.n_heads, config.dropout)

        hand_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=d_model * config.ff_mult,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.hand_encoder = nn.TransformerEncoder(
            hand_layer, num_layers=config.n_hand_layers
        )
        self.chunk_pool = _AttentionPool(d_model, config.n_heads, config.dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        action_type: torch.Tensor,
        street: torch.Tensor,
        actor_role: torch.Tensor,
        cont: torch.Tensor,
        action_mask: torch.Tensor,
        hand_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, hands, actions = action_type.shape
        position_ids = (
            torch.arange(actions, device=action_type.device)
            .unsqueeze(0)
            .expand(batch * hands, actions)
        )
        flat_action_type = action_type.reshape(batch * hands, actions)
        flat_street = street.reshape(batch * hands, actions)
        flat_actor = actor_role.reshape(batch * hands, actions)
        flat_cont = cont.reshape(batch * hands, actions, -1)
        flat_action_mask = action_mask.reshape(batch * hands, actions)

        embed = (
            self.action_type_emb(flat_action_type)
            + self.street_emb(flat_street)
            + self.actor_emb(flat_actor)
            + self.action_pos_emb(position_ids)
            + self.cont_proj(flat_cont)
        )
        embed = self.input_norm(embed)
        embed = self.input_dropout(embed)

        key_padding = ~flat_action_mask
        encoded = self.action_encoder(embed, src_key_padding_mask=key_padding)
        hand_emb = self.action_pool(encoded, key_padding_mask=key_padding)
        hand_emb = hand_emb.reshape(batch, hands, -1)

        hand_kp = ~hand_mask
        encoded_hands = self.hand_encoder(hand_emb, src_key_padding_mask=hand_kp)
        chunk_emb = self.chunk_pool(encoded_hands, key_padding_mask=hand_kp)
        logit = self.head(chunk_emb).squeeze(-1)
        return logit


# --- sklearn-style wrapper -------------------------------------------------


@dataclass
class SequenceModelWrapper:
    """sklearn-style wrapper around :class:`ChunkSetTransformer`.

    Use ``fit(chunks, y, sample_weight=...)`` with **raw chunk payloads** (lists
    of hand dicts), not feature rows. Use ``predict_proba(chunks)`` to get an
    ``Nx2`` array compatible with the rest of the stacking pipeline.
    """

    config: SequenceModelConfig = field(default_factory=SequenceModelConfig)
    n_epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    early_stopping_patience: int = 3
    seed: int = 42
    device: str = "cpu"
    verbose: bool = False
    _model_state: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.config = (
            self.config
            if isinstance(self.config, SequenceModelConfig)
            else SequenceModelConfig(**dict(self.config))
        )

    def _new_model(self) -> ChunkSetTransformer:
        torch.manual_seed(int(self.seed))
        return ChunkSetTransformer(self.config).to(self.device)

    def fit(
        self,
        chunks: Sequence[Sequence[Dict[str, Any]]],
        y: Sequence[int],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "SequenceModelWrapper":
        labels = np.asarray(y, dtype=np.float32)
        weights = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None
            else np.ones(len(chunks), dtype=np.float32)
        )

        rng = np.random.default_rng(int(self.seed))
        order = rng.permutation(len(chunks))
        val_size = max(int(round(self.val_fraction * len(chunks))), 1)
        val_idx = order[:val_size]
        train_idx = order[val_size:]

        train_chunks = [chunks[i] for i in train_idx]
        train_labels = labels[train_idx]
        train_weights = weights[train_idx]
        val_chunks = [chunks[i] for i in val_idx]
        val_labels = labels[val_idx]
        val_weights = weights[val_idx]

        train_ds = _ChunkDataset(train_chunks, train_labels, train_weights)
        val_ds = _ChunkDataset(val_chunks, val_labels, val_weights)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

        model = self._new_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience = 0

        for epoch in range(int(self.n_epochs)):
            model.train()
            total_train = 0.0
            n_train = 0
            for batch in train_loader:
                logits = model(
                    action_type=batch["action_type"].to(self.device),
                    street=batch["street"].to(self.device),
                    actor_role=batch["actor_role"].to(self.device),
                    cont=batch["cont"].to(self.device),
                    action_mask=batch["action_mask"].to(self.device),
                    hand_mask=batch["hand_mask"].to(self.device),
                )
                label_t = batch["label"].to(self.device)
                weight_t = batch["weight"].to(self.device)
                raw_loss = loss_fn(logits, label_t)
                loss = (raw_loss * weight_t).sum() / weight_t.sum().clamp(min=1e-6)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train += float(loss.item()) * label_t.size(0)
                n_train += label_t.size(0)

            val_loss = self._evaluate_loss(model, val_loader, loss_fn)
            if self.verbose:
                print(
                    f"    seq epoch {epoch + 1}/{self.n_epochs} "
                    f"train_loss={total_train / max(n_train, 1):.4f} "
                    f"val_loss={val_loss:.4f}"
                )
            if val_loss + 1e-5 < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    key: tensor.detach().clone() for key, tensor in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= int(self.early_stopping_patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model_state = {
            "state_dict": {key: tensor.cpu() for key, tensor in model.state_dict().items()},
            "config": self.config.to_dict(),
        }
        return self

    def _evaluate_loss(
        self,
        model: ChunkSetTransformer,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> float:
        if len(loader.dataset) == 0:  # type: ignore[arg-type]
            return float("inf")
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    action_type=batch["action_type"].to(self.device),
                    street=batch["street"].to(self.device),
                    actor_role=batch["actor_role"].to(self.device),
                    cont=batch["cont"].to(self.device),
                    action_mask=batch["action_mask"].to(self.device),
                    hand_mask=batch["hand_mask"].to(self.device),
                )
                raw_loss = loss_fn(logits, batch["label"].to(self.device))
                w = batch["weight"].to(self.device)
                total += float((raw_loss * w).sum().item())
                count += int(w.sum().item())
        return total / max(count, 1)

    def predict_proba(
        self, chunks: Sequence[Sequence[Dict[str, Any]]]
    ) -> np.ndarray:
        if self._model_state is None:
            raise RuntimeError("SequenceModelWrapper.predict_proba called before fit.")
        model = ChunkSetTransformer(self.config).to(self.device)
        model.load_state_dict(self._model_state["state_dict"])
        model.eval()
        ds = _ChunkDataset(list(chunks))
        loader = DataLoader(
            ds,
            batch_size=max(self.batch_size, 1),
            shuffle=False,
            collate_fn=_collate,
        )
        out: List[float] = []
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    action_type=batch["action_type"].to(self.device),
                    street=batch["street"].to(self.device),
                    actor_role=batch["actor_role"].to(self.device),
                    cont=batch["cont"].to(self.device),
                    action_mask=batch["action_mask"].to(self.device),
                    hand_mask=batch["hand_mask"].to(self.device),
                )
                out.extend(torch.sigmoid(logits).cpu().tolist())
        arr = np.asarray(out, dtype=np.float64)
        arr = np.clip(arr, 0.0, 1.0)
        return np.stack([1.0 - arr, arr], axis=1)

    def predict_chunk_scores(
        self, chunks: Sequence[Sequence[Dict[str, Any]]]
    ) -> List[float]:
        return self.predict_proba(chunks)[:, 1].tolist()

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "val_fraction": float(self.val_fraction),
            "early_stopping_patience": int(self.early_stopping_patience),
            "seed": int(self.seed),
            "device": str(self.device),
            "verbose": bool(self.verbose),
            "_model_state": self._model_state,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.config = SequenceModelConfig(**state["config"])
        self.n_epochs = int(state["n_epochs"])
        self.batch_size = int(state["batch_size"])
        self.learning_rate = float(state["learning_rate"])
        self.weight_decay = float(state["weight_decay"])
        self.val_fraction = float(state["val_fraction"])
        self.early_stopping_patience = int(state["early_stopping_patience"])
        self.seed = int(state["seed"])
        self.device = str(state["device"])
        self.verbose = bool(state["verbose"])
        self._model_state = state.get("_model_state")
