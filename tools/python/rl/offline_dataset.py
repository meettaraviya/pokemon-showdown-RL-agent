"""Offline dataset representation and serialization for batch RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OfflineBatch:
    """OfflineBatch helper type/container used by the RL self-play tooling."""
    scalar: np.ndarray
    party: np.ndarray
    action_vectors: np.ndarray
    action_mask: np.ndarray
    action_index: np.ndarray
    reward: np.ndarray
    next_scalar: np.ndarray
    next_party: np.ndarray
    next_action_vectors: np.ndarray
    next_action_mask: np.ndarray
    done: np.ndarray

    @property
    def size(self) -> int:
        return int(self.scalar.shape[0])


class OfflineDatasetBuilder:
    """OfflineDatasetBuilder helper type/container used by the RL self-play tooling."""
    def __init__(self) -> None:
        self.scalar: list[np.ndarray] = []
        self.party: list[np.ndarray] = []
        self.action_vectors: list[np.ndarray] = []
        self.action_mask: list[np.ndarray] = []
        self.action_index: list[int] = []
        self.reward: list[float] = []
        self.next_scalar: list[np.ndarray] = []
        self.next_party: list[np.ndarray] = []
        self.next_action_vectors: list[np.ndarray] = []
        self.next_action_mask: list[np.ndarray] = []
        self.done: list[bool] = []

    def append(
        self,
        *,
        scalar: np.ndarray,
        party: np.ndarray,
        action_vectors: np.ndarray,
        action_mask: np.ndarray,
        action_index: int,
        reward: float,
        next_scalar: np.ndarray,
        next_party: np.ndarray,
        next_action_vectors: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.scalar.append(scalar.astype(np.float32, copy=False))
        self.party.append(party.astype(np.float32, copy=False))
        self.action_vectors.append(action_vectors.astype(np.float32, copy=False))
        self.action_mask.append(action_mask.astype(np.bool_, copy=False))
        self.action_index.append(int(action_index))
        self.reward.append(float(reward))
        self.next_scalar.append(next_scalar.astype(np.float32, copy=False))
        self.next_party.append(next_party.astype(np.float32, copy=False))
        self.next_action_vectors.append(next_action_vectors.astype(np.float32, copy=False))
        self.next_action_mask.append(next_action_mask.astype(np.bool_, copy=False))
        self.done.append(bool(done))

    def build(self) -> OfflineBatch:
        if not self.scalar:
            raise ValueError("Offline dataset is empty")
        return OfflineBatch(
            scalar=np.stack(self.scalar, axis=0),
            party=np.stack(self.party, axis=0),
            action_vectors=np.stack(self.action_vectors, axis=0),
            action_mask=np.stack(self.action_mask, axis=0),
            action_index=np.asarray(self.action_index, dtype=np.int64),
            reward=np.asarray(self.reward, dtype=np.float32),
            next_scalar=np.stack(self.next_scalar, axis=0),
            next_party=np.stack(self.next_party, axis=0),
            next_action_vectors=np.stack(self.next_action_vectors, axis=0),
            next_action_mask=np.stack(self.next_action_mask, axis=0),
            done=np.asarray(self.done, dtype=np.bool_),
        )


def save_offline_batch(path: Path, batch: OfflineBatch) -> None:
    """save_offline_batch utility used by the RL self-play workflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scalar=batch.scalar,
        party=batch.party,
        action_vectors=batch.action_vectors,
        action_mask=batch.action_mask,
        action_index=batch.action_index,
        reward=batch.reward,
        next_scalar=batch.next_scalar,
        next_party=batch.next_party,
        next_action_vectors=batch.next_action_vectors,
        next_action_mask=batch.next_action_mask,
        done=batch.done,
    )


def load_offline_batch(path: Path) -> OfflineBatch:
    """load_offline_batch utility used by the RL self-play workflow."""
    with np.load(path, allow_pickle=False) as payload:
        return OfflineBatch(
            scalar=payload["scalar"].astype(np.float32),
            party=payload["party"].astype(np.float32),
            action_vectors=payload["action_vectors"].astype(np.float32),
            action_mask=payload["action_mask"].astype(np.bool_),
            action_index=payload["action_index"].astype(np.int64),
            reward=payload["reward"].astype(np.float32),
            next_scalar=payload["next_scalar"].astype(np.float32),
            next_party=payload["next_party"].astype(np.float32),
            next_action_vectors=payload["next_action_vectors"].astype(np.float32),
            next_action_mask=payload["next_action_mask"].astype(np.bool_),
            done=payload["done"].astype(np.bool_),
        )
