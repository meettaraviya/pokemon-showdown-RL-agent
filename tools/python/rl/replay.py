"""Replay storage for self-play training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import random

from .custom_types import BattleFeatures


@dataclass(frozen=True)
class Transition:
    """A single RL transition collected from the battle adapter."""

    observation: BattleFeatures
    action_index: int
    reward: float
    next_observation: BattleFeatures
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """Bounded replay buffer with sampling support."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def append(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.append(transition)

    def sample(self, batch_size: int, seed: int | None = None) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {len(self._buffer)}")
        rng = random.Random(seed)
        return rng.sample(list(self._buffer), batch_size)

    def clear(self) -> None:
        self._buffer.clear()