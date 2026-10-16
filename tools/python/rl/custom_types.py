"""Shared types for the isolated RL training package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

import numpy as np

from ..ou_selfplay_client import BattleState, LegalAction, PlayerId


@dataclass(frozen=True)
class BattleFeatures:
    """Compact RL-facing representation of the current battle state."""

    player_id: PlayerId
    turn_index: int
    done: bool
    winner: str | None
    request_kind: str
    pending_players: tuple[PlayerId, ...]
    request: dict[str, Any] | None
    scalar_features: np.ndarray
    party_features: np.ndarray
    legal_actions: tuple[LegalAction, ...]
    legal_action_mask: np.ndarray


class OpponentPolicy(Protocol):
    """Callable used by the environment adapter to auto-play the opponent."""

    def __call__(self, state: BattleState, player: PlayerId, legal_actions: tuple[LegalAction, ...]) -> int:
        """Return the index of the chosen legal action."""
        ...