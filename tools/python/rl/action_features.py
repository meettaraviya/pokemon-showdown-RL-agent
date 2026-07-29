"""Action featurization utilities for variable-size legal action sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ..ou_selfplay_client import LegalAction
except ImportError:  # pragma: no cover - allows script-style execution
    from tools.python.ou_selfplay_client import LegalAction


ACTION_KINDS: tuple[str, ...] = ("move", "switch", "team", "pass", "mixed")
ACTION_FEATURE_DIM = len(ACTION_KINDS) + 9


@dataclass(frozen=True)
class EncodedActions:
    vectors: np.ndarray
    mask: np.ndarray


def encode_action(action: LegalAction) -> np.ndarray:
    command = action.command.strip().lower()
    parts = command.split()
    kind_vector = np.zeros(len(ACTION_KINDS), dtype=np.float32)
    if action.kind in ACTION_KINDS:
        kind_vector[ACTION_KINDS.index(action.kind)] = 1.0

    move_slot = 0.0
    switch_slot = 0.0
    target_slot = 0.0
    uses_mega = 0.0
    uses_ultra = 0.0
    uses_zmove = 0.0
    uses_max = 0.0
    uses_tera = 0.0

    if parts:
        if parts[0] == "move" and len(parts) >= 2:
            try:
                move_slot = float(int(parts[1]))
            except ValueError:
                move_slot = 0.0
            for token in parts[2:]:
                if token == "mega":
                    uses_mega = 1.0
                elif token == "ultra":
                    uses_ultra = 1.0
                elif token == "zmove":
                    uses_zmove = 1.0
                elif token == "max":
                    uses_max = 1.0
                elif token == "terastallize":
                    uses_tera = 1.0
                else:
                    try:
                        target_slot = float(int(token.lstrip("+")))
                    except ValueError:
                        if token.startswith("-"):
                            try:
                                target_slot = -float(int(token[1:]))
                            except ValueError:
                                pass
        elif parts[0] == "switch" and len(parts) >= 2:
            try:
                switch_slot = float(int(parts[1]))
            except ValueError:
                switch_slot = 0.0
        elif parts[0] == "team" and len(parts) >= 2:
            try:
                switch_slot = float(int(parts[1][0]))
            except (ValueError, IndexError):
                switch_slot = 0.0

    dense = np.array(
        [
            move_slot / 6.0,
            switch_slot / 6.0,
            target_slot / 3.0,
            uses_mega,
            uses_ultra,
            uses_zmove,
            uses_max,
            uses_tera,
            min(1.0, len(command) / 24.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([kind_vector, dense], dtype=np.float32)


def encode_legal_actions(actions: tuple[LegalAction, ...], max_actions: int) -> EncodedActions:
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    vectors = np.zeros((max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=np.bool_)
    limit = min(max_actions, len(actions))
    for index in range(limit):
        vectors[index] = encode_action(actions[index])
        mask[index] = True
    return EncodedActions(vectors=vectors, mask=mask)
