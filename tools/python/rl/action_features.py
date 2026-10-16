"""Action featurization utilities for variable-size legal action sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from ..ou_selfplay_client import LegalAction
except ImportError:  # pragma: no cover - allows script-style execution
    from tools.python.ou_selfplay_client import LegalAction


ACTION_KINDS: tuple[str, ...] = ("move", "switch", "team", "pass", "mixed")
TARGET_KINDS: tuple[str, ...] = (
    "normal",
    "any",
    "adjacentfoe",
    "adjacentally",
    "adjacentallyorself",
    "self",
    "scripted",
    "other",
)
ACTION_DENSE_FEATURE_DIM = 17
REQUEST_FLAG_DIM = 6
MOVE_EMBED_DIM = 8
ACTION_FEATURE_DIM = (
    len(ACTION_KINDS)
    + len(TARGET_KINDS)
    + ACTION_DENSE_FEATURE_DIM
    + REQUEST_FLAG_DIM
    + MOVE_EMBED_DIM
    + MOVE_EMBED_DIM
)


@dataclass(frozen=True)
class EncodedActions:
    """EncodedActions helper type/container used by the RL self-play tooling."""
    vectors: np.ndarray
    mask: np.ndarray


def _target_onehot(target: str) -> np.ndarray:
    """_target_onehot utility used by the RL self-play workflow."""
    out = np.zeros(len(TARGET_KINDS), dtype=np.float32)
    key = target.strip().lower()
    if key in TARGET_KINDS:
        out[TARGET_KINDS.index(key)] = 1.0
    else:
        out[-1] = 1.0
    return out


def _safe_active_entry(request: dict[str, Any] | None) -> dict[str, Any]:
    """_safe_active_entry utility used by the RL self-play workflow."""
    if not request:
        return {}
    active = request.get("active")
    if isinstance(active, list) and active and isinstance(active[0], dict):
        return active[0]
    return {}


def _safe_side_entry(request: dict[str, Any] | None, slot: int) -> dict[str, Any]:
    """_safe_side_entry utility used by the RL self-play workflow."""
    if not request:
        return {}
    side = request.get("side")
    if not isinstance(side, dict):
        return {}
    pokemon = side.get("pokemon")
    if not isinstance(pokemon, list):
        return {}
    index = slot - 1
    if index < 0 or index >= len(pokemon):
        return {}
    entry = pokemon[index]
    return entry if isinstance(entry, dict) else {}


def _extract_move_context(move_slot: int, request: dict[str, Any] | None) -> tuple[dict[str, Any], np.ndarray]:
    """_extract_move_context utility used by the RL self-play workflow."""
    active_entry = _safe_active_entry(request)
    if not active_entry:
        return {}, _target_onehot("other")
    move_entries = active_entry.get("moves")
    if not isinstance(move_entries, list) or move_slot <= 0 or move_slot > len(move_entries):
        return {}, _target_onehot("other")
    move_entry = move_entries[move_slot - 1]
    if not isinstance(move_entry, dict):
        return {}, _target_onehot("other")
    target = str(move_entry.get("target", "other"))
    return move_entry, _target_onehot(target)


def encode_action(action: LegalAction, request: dict[str, Any] | None = None) -> np.ndarray:
    """encode_action utility used by the RL self-play workflow."""
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
    target_vector = _target_onehot("other")
    move_pp_frac = 0.0
    move_disabled = 0.0
    move_id_embed = np.zeros(8, dtype=np.float32)
    move_name_embed = np.zeros(8, dtype=np.float32)

    switch_hp_frac = 0.0
    switch_fainted = 0.0
    switch_active = 0.0
    switch_statused = 0.0
    switch_has_item = 0.0
    switch_tera_ready = 0.0

    request_flags = np.array(
        [
            1.0 if request and request.get("teamPreview") else 0.0,
            1.0 if request and request.get("forceSwitch") else 0.0,
            1.0 if request and request.get("active") else 0.0,
            1.0 if request and request.get("wait") else 0.0,
            1.0 if request and _safe_active_entry(request).get("canTerastallize") else 0.0,
            1.0 if request and _safe_active_entry(request).get("trapped") else 0.0,
        ],
        dtype=np.float32,
    )

    if parts:
        if parts[0] == "move" and len(parts) >= 2:
            try:
                move_slot = float(int(parts[1]))
            except ValueError:
                move_slot = 0.0
            move_entry, target_vector = _extract_move_context(int(move_slot), request)
            if move_entry:
                try:
                    pp = float(move_entry.get("pp", 0.0))
                    max_pp = float(move_entry.get("maxpp", 0.0))
                    if max_pp > 0:
                        move_pp_frac = max(0.0, min(1.0, pp / max_pp))
                except (TypeError, ValueError):
                    move_pp_frac = 0.0
                move_disabled = 1.0 if bool(move_entry.get("disabled")) else 0.0
                move_id_embed = _string_hash_embedding(str(move_entry.get("id", "")), 8, "move_id")
                move_name_embed = _string_hash_embedding(str(move_entry.get("move", "")), 8, "move_name")
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
            side_entry = _safe_side_entry(request, int(switch_slot))
            condition = str(side_entry.get("condition", ""))
            if "/" in condition:
                hp_text, max_hp_text = condition.split("/", 1)
                try:
                    hp = float(hp_text.split(" ", 1)[0])
                    max_hp = float(max_hp_text.split(" ", 1)[0])
                    if max_hp > 0:
                        switch_hp_frac = max(0.0, min(1.0, hp / max_hp))
                except ValueError:
                    switch_hp_frac = 0.0
            switch_fainted = 1.0 if " fnt" in condition else 0.0
            switch_active = 1.0 if bool(side_entry.get("active")) else 0.0
            switch_statused = 1.0 if any(status in condition for status in (" brn", " par", " psn", " tox", " slp", " frz")) else 0.0
            switch_has_item = 1.0 if bool(side_entry.get("item")) else 0.0
            switch_tera_ready = 1.0 if bool(side_entry.get("teraType")) and not bool(side_entry.get("terastallized")) else 0.0
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
            move_pp_frac,
            move_disabled,
            switch_hp_frac,
            switch_fainted,
            switch_active,
            switch_statused,
            switch_has_item,
            switch_tera_ready,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            kind_vector,
            target_vector,
            dense,
            request_flags,
            move_id_embed,
            move_name_embed,
        ],
        dtype=np.float32,
    )


def _string_hash_embedding(text: str, dim: int, salt: str) -> np.ndarray:
    """_string_hash_embedding utility used by the RL self-play workflow."""
    if not text:
        return np.zeros(dim, dtype=np.float32)
    import hashlib

    digest = hashlib.blake2b(f"{salt}:{text}".encode("utf-8"), digest_size=dim).digest()
    values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    return values / 127.5 - 1.0


def encode_legal_actions(
    actions: tuple[LegalAction, ...],
    max_actions: int,
    request: dict[str, Any] | None = None,
) -> EncodedActions:
    """encode_legal_actions utility used by the RL self-play workflow."""
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    vectors = np.zeros((max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=np.bool_)
    limit = min(max_actions, len(actions))
    for index in range(limit):
        vectors[index] = encode_action(actions[index], request=request)
        mask[index] = True
    return EncodedActions(vectors=vectors, mask=mask)
