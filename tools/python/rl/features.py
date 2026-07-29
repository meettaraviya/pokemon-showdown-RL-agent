"""Feature extraction for the isolated RL training package."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..ou_selfplay_client import BattleState, PlayerId
from .types import BattleFeatures


def _condition_hp_ratio(condition: str) -> float:
    if not condition or condition.endswith(" fnt"):
        return 0.0
    if "/" not in condition:
        return 1.0
    hp_text, max_hp_text = condition.split("/", 1)
    try:
        current_hp = float(hp_text.split(" ", 1)[0])
        max_hp = float(max_hp_text.split(" ", 1)[0])
    except ValueError:
        return 0.0
    if max_hp <= 0:
        return 0.0
    return max(0.0, min(1.0, current_hp / max_hp))


def _pokemon_slot_features(pokemon: Mapping[str, object]) -> np.ndarray:
    condition = str(pokemon.get("condition", ""))
    return np.array(
        [
            1.0 if bool(pokemon.get("active")) else 0.0,
            1.0 if bool(pokemon.get("fainted")) or condition.endswith(" fnt") else 0.0,
            1.0 if bool(pokemon.get("reviving")) else 0.0,
            _condition_hp_ratio(condition),
        ],
        dtype=np.float32,
    )


def _build_party_features(state: BattleState, player: PlayerId) -> np.ndarray:
    party = state.side_pokemon(player)
    rows = [_pokemon_slot_features(pokemon) for pokemon in party[:6]]
    if len(rows) < 6:
        empty_row = np.zeros(4, dtype=np.float32)
        rows.extend([empty_row] * (6 - len(rows)))
    return np.stack(rows, axis=0)


def _request_kind(request: Mapping[str, object] | None) -> str:
    if request is None:
        return "none"
    if request.get("teamPreview"):
        return "team_preview"
    if request.get("forceSwitch"):
        return "force_switch"
    if request.get("active"):
        return "active"
    if request.get("wait"):
        return "wait"
    return "unknown"


def build_battle_features(state: BattleState, player: PlayerId) -> BattleFeatures:
    request = state.request_for(player)
    legal_actions = tuple(state.possible_actions(player))
    scalar_features = np.array(
        [
            float(state.turns),
            1.0 if state.done else 0.0,
            1.0 if state.winner == player else 0.0,
            1.0 if state.winner not in {None, player, "tie"} else 0.0,
            float(len(state.pending_players)),
            float(len(state.errors.get(player, []))),
            float(len(legal_actions)),
        ],
        dtype=np.float32,
    )
    party_features = _build_party_features(state, player)
    legal_action_mask = np.ones(len(legal_actions), dtype=np.bool_)
    return BattleFeatures(
        player_id=player,
        turn_index=state.turns,
        done=state.done,
        winner=state.winner,
        request_kind=_request_kind(request),
        pending_players=tuple(state.pending_players),
        scalar_features=scalar_features,
        party_features=party_features,
        legal_actions=legal_actions,
        legal_action_mask=legal_action_mask,
    )


def heuristic_position_score(features: BattleFeatures) -> float:
    """Simple dense proxy for position quality from visible HP/faint data."""

    party = features.party_features
    if party.size == 0:
        return 0.0
    hp_sum = float(party[:, 3].sum())
    fainted = float(party[:, 1].sum())
    alive_fraction = max(0.0, min(1.0, 1.0 - fainted / max(1.0, float(party.shape[0]))))
    return 0.7 * hp_sum / max(1.0, float(party.shape[0])) + 0.3 * alive_fraction