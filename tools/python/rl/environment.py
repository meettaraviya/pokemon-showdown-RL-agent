"""Environment adapter that keeps training isolated from battle control logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ou_selfplay_client import BattleState, LegalAction, PlayerId, SelfPlayBattle
from .features import build_battle_features
from .types import BattleFeatures, OpponentPolicy


@dataclass(frozen=True)
class BattleStep:
    """One environment transition returned by the single-agent adapter."""

    observation: BattleFeatures
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


def _default_opponent_policy(state: BattleState, player: PlayerId, legal_actions: tuple[LegalAction, ...]) -> int:
    del state, player
    if not legal_actions:
        raise ValueError("Opponent policy received no legal actions")
    return 0


class SingleAgentBattleEnv:
    """Thin single-agent wrapper around `SelfPlayBattle`.

    The environment controls one player and automatically advances the other
    side with a policy callback. This keeps the RL loop separate from the
    simulator/session layer while still supporting self-play.
    """

    def __init__(
        self,
        battle: SelfPlayBattle,
        player_id: PlayerId = "p1",
        opponent_policy: OpponentPolicy | None = None,
    ) -> None:
        self._battle = battle
        self._player_id = player_id
        self._opponent_policy = opponent_policy or _default_opponent_policy
        self._last_observation: BattleFeatures | None = None

    @property
    def battle(self) -> SelfPlayBattle:
        return self._battle

    @property
    def player_id(self) -> PlayerId:
        return self._player_id

    def reset(self) -> BattleStep:
        self._last_observation = self.observe()
        return BattleStep(
            observation=self._last_observation,
            reward=0.0,
            done=self._battle.state.done,
            info=self._build_info(),
        )

    def observe(self) -> BattleFeatures:
        observation = build_battle_features(self._battle.state, self._player_id)
        self._last_observation = observation
        return observation

    def legal_actions(self) -> tuple[LegalAction, ...]:
        return self._current_legal_actions()

    def _current_legal_actions(self) -> tuple[LegalAction, ...]:
        return tuple(self._battle.possible_actions(self._player_id))

    def _reward_for_state(self, state: BattleState) -> float:
        if not state.done:
            return 0.0
        if state.winner == self._player_id:
            return 1.0
        if state.winner == "tie" or state.winner is None:
            return 0.0
        return -1.0

    def _auto_play_pending_players(self) -> None:
        while not self._battle.state.done and self._player_id not in self._battle.state.pending_players:
            pending_players = [player for player in self._battle.state.pending_players if player != self._player_id]
            if not pending_players:
                break
            actions: dict[PlayerId, str | LegalAction] = {}
            for player in pending_players:
                legal_actions = tuple(self._battle.possible_actions(player))
                if not legal_actions:
                    continue
                choice_index = self._opponent_policy(self._battle.state, player, legal_actions)
                if choice_index < 0 or choice_index >= len(legal_actions):
                    raise ValueError(f"Opponent policy returned invalid index {choice_index} for {player}")
                actions[player] = legal_actions[choice_index]
            if not actions:
                break
            self._battle.step(actions)

    def step(self, action_index: int) -> BattleStep:
        legal_actions = self._current_legal_actions()
        if not legal_actions:
            raise ValueError(f"No legal actions available for {self._player_id}")
        if action_index < 0 or action_index >= len(legal_actions):
            raise ValueError(f"Action index {action_index} out of bounds for {self._player_id}")

        chosen_action = legal_actions[action_index]
        self._battle.choose(self._player_id, chosen_action)
        self._auto_play_pending_players()

        observation = self.observe()
        reward = self._reward_for_state(self._battle.state)
        step_info = self._build_info()
        step_info["chosen_action"] = chosen_action.command
        step_info["legal_action_count"] = len(legal_actions)
        return BattleStep(observation=observation, reward=reward, done=self._battle.state.done, info=step_info)

    def simulate(self, action_index: int) -> BattleFeatures:
        legal_actions = self._current_legal_actions()
        if action_index < 0 or action_index >= len(legal_actions):
            raise ValueError(f"Action index {action_index} out of bounds for {self._player_id}")
        next_state = self._battle.simulate_choose(self._player_id, legal_actions[action_index])
        return build_battle_features(next_state, self._player_id)

    def _build_info(self) -> dict[str, Any]:
        state = self._battle.state
        return {
            "battle_id": state.battle_id,
            "turns": state.turns,
            "done": state.done,
            "winner": state.winner,
            "pending_players": tuple(state.pending_players),
            "history_cursor": self._battle.cursor,
            "history_length": self._battle.max_cursor,
        }