#!/usr/bin/env python3
"""Python client for the Pokemon Showdown OU self-play bridge.

The bridge process speaks JSON lines over stdin/stdout and is implemented in:
sim/examples/ou-selfplay-bridge.ts
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from itertools import permutations
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
from typing import Any, Literal, cast


class BridgeError(RuntimeError):
    """Raised when the JS bridge returns an error payload or exits unexpectedly."""


class InvalidChoiceError(ValueError):
    """Raised when a submitted action is not legal for the current request."""

    def __init__(self, message: str, state: BattleState | None = None) -> None:
        super().__init__(message)
        self.state = state


PlayerId = Literal["p1", "p2"]
ActionKind = Literal["move", "switch", "team", "pass", "mixed"]
PLAYER_IDS: tuple[PlayerId, PlayerId] = ("p1", "p2")
TeamSpec = str | Path


@dataclass
class EpisodeResult:
    formatid: str
    winner: str
    winner_name: str
    turns: int
    seed: list[int] | None = None
    log: list[str] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EpisodeResult":
        return cls(
            formatid=str(payload.get("formatid", "")),
            winner=str(payload.get("winner", "tie")),
            winner_name=str(payload.get("winnerName", "")),
            turns=int(payload.get("turns", 0)),
            seed=payload.get("seed"),
            log=payload.get("log"),
        )


@dataclass(frozen=True)
class LegalAction:
    command: str
    kind: ActionKind
    description: str


@dataclass
class PlayerDecision:
    player: PlayerId
    request: dict[str, Any]
    actions: list[LegalAction]


@dataclass(frozen=True)
class BattleConfig:
    formatid: str
    include_log: bool
    p1team: str | None
    p2team: str | None
    seed: list[int]


@dataclass(frozen=True)
class ActionRecord:
    index: int
    player: PlayerId
    command: str
    turns_before: int
    turns_after: int
    done_after: bool


@dataclass
class BattleState:
    battle_id: str
    formatid: str
    turns: int
    done: bool
    winner: str | None
    winner_name: str
    waiting_for: list[PlayerId]
    requests: dict[PlayerId, dict[str, Any]]
    errors: dict[PlayerId, list[str]]
    logs: list[str]
    seed: list[int] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BattleState":
        requests_payload = _ensure_mapping(payload.get("requests"), "state.requests")
        errors_payload = _ensure_mapping(payload.get("errors"), "state.errors")
        waiting_for_raw = payload.get("waitingFor")
        waiting_for: list[PlayerId] = []
        if isinstance(waiting_for_raw, list):
            waiting_for = [player for player in cast(list[Any], waiting_for_raw) if player in PLAYER_IDS]
        requests: dict[PlayerId, dict[str, Any]] = {}
        for player in PLAYER_IDS:
            request = requests_payload.get(player)
            if isinstance(request, dict):
                requests[player] = cast(dict[str, Any], request)
        errors: dict[PlayerId, list[str]] = {}
        for player in PLAYER_IDS:
            messages = errors_payload.get(player)
            if isinstance(messages, list):
                errors[player] = [str(message) for message in cast(list[Any], messages)]
        logs_payload = payload.get("log")
        logs = [str(chunk) for chunk in cast(list[Any], logs_payload)] if isinstance(logs_payload, list) else []
        seed_payload = payload.get("seed")
        seed = [int(value) for value in cast(list[Any], seed_payload)] if isinstance(seed_payload, list) else None
        return cls(
            battle_id=str(payload.get("battleId", "")),
            formatid=str(payload.get("formatid", "")),
            turns=int(payload.get("turns", 0)),
            done=bool(payload.get("done", False)),
            winner=str(payload["winner"]) if payload.get("winner") is not None else None,
            winner_name=str(payload.get("winnerName", "")),
            waiting_for=waiting_for,
            requests=requests,
            errors=errors,
            logs=logs,
            seed=seed,
        )

    @property
    def pending_players(self) -> list[PlayerId]:
        return self.waiting_for[:]

    def request_for(self, player: PlayerId) -> dict[str, Any] | None:
        request = self.requests.get(player)
        if not request or request.get("wait"):
            return None
        return request

    def decision_for(self, player: PlayerId) -> PlayerDecision | None:
        request = self.request_for(player)
        if request is None:
            return None
        return PlayerDecision(player=player, request=request, actions=enumerate_legal_actions(request))

    def possible_actions(self, player: PlayerId) -> list[LegalAction]:
        decision = self.decision_for(player)
        return decision.actions if decision is not None else []

    def side_for(self, player: PlayerId) -> dict[str, Any] | None:
        request = self.request_for(player)
        if request is None:
            return None
        side = request.get("side")
        return cast(dict[str, Any], side) if isinstance(side, dict) else None

    def side_pokemon(self, player: PlayerId) -> list[dict[str, Any]]:
        side = self.side_for(player)
        if side is None:
            return []
        pokemon = side.get("pokemon")
        if not isinstance(pokemon, list):
            return []
        detailed: list[dict[str, Any]] = []
        for entry in cast(list[Any], pokemon):
            if isinstance(entry, dict):
                detailed.append(cast(dict[str, Any], entry))
        return detailed

    def pokemon_details(self) -> dict[PlayerId, list[dict[str, Any]]]:
        return {player: self.side_pokemon(player) for player in PLAYER_IDS}

    def to_simple(self) -> "SimpleBattleState":
        return SimpleBattleState.from_state(self)


@dataclass
class SimpleBattleState(BattleState):
    @classmethod
    def from_state(cls, state: BattleState) -> "SimpleBattleState":
        return cls(
            battle_id=state.battle_id,
            formatid=state.formatid,
            turns=state.turns,
            done=state.done,
            winner=state.winner,
            winner_name=state.winner_name,
            waiting_for=state.waiting_for[:],
            requests=copy.deepcopy(state.requests),
            errors=copy.deepcopy(state.errors),
            logs=state.logs[:],
            seed=state.seed[:] if state.seed is not None else None,
        )

    def decision_for(self, player: PlayerId) -> PlayerDecision | None:
        request = self.request_for(player)
        if request is None:
            return None
        if request.get("teamPreview"):
            return PlayerDecision(player=player, request=request, actions=_enumerate_simple_team_preview_actions(request))
        return super().decision_for(player)


class SelfPlayBattle:
    def __init__(
        self,
        client: OUSelfPlayClient,
        state: BattleState,
        config: BattleConfig,
        state_cls: type[BattleState] = BattleState,
    ) -> None:
        self._client = client
        self._config = config
        self._state_cls = state_cls
        self.state = state
        self._history: list[ActionRecord] = []
        self._cursor = 0
        self._snapshots: list[BattleState] = [_clone_state(state)]
        self._closed = False
        self._branch_generation = 0

    @property
    def config(self) -> BattleConfig:
        return self._config

    @property
    def history(self) -> list[ActionRecord]:
        return self._history[:]

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def max_cursor(self) -> int:
        return len(self._history)

    def _ensure_open(self) -> None:
        if self._closed:
            raise BridgeError("Battle session is closed")

    def refresh(self) -> BattleState:
        self._ensure_open()
        message = self._client.request({"type": "get_battle_state", "battleId": self.state.battle_id})
        if message.get("type") != "battle_state":
            raise BridgeError(f"Unexpected message from bridge: {message}")
        self.state = _battle_state_from_message(message, self._state_cls)
        return self.state

    def possible_actions(self, player: PlayerId) -> list[LegalAction]:
        return self.state.possible_actions(player)

    def _submit_action(self, player: PlayerId, command: str) -> BattleState:
        message = self._client.request(
            {"type": "choose_action", "battleId": self.state.battle_id, "player": player, "action": command}
        )
        if message.get("type") != "action_result":
            raise BridgeError(f"Unexpected message from bridge: {message}")
        self.state = _battle_state_from_message(message, self._state_cls)
        if self.state.errors.get(player):
            raise InvalidChoiceError("; ".join(self.state.errors[player]), self.state)
        return self.state

    def _truncate_future(self) -> None:
        if self._cursor >= len(self._history):
            return
        self._history = self._history[: self._cursor]
        self._snapshots = self._snapshots[: self._cursor + 1]

    def _clone_config_with_seed(self, seed: list[int]) -> BattleConfig:
        return BattleConfig(
            formatid=self._config.formatid,
            include_log=self._config.include_log,
            p1team=self._config.p1team,
            p2team=self._config.p2team,
            seed=seed,
        )

    def _derive_branch_seed(self, cursor: int) -> list[int]:
        payload = f"{self._config.formatid}:{cursor}:{self._branch_generation}:{','.join(str(value) for value in self._config.seed)}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return [int.from_bytes(digest[index : index + 2], "big") % 65536 for index in range(0, 8, 2)]

    def _rebuild_from_seed(self, seed: list[int], target_cursor: int) -> BattleState:
        self._close_remote()
        config = self._clone_config_with_seed(seed)
        self.state = self._client.create_battle_state(config, self._state_cls)
        rebuilt_snapshots: list[BattleState] = [_clone_state(self.state)]
        for record in self._history[:target_cursor]:
            next_state = self._submit_action(record.player, record.command)
            rebuilt_snapshots.append(_clone_state(next_state))
        self._cursor = target_cursor
        self._snapshots = rebuilt_snapshots
        return self.state

    def _rebuild_to_cursor(self, target_cursor: int) -> BattleState:
        if target_cursor < 0 or target_cursor > len(self._history):
            raise ValueError(f"Cursor {target_cursor} is out of bounds [0, {len(self._history)}]")

        self._branch_generation += 1
        seed = self._derive_branch_seed(target_cursor)
        return self._rebuild_from_seed(seed, target_cursor)

    def _record_action(self, player: PlayerId, command: str, turns_before: int, next_state: BattleState) -> None:
        record = ActionRecord(
            index=len(self._history),
            player=player,
            command=command,
            turns_before=turns_before,
            turns_after=next_state.turns,
            done_after=next_state.done,
        )
        self._history.append(record)
        self._cursor = len(self._history)
        self._snapshots.append(_clone_state(next_state))

    def choose(self, player: PlayerId, action: str | LegalAction, *, validate: bool = True) -> BattleState:
        self._ensure_open()
        request = self.state.request_for(player)
        if request is None:
            raise InvalidChoiceError(f"{player} does not have a pending choice", self.state)
        command = action.command if isinstance(action, LegalAction) else str(action).strip()
        if validate:
            legal = {candidate.command for candidate in enumerate_legal_actions(request)}
            if command not in legal:
                raise InvalidChoiceError(f"Illegal action for {player}: {command}", self.state)
        turns_before = self.state.turns
        self._truncate_future()
        next_state = self._submit_action(player, command)
        self._record_action(player, command, turns_before, next_state)
        return next_state

    def step(self, actions: Mapping[PlayerId, str | LegalAction]) -> BattleState:
        self._ensure_open()
        state = self.state
        for player in ("p1", "p2"):
            if player in actions:
                state = self.choose(player, actions[player])
        return state

    def restore(self, cursor: int) -> BattleState:
        self._ensure_open()
        return self._rebuild_to_cursor(cursor)

    def undo(self, steps: int = 1) -> BattleState:
        self._ensure_open()
        if steps < 0:
            raise ValueError("steps must be non-negative")
        target = max(0, self._cursor - steps)
        return self.restore(target)

    def redo(self, steps: int = 1) -> BattleState:
        self._ensure_open()
        if steps < 0:
            raise ValueError("steps must be non-negative")
        target = min(len(self._history), self._cursor + steps)
        return self.restore(target)

    def state_at(self, cursor: int) -> BattleState:
        if cursor < 0 or cursor > len(self._history):
            raise ValueError(f"Cursor {cursor} is out of bounds [0, {len(self._history)}]")
        if cursor < len(self._snapshots):
            return _clone_state(self._snapshots[cursor])
        forked = self.fork_at(cursor)
        try:
            return _clone_state(forked.state)
        finally:
            forked.close()

    def fork(self) -> SelfPlayBattle:
        return self.fork_at(self._cursor)

    def fork_at(self, cursor: int) -> SelfPlayBattle:
        self._ensure_open()
        if cursor < 0 or cursor > len(self._history):
            raise ValueError(f"Cursor {cursor} is out of bounds [0, {len(self._history)}]")
        branch_seed = self._derive_branch_seed(cursor)
        branch_config = self._clone_config_with_seed(branch_seed)
        branch_state = self._client.create_battle_state(branch_config, self._state_cls)
        branch = SelfPlayBattle(self._client, branch_state, branch_config, self._state_cls)
        for record in self._history[:cursor]:
            branch.choose(record.player, record.command, validate=False)
        return branch

    def simulate_choose(self, player: PlayerId, action: str | LegalAction, *, validate: bool = True) -> BattleState:
        branch = self.fork()
        try:
            next_state = branch.choose(player, action, validate=validate)
            return _clone_state(next_state)
        finally:
            branch.close()

    def simulate_step(self, actions: Mapping[PlayerId, str | LegalAction]) -> BattleState:
        branch = self.fork()
        try:
            next_state = branch.step(actions)
            return _clone_state(next_state)
        finally:
            branch.close()

    def _close_remote(self) -> None:
        if not self.state.battle_id:
            return
        message = self._client.request({"type": "close_battle", "battleId": self.state.battle_id})
        if message.get("type") != "battle_closed":
            raise BridgeError(f"Unexpected message from bridge: {message}")

    def close(self) -> None:
        if self._closed:
            return
        self._close_remote()
        self._closed = True
        self.state = BattleState(
            battle_id="",
            formatid=self.state.formatid,
            turns=self.state.turns,
            done=True,
            winner=self.state.winner,
            winner_name=self.state.winner_name,
            waiting_for=[],
            requests={},
            errors={},
            logs=[],
            seed=self.state.seed,
        )

    def __enter__(self) -> "SelfPlayBattle":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class OUSelfPlayClient:
    def __init__(
        self,
        repo_root: str | Path,
        node_cmd: str = "node",
        bridge_entry: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.node_cmd = self._resolve_node_cmd(node_cmd)
        self.bridge_entry = bridge_entry
        self.bridge_path = self._resolve_bridge_path(bridge_entry)
        self._request_id = 0

        try:
            self._proc = subprocess.Popen(
                [self.node_cmd, str(self.bridge_path)],
                cwd=str(self.repo_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise BridgeError(
                f"Failed to start Node process with '{self.node_cmd}'. "
                "Install Node.js 22+ and ensure it is on PATH."
            ) from exc

        ready = self._read_message()
        if ready.get("type") != "ready":
            raise BridgeError(f"Bridge did not start correctly: {ready}")

    def _resolve_node_cmd(self, node_cmd: str) -> str:
        explicit = shutil.which(node_cmd)
        if explicit:
            return explicit
        # Common Linux distro alias when `node` is not provided.
        if node_cmd == "node":
            fallback = shutil.which("nodejs")
            if fallback:
                return fallback
        raise BridgeError(
            "Node.js executable not found. Install Node.js 22+ and make sure "
            "`node` is available on PATH."
        )

    def _resolve_bridge_path(self, bridge_entry: str | None) -> Path:
        candidates: list[Path] = []
        if bridge_entry:
            candidates.append(self.repo_root / bridge_entry)
        candidates.extend(
            [
                self.repo_root / ".sim-dist/examples/ou-selfplay-bridge",
                self.repo_root / ".sim-dist/examples/ou-selfplay-bridge.js",
                self.repo_root / "dist/sim/examples/ou-selfplay-bridge.js",
            ]
        )
        for bridge_path in candidates:
            if bridge_path.exists():
                return bridge_path
        raise BridgeError(
            "Bridge entry was not found in known locations "
            f"({', '.join(str(path) for path in candidates)}). "
            "Run `node build` from repo root first."
        )

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _read_message(self) -> dict[str, Any]:
        if self._proc.stdout is None:
            raise BridgeError("Bridge stdout is unavailable")
        line = self._proc.stdout.readline()
        if line == "":
            stderr_text = ""
            if self._proc.stderr is not None:
                try:
                    stderr_text = self._proc.stderr.read()
                except Exception:
                    stderr_text = ""
            raise BridgeError(
                f"Bridge exited unexpectedly (code={self._proc.returncode}). stderr={stderr_text}"
            )
        return json.loads(line)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc.stdin is None:
            raise BridgeError("Bridge stdin is unavailable")
        payload = dict(payload)
        payload.setdefault("requestId", self._next_request_id())
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

        message = self._read_message()
        if message.get("type") == "error":
            raise BridgeError(str(message.get("message", "Unknown bridge error")))
        return message

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(payload)

    def _resolve_team_spec(self, team: TeamSpec | None) -> str | None:
        if team is None:
            return None
        if isinstance(team, Path):
            return team.read_text(encoding="utf-8")
        return str(team)

    def _normalize_seed(self, seed: list[int] | None) -> list[int]:
        if seed is None:
            rng = random.SystemRandom()
            return [rng.randrange(0, 65536) for _ in range(4)]
        if len(seed) != 4 or any(value < 0 for value in seed):
            raise BridgeError("seed must be a list of four non-negative integers")
        return [int(value) for value in seed]

    def create_battle_state(self, config: BattleConfig, state_cls: type[BattleState] = BattleState) -> BattleState:
        payload: dict[str, Any] = {
            "type": "create_battle",
            "formatid": config.formatid,
            "includeLog": config.include_log,
            "seed": config.seed,
        }
        if config.p1team is not None:
            payload["p1team"] = config.p1team
        if config.p2team is not None:
            payload["p2team"] = config.p2team

        message = self._request(payload)
        if message.get("type") != "battle_created":
            raise BridgeError(f"Unexpected message from bridge: {message}")
        return _battle_state_from_message(message, state_cls)

    def read_team(self, team_path: str | Path) -> str:
        return Path(team_path).read_text(encoding="utf-8")

    def create_battle(
        self,
        *,
        formatid: str = "gen9ou",
        include_log: bool = False,
        p1team: TeamSpec | None = None,
        p2team: TeamSpec | None = None,
        seed: list[int] | None = None,
        simple_state: bool = False,
    ) -> SelfPlayBattle:
        p1team_text = self._resolve_team_spec(p1team)
        p2team_text = self._resolve_team_spec(p2team)
        state_cls: type[BattleState] = SimpleBattleState if simple_state else BattleState
        config = BattleConfig(
            formatid=formatid,
            include_log=include_log,
            p1team=p1team_text,
            p2team=p2team_text,
            seed=self._normalize_seed(seed),
        )
        state = self.create_battle_state(config, state_cls)
        return SelfPlayBattle(self, state, config, state_cls)

    def run_episode(
        self,
        *,
        formatid: str = "gen9ou",
        include_log: bool = False,
        p1team: TeamSpec | None = None,
        p2team: TeamSpec | None = None,
        seed: list[int] | None = None,
    ) -> EpisodeResult:
        p1team_text = self._resolve_team_spec(p1team)
        p2team_text = self._resolve_team_spec(p2team)
        payload: dict[str, Any] = {
            "type": "run_episode",
            "formatid": formatid,
            "includeLog": include_log,
        }
        if p1team_text is not None:
            payload["p1team"] = p1team_text
        if p2team_text is not None:
            payload["p2team"] = p2team_text
        if seed is not None:
            payload["seed"] = seed

        message = self._request(payload)
        if message.get("type") != "episode_result":
            raise BridgeError(f"Unexpected message from bridge: {message}")
        result_payload = message.get("result")
        if not isinstance(result_payload, dict):
            raise BridgeError(f"Malformed episode_result message: {message}")
        return EpisodeResult.from_payload(cast(dict[str, Any], result_payload))

    def run_batch(
        self,
        count: int,
        *,
        formatid: str = "gen9ou",
        include_log: bool = False,
        p1team: TeamSpec | None = None,
        p2team: TeamSpec | None = None,
    ) -> list[EpisodeResult]:
        p1team_text = self._resolve_team_spec(p1team)
        p2team_text = self._resolve_team_spec(p2team)
        payload: dict[str, Any] = {
            "type": "run_batch",
            "count": int(count),
            "formatid": formatid,
            "includeLog": include_log,
        }
        if p1team_text is not None:
            payload["p1team"] = p1team_text
        if p2team_text is not None:
            payload["p2team"] = p2team_text

        message = self._request(payload)
        if message.get("type") != "batch_result":
            raise BridgeError(f"Unexpected message from bridge: {message}")
        episodes_payload = message.get("episodes")
        if not isinstance(episodes_payload, list):
            raise BridgeError(f"Malformed batch_result message: {message}")
        return [
            EpisodeResult.from_payload(cast(dict[str, Any], ep))
            for ep in cast(list[Any], episodes_payload)
            if isinstance(ep, dict)
        ]


def _battle_state_from_message(
    message: Mapping[str, Any],
    state_cls: type[BattleState] = BattleState,
) -> BattleState:
    state_payload = _ensure_mapping(message.get("state"), "message.state")
    return state_cls.from_payload(state_payload)


def _ensure_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeError(f"Expected {context} to be a JSON object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _ensure_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise BridgeError(f"Expected {context} to be a JSON array, got {type(value).__name__}")
    return cast(Sequence[Any], value)


def _clone_state(state: BattleState) -> BattleState:
    return copy.deepcopy(state)


def _condition_is_fainted(condition: str) -> bool:
    return condition.endswith(" fnt")


def _pokemon_label(pokemon: Mapping[str, Any]) -> str:
    details = str(pokemon.get("details", ""))
    if details:
        return details.split(",", 1)[0].strip()
    ident = str(pokemon.get("ident", ""))
    return ident.split(":", 1)[-1].strip() if ident else "Pokemon"


def _slot_target_suffixes(target: str, actor_index: int, active_count: int) -> list[str | None]:
    if active_count <= 1:
        return [None]
    if target in {"normal", "any", "adjacentFoe"}:
        return [str(slot) for slot in range(1, active_count + 1)]
    if target == "adjacentAlly":
        return [f"-{slot + 1}" for slot in range(active_count) if slot != actor_index]
    if target == "adjacentAllyOrSelf":
        return [f"-{slot + 1}" for slot in range(active_count)]
    return [None]


def _describe_target(target_suffix: str | None) -> str:
    if target_suffix is None:
        return ""
    if target_suffix.startswith("-"):
        return f" targeting ally slot {target_suffix[1:]}"
    return f" targeting foe slot {target_suffix.lstrip('+')}"


@dataclass(frozen=True)
class _ChoiceOption:
    command: str
    kind: ActionKind
    description: str
    switch_slot: int | None = None
    uses_mega: bool = False
    uses_ultra: bool = False
    uses_zmove: bool = False
    uses_max: bool = False
    uses_tera: bool = False


def enumerate_legal_actions(request: Mapping[str, Any]) -> list[LegalAction]:
    if request.get("wait"):
        return []
    if request.get("teamPreview"):
        return _enumerate_team_preview_actions(request)
    if request.get("forceSwitch"):
        return _enumerate_force_switch_actions(request)
    if request.get("active"):
        return _enumerate_move_actions(request)
    raise BridgeError(f"Unsupported request payload: {request}")


def _enumerate_team_preview_actions(request: Mapping[str, Any]) -> list[LegalAction]:
    side = _ensure_mapping(request.get("side"), "request.side")
    pokemon = list(_ensure_sequence(side.get("pokemon"), "request.side.pokemon"))
    total_slots = len(pokemon)
    chosen_slots = int(request.get("maxChosenTeamSize") or total_slots)
    chosen_slots = max(1, min(total_slots, chosen_slots))
    actions: list[LegalAction] = []
    for prefix in permutations(range(1, total_slots + 1), chosen_slots):
        suffix = tuple(slot for slot in range(1, total_slots + 1) if slot not in prefix)
        team_order = prefix + suffix
        order_text = "".join(str(slot) for slot in team_order)
        preview = " -> ".join(_pokemon_label(cast(Mapping[str, Any], pokemon[slot - 1])) for slot in team_order)
        actions.append(LegalAction(command=f"team {order_text}", kind="team", description=f"team order {preview}"))
    return actions


def _enumerate_simple_team_preview_actions(request: Mapping[str, Any]) -> list[LegalAction]:
    side = _ensure_mapping(request.get("side"), "request.side")
    pokemon = list(_ensure_sequence(side.get("pokemon"), "request.side.pokemon"))
    total_slots = len(pokemon)
    if total_slots == 0:
        return []
    chosen_slots = int(request.get("maxChosenTeamSize") or total_slots)
    chosen_slots = max(1, min(total_slots, chosen_slots))
    actions: list[LegalAction] = []
    for lead_slot in range(1, total_slots + 1):
        remaining_slots = [slot for slot in range(1, total_slots + 1) if slot != lead_slot]
        tail_slots = remaining_slots[: max(0, chosen_slots - 1)]
        suffix_slots = remaining_slots[max(0, chosen_slots - 1):]
        order = [lead_slot, *tail_slots, *suffix_slots]
        order_text = "".join(str(slot) for slot in order)
        lead_label = _pokemon_label(cast(Mapping[str, Any], pokemon[lead_slot - 1]))
        if chosen_slots > 1:
            preview_tail = " -> ".join(
                _pokemon_label(cast(Mapping[str, Any], pokemon[slot - 1])) for slot in order[1:chosen_slots]
            )
            description = f"lead {lead_label}; auto-order rest as {preview_tail}"
        else:
            description = f"lead {lead_label}"
        actions.append(LegalAction(command=f"team {order_text}", kind="team", description=description))
    return actions


def _enumerate_force_switch_actions(request: Mapping[str, Any]) -> list[LegalAction]:
    force_switch = list(_ensure_sequence(request.get("forceSwitch"), "request.forceSwitch"))
    side = _ensure_mapping(request.get("side"), "request.side")
    pokemon = [cast(Mapping[str, Any], entry) for entry in _ensure_sequence(side.get("pokemon"), "request.side.pokemon")]
    options_per_slot: list[list[_ChoiceOption]] = []
    active_count = len(force_switch)

    for actor_index, must_switch in enumerate(force_switch):
        if not must_switch:
            options_per_slot.append([_ChoiceOption(command="pass", kind="pass", description="pass")])
            continue
        actor = pokemon[actor_index]
        actor_reviving = bool(actor.get("reviving"))
        candidates: list[_ChoiceOption] = []
        for slot in range(1, len(pokemon) + 1):
            candidate = pokemon[slot - 1]
            if slot <= active_count and not actor_reviving:
                continue
            candidate_is_fainted = _condition_is_fainted(str(candidate.get("condition", "")))
            if candidate_is_fainted != actor_reviving:
                continue
            label = _pokemon_label(candidate)
            candidates.append(
                _ChoiceOption(
                    command=f"switch {slot}",
                    kind="switch",
                    description=f"switch slot {actor_index + 1} to {label}",
                    switch_slot=slot,
                )
            )
        if not candidates:
            candidates = [_ChoiceOption(command="pass", kind="pass", description="pass")]
        options_per_slot.append(candidates)

    return _combine_choice_options(options_per_slot)


def _enumerate_move_actions(request: Mapping[str, Any]) -> list[LegalAction]:
    active_entries = [cast(Mapping[str, Any], entry) for entry in _ensure_sequence(request.get("active"), "request.active")]
    side = _ensure_mapping(request.get("side"), "request.side")
    pokemon = [cast(Mapping[str, Any], entry) for entry in _ensure_sequence(side.get("pokemon"), "request.side.pokemon")]
    options_per_slot: list[list[_ChoiceOption]] = []
    active_count = len(active_entries)

    for actor_index, active in enumerate(active_entries):
        actor = pokemon[actor_index]
        condition = str(actor.get("condition", ""))
        if _condition_is_fainted(condition) or bool(actor.get("commanding")):
            options_per_slot.append([_ChoiceOption(command="pass", kind="pass", description="pass")])
            continue

        slot_options: list[_ChoiceOption] = []
        move_entries = [cast(Mapping[str, Any], entry) for entry in _ensure_sequence(active.get("moves"), "active.moves")]
        zmove_entries = active.get("canZMove")
        zmoves: list[Any] = list(cast(list[Any], zmove_entries)) if isinstance(zmove_entries, list) else []
        max_moves_block = active.get("maxMoves")
        max_move_entries: list[Mapping[str, Any]] = []
        if isinstance(max_moves_block, Mapping):
            max_moves = cast(Mapping[str, Any], max_moves_block).get("maxMoves")
            if isinstance(max_moves, list):
                max_move_entries = [cast(Mapping[str, Any], entry) for entry in cast(list[Any], max_moves)]

        using_max_moves = bool(max_move_entries) and not bool(active.get("canDynamax"))
        candidate_move_entries = max_move_entries if using_max_moves else move_entries
        for move_slot, move in enumerate(candidate_move_entries, start=1):
            if move.get("disabled"):
                continue
            move_name = str(move.get("move", f"Move {move_slot}"))
            target = str(move.get("target", "normal"))
            for target_suffix in _slot_target_suffixes(target, actor_index, active_count):
                base_command = f"move {move_slot}"
                if target_suffix is not None:
                    base_command = f"{base_command} {target_suffix}"
                slot_options.append(
                    _ChoiceOption(
                        command=base_command,
                        kind="move",
                        description=f"use {move_name}{_describe_target(target_suffix)}",
                    )
                )
                if bool(active.get("canMegaEvo")) or bool(active.get("canMegaEvoX")) or bool(active.get("canMegaEvoY")):
                    slot_options.append(
                        _ChoiceOption(
                            command=f"{base_command} mega",
                            kind="move",
                            description=f"mega evolve and use {move_name}{_describe_target(target_suffix)}",
                            uses_mega=True,
                        )
                    )
                if bool(active.get("canUltraBurst")):
                    slot_options.append(
                        _ChoiceOption(
                            command=f"{base_command} ultra",
                            kind="move",
                            description=f"ultra burst and use {move_name}{_describe_target(target_suffix)}",
                            uses_ultra=True,
                        )
                    )
                if bool(active.get("canTerastallize")):
                    slot_options.append(
                        _ChoiceOption(
                            command=f"{base_command} terastallize",
                            kind="move",
                            description=f"terastallize and use {move_name}{_describe_target(target_suffix)}",
                            uses_tera=True,
                        )
                    )
                if bool(active.get("canDynamax")):
                    slot_options.append(
                        _ChoiceOption(
                            command=f"{base_command} max",
                            kind="move",
                            description=f"dynamax and use {move_name}{_describe_target(target_suffix)}",
                            uses_max=True,
                        )
                    )

            if move_slot <= len(zmoves) and isinstance(zmoves[move_slot - 1], Mapping):
                zmove = cast(Mapping[str, Any], zmoves[move_slot - 1])
                zmove_name = str(zmove.get("move", move_name))
                zmove_target = str(zmove.get("target", target))
                for target_suffix in _slot_target_suffixes(zmove_target, actor_index, active_count):
                    command = f"move {move_slot}"
                    if target_suffix is not None:
                        command = f"{command} {target_suffix}"
                    slot_options.append(
                        _ChoiceOption(
                            command=f"{command} zmove",
                            kind="move",
                            description=f"use Z-Move {zmove_name}{_describe_target(target_suffix)}",
                            uses_zmove=True,
                        )
                    )

        trapped = bool(active.get("trapped"))
        if not trapped:
            for slot in range(1, len(pokemon) + 1):
                candidate = pokemon[slot - 1]
                if bool(candidate.get("active")) or _condition_is_fainted(str(candidate.get("condition", ""))):
                    continue
                label = _pokemon_label(candidate)
                slot_options.append(
                    _ChoiceOption(
                        command=f"switch {slot}",
                        kind="switch",
                        description=f"switch to {label}",
                        switch_slot=slot,
                    )
                )

        options_per_slot.append(slot_options)

    return _combine_choice_options(options_per_slot)


def _combine_choice_options(options_per_slot: Sequence[Sequence[_ChoiceOption]]) -> list[LegalAction]:
    actions_by_command: dict[str, LegalAction] = {}

    def backtrack(
        index: int,
        chosen: list[_ChoiceOption],
        used_switches: set[int],
        used_mega: bool,
        used_ultra: bool,
        used_zmove: bool,
        used_max: bool,
        used_tera: bool,
    ) -> None:
        if index >= len(options_per_slot):
            commands = ", ".join(option.command for option in chosen)
            kinds = {option.kind for option in chosen if option.kind != "pass"}
            kind: ActionKind = "pass" if not kinds else cast(ActionKind, kinds.pop() if len(kinds) == 1 else "mixed")
            description = "; ".join(option.description for option in chosen)
            actions_by_command[commands] = LegalAction(command=commands, kind=kind, description=description)
            return

        for option in options_per_slot[index]:
            if option.switch_slot is not None and option.switch_slot in used_switches:
                continue
            if option.uses_mega and used_mega:
                continue
            if option.uses_ultra and used_ultra:
                continue
            if option.uses_zmove and used_zmove:
                continue
            if option.uses_max and used_max:
                continue
            if option.uses_tera and used_tera:
                continue
            next_switches = set(used_switches)
            if option.switch_slot is not None:
                next_switches.add(option.switch_slot)
            backtrack(
                index + 1,
                chosen + [option],
                next_switches,
                used_mega or option.uses_mega,
                used_ultra or option.uses_ultra,
                used_zmove or option.uses_zmove,
                used_max or option.uses_max,
                used_tera or option.uses_tera,
            )

    backtrack(0, [], set(), False, False, False, False, False)
    return list(actions_by_command.values())


def _state_snapshot(state: BattleState) -> dict[str, Any]:
    return {
        "battle_id": state.battle_id,
        "formatid": state.formatid,
        "turns": state.turns,
        "done": state.done,
        "winner": state.winner,
        "winner_name": state.winner_name,
        "pending_players": state.pending_players,
        "errors": state.errors,
        "requests": state.requests,
    }


def _print_json(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _trace_pending_decisions(battle: SelfPlayBattle) -> dict[PlayerId, LegalAction]:
    actions: dict[PlayerId, LegalAction] = {}
    for player in battle.state.pending_players:
        request = battle.state.request_for(player)
        legal_actions = battle.possible_actions(player)
        if request is None or not legal_actions:
            raise BridgeError(f"No legal actions exposed for {player}: {request}")
        _print_json(f"{player} request", request)
        _print_json(
            f"{player} legal actions",
            [asdict(action) for action in legal_actions],
        )
        actions[player] = legal_actions[0]
    _print_json(
        "chosen actions",
        {player: asdict(action) for player, action in actions.items()},
    )
    return actions


def _demo(
    *,
    player_team: TeamSpec | None = None,
    opponent_team: TeamSpec | None = None,
    num_battles: int = 1,
    verbose: bool = False,
    print_log: bool = False,
    simple_state: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    client = OUSelfPlayClient(repo_root=repo_root)
    try:
        summaries: list[dict[str, str | int | None]] = []
        for battle_index in range(1, num_battles + 1):
            battle = client.create_battle(
                include_log=print_log or verbose,
                p1team=player_team,
                p2team=opponent_team,
                simple_state=simple_state,
            )
            try:
                if verbose:
                    _print_json(f"battle {battle_index} initial state", _state_snapshot(battle.state))
                while not battle.state.done:
                    if verbose:
                        _print_json(f"battle {battle_index} pre-step state", _state_snapshot(battle.state))
                        actions = _trace_pending_decisions(battle)
                    else:
                        actions = {}
                        for player in battle.state.pending_players:
                            legal_actions = battle.possible_actions(player)
                            if not legal_actions:
                                raise BridgeError(
                                    f"No legal actions exposed for {player}: {battle.state.request_for(player)}"
                                )
                            actions[player] = legal_actions[0]
                    battle.step(actions)
                    if verbose:
                        _print_json(f"battle {battle_index} post-step state", _state_snapshot(battle.state))
                        if battle.state.logs:
                            _print_json(f"battle {battle_index} log delta", battle.state.logs)

                summary: dict[str, str | int | None] = {
                    "battle_index": battle_index,
                    "battle_id": battle.state.battle_id,
                    "winner": battle.state.winner,
                    "winner_name": battle.state.winner_name,
                    "turns": battle.state.turns,
                }
                summaries.append(summary)
                _print_json(f"battle {battle_index} result", summary)
                if print_log and not verbose and battle.state.logs:
                    _print_json(f"battle {battle_index} final log delta", battle.state.logs)
            finally:
                battle.close()

        total_turns = sum(summary["turns"] for summary in summaries if isinstance(summary["turns"], int))
        aggregate: dict[str, int | float] = {
            "battles": len(summaries),
            "p1_wins": sum(1 for summary in summaries if summary["winner"] == "p1"),
            "p2_wins": sum(1 for summary in summaries if summary["winner"] == "p2"),
            "ties": sum(1 for summary in summaries if summary["winner"] == "tie"),
            "average_turns": round(
                total_turns / max(len(summaries), 1),
                2,
            ),
        }
        _print_json("aggregate results", aggregate)
    finally:
        client.close()


def _resolve_player_id(player_text: str) -> PlayerId:
    normalized = player_text.strip().lower()
    if normalized not in PLAYER_IDS:
        raise ValueError(f"Unknown player '{player_text}'. Use p1 or p2.")
    return normalized


def _rl_debug_snapshot(battle: SelfPlayBattle) -> dict[str, Any]:
    return {
        "state": _state_snapshot(battle.state),
        "cursor": battle.cursor,
        "history_length": battle.max_cursor,
        "pending_decisions": {
            player: [asdict(action) for action in battle.possible_actions(player)]
            for player in battle.state.pending_players
        },
    }


def _player_view_snapshot(battle: SelfPlayBattle, player: PlayerId) -> dict[str, Any]:
    request = battle.state.request_for(player)
    return {
        "player": player,
        "request": request,
        "legal_actions": [asdict(action) for action in battle.possible_actions(player)],
    }


def _print_history(battle: SelfPlayBattle) -> None:
    payload: dict[str, Any] = {
        "cursor": battle.cursor,
        "history_length": battle.max_cursor,
        "records": [asdict(record) for record in battle.history],
    }
    _print_json("action history", payload)


def _choose_by_index(battle: SelfPlayBattle, player: PlayerId, index: int) -> BattleState:
    legal = battle.possible_actions(player)
    if index < 0 or index >= len(legal):
        raise ValueError(f"Action index {index} out of bounds for {player}. Range is [0, {len(legal) - 1}].")
    return battle.choose(player, legal[index])


def _simulate_by_index(battle: SelfPlayBattle, player: PlayerId, index: int) -> BattleState:
    legal = battle.possible_actions(player)
    if index < 0 or index >= len(legal):
        raise ValueError(f"Action index {index} out of bounds for {player}. Range is [0, {len(legal) - 1}].")
    return battle.simulate_choose(player, legal[index])


def _print_debug_help() -> None:
    print(
        "\nDebug commands:\n"
        "  help                                 Show this help text\n"
        "  summary                              Show compact battle summary\n"
        "  rl                                   Show RL-facing state and legal actions\n"
        "  state [all|p1|p2]                    Show all state or standard player view\n"
        "  requests [all|p1|p2]                 Show raw request payload(s)\n"
        "  pokemon [all|p1|p2]                  Show detailed pokemon payload(s)\n"
        "  actions <p1|p2> [limit]              List legal actions for a player\n"
        "  choose <p1|p2> <action command>      Commit an action (branches if rewound)\n"
        "  choosei <p1|p2> <index>              Commit action by index\n"
        "  stepauto                             Commit first legal action for each pending player\n"
        "  simulate <p1|p2> <action command>    Simulate action without mutating live state\n"
        "  simulatei <p1|p2> <index>            Simulate indexed action without mutating\n"
        "  simstepauto                          Simulate auto step without mutating\n"
        "  undo [steps]                         Rewind action cursor (default 1)\n"
        "  redo [steps]                         Replay forward in history (default 1)\n"
        "  goto <cursor>                        Restore exact action cursor\n"
        "  history                              Show action history and cursor\n"
        "  refresh                              Pull latest state from bridge\n"
        "  logs [count]                         Print recent log chunks (default all current delta)\n"
        "  quit | exit                          End debug session\n"
    )


def _debug_cli(
    *,
    player_team: TeamSpec | None = None,
    opponent_team: TeamSpec | None = None,
    formatid: str = "gen9ou",
    include_log: bool = True,
    simple_state: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    client = OUSelfPlayClient(repo_root=repo_root)
    try:
        battle = client.create_battle(
            formatid=formatid,
            include_log=include_log,
            p1team=player_team,
            p2team=opponent_team,
            simple_state=simple_state,
        )
        try:
            print("Interactive self-play debug CLI started. Type 'help' for commands.")
            while True:
                line = input("rl-debug> ").strip()
                if not line:
                    continue
                try:
                    parts = shlex.split(line)
                except ValueError as exc:
                    print(f"Parse error: {exc}")
                    continue
                command = parts[0].lower()
                args = parts[1:]

                try:
                    if command in {"quit", "exit"}:
                        break
                    if command == "help":
                        _print_debug_help()
                        continue
                    if command == "summary":
                        _print_json("summary", _state_snapshot(battle.state))
                        print(f"cursor={battle.cursor} / history={battle.max_cursor}")
                        continue
                    if command == "rl":
                        _print_json("rl interface", _rl_debug_snapshot(battle))
                        continue
                    if command == "state":
                        scope = args[0].lower() if args else "all"
                        if scope == "all":
                            _print_json("full state", _state_snapshot(battle.state))
                        else:
                            player = _resolve_player_id(scope)
                            _print_json(f"{player} player-visible state", _player_view_snapshot(battle, player))
                        continue
                    if command == "requests":
                        scope = args[0].lower() if args else "all"
                        if scope == "all":
                            _print_json("requests", battle.state.requests)
                        else:
                            player = _resolve_player_id(scope)
                            _print_json(f"{player} request", battle.state.request_for(player))
                        continue
                    if command == "pokemon":
                        scope = args[0].lower() if args else "all"
                        if scope == "all":
                            _print_json("pokemon details", battle.state.pokemon_details())
                        else:
                            player = _resolve_player_id(scope)
                            _print_json(f"{player} pokemon", battle.state.side_pokemon(player))
                        continue
                    if command == "actions":
                        if not args:
                            raise ValueError("Usage: actions <p1|p2> [limit]")
                        player = _resolve_player_id(args[0])
                        limit = int(args[1]) if len(args) > 1 else 30
                        legal = [asdict(action) for action in battle.possible_actions(player)]
                        _print_json(f"{player} legal actions", legal[: max(0, limit)])
                        print(f"total_actions={len(legal)}")
                        continue
                    if command == "choose":
                        if len(args) < 2:
                            raise ValueError("Usage: choose <p1|p2> <action command>")
                        player = _resolve_player_id(args[0])
                        action_text = " ".join(args[1:])
                        battle.choose(player, action_text)
                        _print_json("state after choose", _state_snapshot(battle.state))
                        continue
                    if command == "choosei":
                        if len(args) != 2:
                            raise ValueError("Usage: choosei <p1|p2> <index>")
                        player = _resolve_player_id(args[0])
                        index = int(args[1])
                        _choose_by_index(battle, player, index)
                        _print_json("state after choose", _state_snapshot(battle.state))
                        continue
                    if command == "stepauto":
                        auto_actions: dict[PlayerId, LegalAction] = {}
                        for player in battle.state.pending_players:
                            legal = battle.possible_actions(player)
                            if legal:
                                auto_actions[player] = legal[0]
                        if not auto_actions:
                            raise BridgeError("No pending players with legal actions")
                        battle.step(auto_actions)
                        _print_json("state after stepauto", _state_snapshot(battle.state))
                        continue
                    if command == "simulate":
                        if len(args) < 2:
                            raise ValueError("Usage: simulate <p1|p2> <action command>")
                        player = _resolve_player_id(args[0])
                        action_text = " ".join(args[1:])
                        simulated = battle.simulate_choose(player, action_text)
                        _print_json("simulated next state", _state_snapshot(simulated))
                        _print_json("live state (unchanged)", _state_snapshot(battle.state))
                        continue
                    if command == "simulatei":
                        if len(args) != 2:
                            raise ValueError("Usage: simulatei <p1|p2> <index>")
                        player = _resolve_player_id(args[0])
                        index = int(args[1])
                        simulated = _simulate_by_index(battle, player, index)
                        _print_json("simulated next state", _state_snapshot(simulated))
                        _print_json("live state (unchanged)", _state_snapshot(battle.state))
                        continue
                    if command == "simstepauto":
                        auto_actions: dict[PlayerId, LegalAction] = {}
                        for player in battle.state.pending_players:
                            legal = battle.possible_actions(player)
                            if legal:
                                auto_actions[player] = legal[0]
                        if not auto_actions:
                            raise BridgeError("No pending players with legal actions")
                        simulated = battle.simulate_step(auto_actions)
                        _print_json("simulated next state", _state_snapshot(simulated))
                        _print_json("live state (unchanged)", _state_snapshot(battle.state))
                        continue
                    if command == "undo":
                        steps = int(args[0]) if args else 1
                        battle.undo(steps)
                        _print_json("state after undo", _state_snapshot(battle.state))
                        continue
                    if command == "redo":
                        steps = int(args[0]) if args else 1
                        battle.redo(steps)
                        _print_json("state after redo", _state_snapshot(battle.state))
                        continue
                    if command == "goto":
                        if len(args) != 1:
                            raise ValueError("Usage: goto <cursor>")
                        cursor = int(args[0])
                        battle.restore(cursor)
                        _print_json("state after goto", _state_snapshot(battle.state))
                        continue
                    if command == "history":
                        _print_history(battle)
                        continue
                    if command == "refresh":
                        battle.refresh()
                        _print_json("state after refresh", _state_snapshot(battle.state))
                        continue
                    if command == "logs":
                        count = int(args[0]) if args else len(battle.state.logs)
                        if count < 0:
                            raise ValueError("count must be non-negative")
                        _print_json("log chunks", battle.state.logs[-count:] if count else [])
                        continue
                    print("Unknown command. Type 'help' for usage.")
                except (ValueError, InvalidChoiceError, BridgeError) as exc:
                    print(f"Error: {exc}")
        finally:
            battle.close()
    finally:
        client.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pokemon Showdown self-play battles from Python.")
    subparsers = parser.add_subparsers(dest="mode")

    demo = subparsers.add_parser("demo", help="Run non-interactive self-play demo battles.")
    demo.add_argument(
        "--player-team",
        type=Path,
        help="Path to a Pokemon Showdown export file for player 1.",
    )
    demo.add_argument(
        "--opponent-team",
        type=Path,
        help="Path to a Pokemon Showdown export file for player 2.",
    )
    demo.add_argument(
        "--num-battles",
        type=int,
        default=1,
        help="Number of battles to run in the demo.",
    )
    demo.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate battle states, legal actions, and chosen actions for each step.",
    )
    demo.add_argument(
        "--log",
        action="store_true",
        help="Print raw omniscient battle log deltas collected from the bridge.",
    )
    demo.add_argument(
        "--simple-state",
        action="store_true",
        help="Collapse team-preview permutations down to lead-only choices.",
    )

    debug = subparsers.add_parser("debug", help="Run interactive battle debugger with rewind/replay controls.")
    debug.add_argument(
        "--player-team",
        type=Path,
        help="Path to a Pokemon Showdown export file for player 1.",
    )
    debug.add_argument(
        "--opponent-team",
        type=Path,
        help="Path to a Pokemon Showdown export file for player 2.",
    )
    debug.add_argument(
        "--formatid",
        type=str,
        default="gen9ou",
        help="Battle format id (default: gen9ou).",
    )
    debug.add_argument(
        "--no-log",
        action="store_true",
        help="Disable omniscient log collection for the debug session.",
    )
    debug.add_argument(
        "--simple-state",
        action="store_true",
        help="Collapse team-preview permutations down to lead-only choices.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if not argv_list or argv_list[0].startswith("-"):
        argv_list = ["demo", *argv_list]
    args = parser.parse_args(argv_list)
    try:
        if args.mode == "debug":
            _debug_cli(
                player_team=args.player_team,
                opponent_team=args.opponent_team,
                formatid=args.formatid,
                include_log=not args.no_log,
                simple_state=args.simple_state,
            )
        else:
            if args.num_battles <= 0:
                parser.error("--num-battles must be a positive integer")
            _demo(
                player_team=args.player_team,
                opponent_team=args.opponent_team,
                num_battles=args.num_battles,
                verbose=args.verbose,
                print_log=args.log,
                simple_state=args.simple_state,
            )
    except BridgeError as exc:
        print(f"Bridge startup error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())