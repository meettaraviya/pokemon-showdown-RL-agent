"""Local browser UI server for Pokemon Showdown RL tooling.

This server wraps the existing self-play bridge, feature extraction, and
training/evaluation entrypoints behind a small JSON API and serves a static UI.
It is intentionally standalone so it can be run locally without modifying the
main Showdown server process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Literal, cast
from urllib.parse import parse_qs, quote, urlparse
import uuid

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

try:
	from ..ou_selfplay_client import OUSelfPlayClient, PlayerId, SelfPlayBattle
	from .action_features import encode_legal_actions
	from .environment import SingleAgentBattleEnv
	from .features import build_battle_features
	from .planning import pick_action_with_model
	from .train import PPOEvalConfig, PPOTrainConfig, _load_model_for_inference, evaluate_ppo_agent, train_on_policy_ppo
except ImportError:  # pragma: no cover - allows script-style execution
	from tools.python.ou_selfplay_client import OUSelfPlayClient, PlayerId, SelfPlayBattle
	from tools.python.rl.action_features import encode_legal_actions
	from tools.python.rl.environment import SingleAgentBattleEnv
	from tools.python.rl.features import build_battle_features
	from tools.python.rl.planning import pick_action_with_model
	from tools.python.rl.train import (
		PPOEvalConfig,
		PPOTrainConfig,
		_load_model_for_inference,
		evaluate_ppo_agent,
		train_on_policy_ppo,
	)

STATIC_ROOT = Path(__file__).with_name("ui")
TEAM_ROOT = REPO_ROOT / "tools/python/sample_teams"
CHECKPOINT_ROOT = REPO_ROOT / "tools/python/rl/checkpoints"
PYTHON_ROOT = REPO_ROOT / "tools/python"
RELOAD_SUFFIXES = {".py", ".html", ".js", ".css"}


def _now() -> float:
	return time.time()


def _iter_reload_watch_roots(extra_paths: list[str] | None) -> list[Path]:
	roots: list[Path] = [
		Path(__file__).resolve(),
		PYTHON_ROOT,
		STATIC_ROOT,
	]
	for raw in extra_paths or []:
		path = (REPO_ROOT / raw).resolve()
		if REPO_ROOT.resolve() not in path.parents and path != REPO_ROOT.resolve():
			raise ValueError(f"reload path is outside repository root: {raw}")
		roots.append(path)
	return roots


def _collect_watch_snapshot(roots: list[Path]) -> dict[str, int]:
	snapshot: dict[str, int] = {}
	for root in roots:
		if not root.exists():
			continue
		if root.is_file():
			if root.suffix.lower() not in RELOAD_SUFFIXES:
				continue
			try:
				snapshot[str(root)] = root.stat().st_mtime_ns
			except FileNotFoundError:
				continue
			continue
		for path in root.rglob("*"):
			if not path.is_file() or path.suffix.lower() not in RELOAD_SUFFIXES:
				continue
			try:
				snapshot[str(path)] = path.stat().st_mtime_ns
			except FileNotFoundError:
				continue
	return snapshot


def _diff_watch_snapshots(before: dict[str, int], after: dict[str, int]) -> str:
	added = sorted(path for path in after if path not in before)
	removed = sorted(path for path in before if path not in after)
	changed = sorted(path for path, mtime in after.items() if before.get(path) not in (None, mtime))
	if added:
		return f"added {Path(added[0]).relative_to(REPO_ROOT).as_posix()}"
	if changed:
		return f"changed {Path(changed[0]).relative_to(REPO_ROOT).as_posix()}"
	if removed:
		return f"removed {Path(removed[0]).relative_to(REPO_ROOT).as_posix()}"
	return "filesystem change"


def _stop_child(child: subprocess.Popen[Any], timeout: float = 5.0) -> None:
	if child.poll() is not None:
		return
	child.terminate()
	try:
		child.wait(timeout=timeout)
	except subprocess.TimeoutExpired:
		child.kill()
		child.wait(timeout=timeout)


def _run_with_reloader(args: argparse.Namespace) -> int:
	watch_roots = _iter_reload_watch_roots(args.reload_path)
	watch_list = ", ".join(path.relative_to(REPO_ROOT).as_posix() for path in watch_roots if REPO_ROOT.resolve() in path.parents or path == REPO_ROOT.resolve())
	print(f"RL UI reload mode enabled (interval={args.reload_interval:.2f}s)")
	print(f"Watching: {watch_list}")

	base_cmd = [
		sys.executable,
		str(Path(__file__).resolve()),
		"--host",
		str(args.host),
		"--port",
		str(args.port),
	]

	child: subprocess.Popen[Any] | None = None
	snapshot = _collect_watch_snapshot(watch_roots)
	try:
		while True:
			if child is None or child.poll() is not None:
				if child is not None:
					code = child.returncode
					print(f"RL UI worker exited with code {code}; restarting...")
				child = subprocess.Popen(base_cmd, cwd=str(REPO_ROOT), env=os.environ.copy())
				snapshot = _collect_watch_snapshot(watch_roots)

			time.sleep(max(0.2, float(args.reload_interval)))
			new_snapshot = _collect_watch_snapshot(watch_roots)
			if new_snapshot == snapshot:
				continue

			reason = _diff_watch_snapshots(snapshot, new_snapshot)
			print(f"Detected {reason}; reloading RL UI worker...")
			snapshot = new_snapshot
			if child is not None:
				_stop_child(child)
			child = None
	except KeyboardInterrupt:
		pass
	finally:
		if child is not None:
			_stop_child(child)
	return 0


def _team_paths() -> list[Path]:
	teams: list[Path] = []
	if not TEAM_ROOT.exists():
		return teams
	for path in sorted(TEAM_ROOT.rglob("*.txt")):
		if path.name.startswith("_"):
			continue
		teams.append(path)
	return teams


def _checkpoint_paths() -> list[Path]:
	if not CHECKPOINT_ROOT.exists():
		return []
	return sorted(CHECKPOINT_ROOT.rglob("*.pt"))


def _repo_relative(path: Path) -> str:
	return path.relative_to(REPO_ROOT).as_posix()


def _json_safe(value: Any) -> Any:
	if isinstance(value, Path):
		try:
			return _repo_relative(value)
		except ValueError:
			return value.as_posix()
	if isinstance(value, dict):
		return {key: _json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(item) for item in value]
	if is_dataclass(value):
		return _json_safe(asdict(value))
	return value


def _serialize_team_file(path: Path) -> dict[str, Any]:
	preview = path.read_text(encoding="utf-8", errors="ignore").splitlines()
	first_line = next((line.strip() for line in preview if line.strip()), path.stem)
	return {
		"id": _repo_relative(path),
		"label": f"{path.parent.name}/{path.name}",
		"headline": first_line,
	}


def _serialize_checkpoint(path: Path) -> dict[str, str]:
	return {
		"id": _repo_relative(path),
		"label": path.name,
		"path": _repo_relative(path),
	}


def _coerce_path(value: str | None) -> Path | None:
	if not value:
		return None
	path = (REPO_ROOT / value).resolve()
	if not path.exists():
		raise FileNotFoundError(f"Path does not exist: {value}")
	if REPO_ROOT.resolve() not in path.parents and path != REPO_ROOT.resolve():
		raise ValueError(f"Path is outside repository root: {value}")
	return path


def _resolve_repo_path(value: str | None) -> Path | None:
	if not value:
		return None
	path = (REPO_ROOT / value).resolve()
	if REPO_ROOT.resolve() not in path.parents and path != REPO_ROOT.resolve():
		raise ValueError(f"Path is outside repository root: {value}")
	return path


def _resolve_team_input(path_value: str | None, text_value: str | None) -> str | Path | None:
	if text_value and text_value.strip():
		return text_value.strip()
	return _coerce_path(path_value)


def _terminal_reward(state: Any, player: PlayerId) -> float:
	if not state.done:
		return 0.0
	if state.winner == player:
		return 1.0
	if state.winner in (None, "tie"):
		return 0.0
	return -1.0


@dataclass
class ManagedJob:
	"""Background train/eval job tracked by the UI server."""

	id: str
	kind: Literal["ppo-train", "ppo-eval"]
	params: dict[str, Any]
	status: Literal["queued", "running", "completed", "failed"] = "queued"
	created_at: float = field(default_factory=_now)
	started_at: float | None = None
	finished_at: float | None = None
	result: dict[str, Any] | None = None
	error: str | None = None
	log_file: Path | None = None

	def to_payload(self) -> dict[str, Any]:
		payload = _json_safe(asdict(self))
		if self.log_file is not None:
			payload["log_url"] = f"/api/jobs/{self.id}/log"
			payload["log_file_abs"] = str(self.log_file)
			payload["vscode_url"] = f"vscode://file/{quote(str(self.log_file))}"
		return payload


class ManagedBattle:
	"""In-memory battle session with optional model-backed agent policies."""

	def __init__(
		self,
		*,
		battle_id: str,
		client: OUSelfPlayClient,
		battle: SelfPlayBattle,
		checkpoint_path: Path | None,
		max_actions: int,
		agent_strategy: str,
		human_player: PlayerId | None,
		gamma: float,
		planning_top_k: int,
		planning_depth: int,
	) -> None:
		self.id = battle_id
		self.client = client
		self.battle = battle
		self.checkpoint_path = checkpoint_path
		self.max_actions = max_actions
		self.agent_strategy = agent_strategy
		self.human_player = human_player
		self.gamma = gamma
		self.planning_top_k = planning_top_k
		self.planning_depth = planning_depth
		self.created_at = _now()
		self.lock = threading.RLock()
		self.model: Any | None = None
		self.device: Any | None = None
		if checkpoint_path is not None:
			import torch

			checkpoint = torch.load(checkpoint_path, map_location="cpu")
			self.model, self.device = _load_model_for_inference(checkpoint, device="cpu")

	def close(self) -> None:
		with self.lock:
			self.battle.close()
			self.client.close()

	def _policy_action_index(self, state: Any, player: PlayerId, *, sample: bool) -> int:
		legal_actions = tuple(state.possible_actions(player))
		if not legal_actions:
			raise ValueError(f"No legal actions available for {player}")
		if self.model is None:
			return 0

		import torch

		observation = build_battle_features(state, player)
		encoded = encode_legal_actions(legal_actions, max_actions=self.max_actions, request=observation.request)
		scalar = torch.from_numpy(observation.scalar_features).float().unsqueeze(0).to(self.device)
		party = torch.from_numpy(observation.party_features).float().unsqueeze(0).to(self.device)
		action_vectors = torch.from_numpy(encoded.vectors).float().unsqueeze(0).to(self.device)
		action_mask = torch.from_numpy(encoded.mask).bool().unsqueeze(0).to(self.device)

		with torch.no_grad():
			logits, _ = self.model(scalar, party, action_vectors, action_mask)
		valid_count = min(len(legal_actions), self.max_actions)
		valid_logits = logits[0, :valid_count]
		if sample:
			dist = torch.distributions.Categorical(logits=valid_logits)
			return int(dist.sample().item())
		return int(torch.argmax(valid_logits).item())

	def _planning_opponent_index(self, state: Any, player: PlayerId, legal_actions: tuple[Any, ...]) -> int:
		del legal_actions
		return self._policy_action_index(state, player, sample=False)

	def _agent_action(self, state: Any, player: PlayerId):
		legal_actions = tuple(state.possible_actions(player))
		if not legal_actions:
			raise ValueError(f"No legal actions available for {player}")
		if self.model is None:
			return legal_actions[0]
		if self.agent_strategy == "planned":
			planning_env = SingleAgentBattleEnv(
				battle=self.battle.fork(),
				player_id=player,
				opponent_policy=self._planning_opponent_index,
			)
			try:
				planned = pick_action_with_model(
					env=planning_env,
					model=self.model,
					device=self.device,
					max_actions=self.max_actions,
					top_k=self.planning_top_k,
					depth=self.planning_depth,
					gamma=self.gamma,
				)
				index = planned.action_index
			finally:
				planning_env.battle.close()
		else:
			index = self._policy_action_index(state, player, sample=self.agent_strategy == "sample")
		return legal_actions[index]

	def _advance_until_human_or_done(self) -> None:
		while not self.battle.state.done:
			pending = list(self.battle.state.pending_players)
			if self.human_player is not None and self.human_player in pending:
				return
			agent_players = [player for player in pending if player != self.human_player]
			if not agent_players:
				return
			for player in agent_players:
				action = self._agent_action(self.battle.state, player)
				self.battle.choose(player, action)
				if self.battle.state.done:
					return
				if self.human_player is not None and self.human_player in self.battle.state.pending_players:
					return

	def advance(self, mode: Literal["action", "turn", "battle"] = "turn", count: int = 1) -> None:
		with self.lock:
			for _ in range(max(1, count)):
				if self.battle.state.done:
					return
				turn_before = self.battle.state.turns
				actions_taken = 0
				while not self.battle.state.done:
					pending = [player for player in self.battle.state.pending_players if player != self.human_player]
					if not pending:
						return
					for player in pending:
						action = self._agent_action(self.battle.state, player)
						self.battle.choose(player, action)
						actions_taken += 1
						if mode == "action" and actions_taken >= 1:
							return
						if self.battle.state.done:
							return
						if self.human_player is not None and self.human_player in self.battle.state.pending_players:
							return
						if mode == "turn" and self.battle.state.turns > turn_before:
							break
					if mode == "turn" and self.battle.state.turns > turn_before:
						break
				if mode in {"action", "turn"}:
					return

	def choose(self, player: PlayerId, action: str) -> None:
		with self.lock:
			self.battle.choose(player, action)
			self._advance_until_human_or_done()

	def snapshot_payload(self, cursor: int | None = None, player: PlayerId | None = None) -> dict[str, Any]:
		with self.lock:
			live_cursor = self.battle.cursor
			requested_cursor = live_cursor if cursor is None else max(0, min(cursor, self.battle.max_cursor))
			state = self.battle.state if requested_cursor == live_cursor else self.battle.state_at(requested_cursor)
			replay_lines = self._replay_lines_up_to_cursor(requested_cursor)
			payload: dict[str, Any] = {
				"id": self.id,
				"cursor": requested_cursor,
				"live_cursor": live_cursor,
				"max_cursor": self.battle.max_cursor,
				"human_player": self.human_player,
				"checkpoint": _repo_relative(self.checkpoint_path) if self.checkpoint_path else None,
				"state": {
					"battle_id": state.battle_id,
					"formatid": state.formatid,
					"turns": state.turns,
					"done": state.done,
					"winner": state.winner,
					"winner_name": state.winner_name,
					"pending_players": state.pending_players,
					"logs": state.logs,
					"requests": state.requests,
					"errors": state.errors,
					"pokemon": {battle_player: state.side_pokemon(battle_player) for battle_player in cast(tuple[PlayerId, PlayerId], ("p1", "p2"))},
				},
				"replay": {
					"formatid": state.formatid,
					"lines": replay_lines,
				},
				"history": [asdict(record) for record in self.battle.history],
			}

			if player in ("p1", "p2"):
				payload["analysis"] = self._analysis_for_state(state, player, requested_cursor)
			return payload

	def _replay_lines_up_to_cursor(self, cursor: int) -> list[str]:
		"""Build replay protocol lines up to a given cursor from snapshot log chunks."""
		snapshots = cast(list[Any], getattr(self.battle, "_snapshots", []))
		if not snapshots:
			return []

		max_index = min(cursor, len(snapshots) - 1)
		lines: list[str] = []
		for index in range(max_index + 1):
			snapshot = snapshots[index]
			for chunk in cast(list[str], getattr(snapshot, "logs", []) or []):
				for line in str(chunk).splitlines():
					if not line or not line.startswith("|"):
						continue
					lines.append(line)
		return lines

	def _analysis_for_state(self, state: Any, player: PlayerId, cursor: int) -> dict[str, Any]:
		observation = build_battle_features(state, player)
		analysis: dict[str, Any] = {
			"player": player,
			"request_kind": observation.request_kind,
			"scalar_features": observation.scalar_features.tolist(),
			"party_features": observation.party_features.tolist(),
			"legal_actions": [
				{
					"index": index,
					"command": action.command,
					"description": action.description,
				}
				for index, action in enumerate(observation.legal_actions)
			],
		}
		if self.model is not None:
			analysis.update(self._model_scores(observation, cursor, player))
		return analysis

	def _resolve_branch_pending_actions(self, branch: SelfPlayBattle, player: PlayerId) -> Any:
		"""Advance forked branch by resolving non-player pending choices with greedy policy."""
		while not branch.state.done:
			pending = [candidate for candidate in branch.state.pending_players if candidate != player]
			if not pending:
				break
			for candidate in pending:
				legal_actions = tuple(branch.state.possible_actions(candidate))
				if not legal_actions:
					continue
				if self.model is None:
					action = legal_actions[0]
				else:
					index = self._policy_action_index(branch.state, candidate, sample=False)
					index = max(0, min(index, len(legal_actions) - 1))
					action = legal_actions[index]
				branch.choose(candidate, action, validate=False)
				if branch.state.done:
					break
			if player in branch.state.pending_players:
				break
		return branch.state

	def _model_scores(self, observation: Any, cursor: int, player: PlayerId) -> dict[str, Any]:
		import torch

		legal_actions = tuple(observation.legal_actions)
		encoded = encode_legal_actions(legal_actions, max_actions=self.max_actions, request=observation.request)
		scalar = torch.from_numpy(observation.scalar_features).float().unsqueeze(0).to(self.device)
		party = torch.from_numpy(observation.party_features).float().unsqueeze(0).to(self.device)
		action_vectors = torch.from_numpy(encoded.vectors).float().unsqueeze(0).to(self.device)
		action_mask = torch.from_numpy(encoded.mask).bool().unsqueeze(0).to(self.device)
		with torch.no_grad():
			logits, value = self.model(scalar, party, action_vectors, action_mask)
		valid_count = min(len(legal_actions), self.max_actions)
		valid_logits = logits[0, :valid_count]
		probabilities = torch.softmax(valid_logits, dim=-1) if valid_count else torch.empty(0)
		action_rows: list[dict[str, Any]] = []
		for index, action in enumerate(legal_actions[:valid_count]):
			next_value = 0.0
			one_step_q = None
			branch = self.battle.fork_at(cursor)
			try:
				branch.choose(player, action, validate=False)
				next_state = self._resolve_branch_pending_actions(branch, player)
				terminal_reward = _terminal_reward(next_state, player)
				if not next_state.done:
					next_obs = build_battle_features(next_state, player)
					next_encoded = encode_legal_actions(next_obs.legal_actions, max_actions=self.max_actions, request=next_obs.request)
					next_scalar = torch.from_numpy(next_obs.scalar_features).float().unsqueeze(0).to(self.device)
					next_party = torch.from_numpy(next_obs.party_features).float().unsqueeze(0).to(self.device)
					next_action_vectors = torch.from_numpy(next_encoded.vectors).float().unsqueeze(0).to(self.device)
					next_action_mask = torch.from_numpy(next_encoded.mask).bool().unsqueeze(0).to(self.device)
					with torch.no_grad():
						_, next_value_tensor = self.model(next_scalar, next_party, next_action_vectors, next_action_mask)
					next_value = float(next_value_tensor.item())
				one_step_q = terminal_reward + self.gamma * (0.0 if next_state.done else next_value)
			finally:
				branch.close()
			action_rows.append(
				{
					"index": index,
					"command": action.command,
					"description": action.description,
					"policy_logit": float(valid_logits[index].item()),
					"policy_probability": float(probabilities[index].item()) if valid_count else 0.0,
					"one_step_q_estimate": one_step_q,
					"next_state_value": next_value,
				}
			)
		return {
			"state_value": float(value.item()),
			"action_scores": action_rows,
		}


class AppState:
	"""Shared mutable state for the local RL UI server."""

	def __init__(self) -> None:
		self.jobs: dict[str, ManagedJob] = {}
		self.battles: dict[str, ManagedBattle] = {}
		self.lock = threading.RLock()

	def metadata(self) -> dict[str, Any]:
		return {
			"repo_root": str(REPO_ROOT),
			"teams": [_serialize_team_file(path) for path in _team_paths()],
			"checkpoints": [_serialize_checkpoint(path) for path in _checkpoint_paths()],
		}

	def _job_log_file(self, kind: Literal["ppo-train", "ppo-eval"], params: dict[str, Any], job_id: str) -> Path:
		if kind == "ppo-train":
			base = cast(Path | None, params.get("output"))
			if base is not None:
				return base.with_name(f"{base.stem}.train.log")
		else:
			base = cast(Path | None, params.get("checkpoint"))
			if base is not None:
				return base.with_name(f"{base.stem}.eval.log")
		return CHECKPOINT_ROOT / f"job-{job_id}.{kind}.log"

	def start_job(self, kind: Literal["ppo-train", "ppo-eval"], params: dict[str, Any]) -> ManagedJob:
		job = ManagedJob(id=uuid.uuid4().hex[:12], kind=kind, params=params)
		job.log_file = self._job_log_file(kind, params, job.id)
		assert job.log_file is not None
		job.log_file.parent.mkdir(parents=True, exist_ok=True)
		with self.lock:
			self.jobs[job.id] = job

		def _runner() -> None:
			log_lock = threading.Lock()

			def _write_log(line: str) -> None:
				assert job.log_file is not None
				timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
				with log_lock:
					with job.log_file.open("a", encoding="utf-8") as handle:
						handle.write(f"[{timestamp}] {line}\n")

			_write_log(f"job_id={job.id} kind={kind} status=queued")
			_write_log(f"params={json.dumps(_json_safe(params), sort_keys=True)}")
			job.status = "running"
			job.started_at = _now()
			_write_log("status=running")
			try:
				if kind == "ppo-train":
					config = PPOTrainConfig(**params)
					job.result = train_on_policy_ppo(config, progress_callback=_write_log)
				else:
					config = PPOEvalConfig(**params)
					_write_log("starting evaluation")
					job.result = evaluate_ppo_agent(config)
					_write_log("evaluation completed")
				job.status = "completed"
				_write_log("status=completed")
			except Exception as exc:  # pragma: no cover - surfaced to UI
				job.status = "failed"
				job.error = str(exc)
				_write_log(f"status=failed error={exc}")
			finally:
				job.finished_at = _now()
				_write_log(f"finished_at={job.finished_at}")

		threading.Thread(target=_runner, name=f"rl-ui-{job.id}", daemon=True).start()
		return job

	def create_battle(self, body: dict[str, Any]) -> ManagedBattle:
		checkpoint_path = _coerce_path(cast(str | None, body.get("checkpoint")))
		p1team = _resolve_team_input(cast(str | None, body.get("p1team_path")), cast(str | None, body.get("p1team_text")))
		p2team = _resolve_team_input(cast(str | None, body.get("p2team_path")), cast(str | None, body.get("p2team_text")))
		max_actions = int(body.get("max_actions", 128))
		human_player = body.get("human_player")
		if human_player not in (None, "p1", "p2"):
			raise ValueError("human_player must be 'p1', 'p2', or null")

		client = OUSelfPlayClient(REPO_ROOT)
		battle = client.create_battle(
			formatid=str(body.get("formatid", "gen9ou")),
			include_log=bool(body.get("include_log", True)),
			p1team=p1team,
			p2team=p2team,
			simple_state=bool(body.get("simple_state", True)),
		)
		managed = ManagedBattle(
			battle_id=uuid.uuid4().hex[:12],
			client=client,
			battle=battle,
			checkpoint_path=checkpoint_path,
			max_actions=max_actions,
			agent_strategy=str(body.get("agent_strategy", "greedy")),
			human_player=cast(PlayerId | None, human_player),
			gamma=float(body.get("gamma", 0.99)),
			planning_top_k=max(1, int(body.get("planning_top_k", 4))),
			planning_depth=max(1, int(body.get("planning_depth", 3))),
		)
		with self.lock:
			self.battles[managed.id] = managed
		if managed.human_player is not None:
			managed._advance_until_human_or_done()
		return managed

	def get_battle(self, battle_id: str) -> ManagedBattle:
		with self.lock:
			battle = self.battles.get(battle_id)
		if battle is None:
			raise KeyError(f"Unknown battle: {battle_id}")
		return battle

	def close_battle(self, battle_id: str) -> None:
		with self.lock:
			battle = self.battles.pop(battle_id, None)
		if battle is not None:
			battle.close()


APP_STATE = AppState()


class RLUIRequestHandler(BaseHTTPRequestHandler):
	"""HTTP request handler for the standalone RL dashboard."""

	server_version = "RLUI/0.1"

	def do_GET(self) -> None:
		parsed = urlparse(self.path)
		if parsed.path == "/api/meta":
			return self._write_json(APP_STATE.metadata())
		if parsed.path == "/api/jobs":
			return self._write_json({"jobs": [job.to_payload() for job in APP_STATE.jobs.values()]})
		if parsed.path.startswith("/api/jobs/"):
			parts = [part for part in parsed.path.split("/") if part]
			if len(parts) == 4 and parts[3] == "log":
				job_id = parts[2]
				job = APP_STATE.jobs.get(job_id)
				if job is None:
					return self._write_error(HTTPStatus.NOT_FOUND, f"Unknown job: {job_id}")
				if job.log_file is None or not job.log_file.exists():
					return self._write_error(HTTPStatus.NOT_FOUND, f"No log file available for job: {job_id}")
				self.send_response(HTTPStatus.OK)
				self.send_header("Content-Type", "text/plain; charset=utf-8")
				self.send_header("Cache-Control", "no-store")
				self.end_headers()
				self.wfile.write(job.log_file.read_bytes())
				return

			job_id = parts[-1]
			job = APP_STATE.jobs.get(job_id)
			if job is None:
				return self._write_error(HTTPStatus.NOT_FOUND, f"Unknown job: {job_id}")
			return self._write_json(job.to_payload())
		if parsed.path == "/api/battles":
			return self._write_json({"battles": [battle.snapshot_payload() for battle in APP_STATE.battles.values()]})
		if parsed.path.startswith("/api/battles/"):
			battle_id = parsed.path.split("/")[3]
			query = parse_qs(parsed.query)
			cursor = int(query["cursor"][0]) if "cursor" in query else None
			player_raw = query.get("player", [None])[0]
			player = cast(PlayerId | None, player_raw if player_raw in ("p1", "p2") else None)
			try:
				battle = APP_STATE.get_battle(battle_id)
			except KeyError as exc:
				return self._write_error(HTTPStatus.NOT_FOUND, str(exc))
			return self._write_json(battle.snapshot_payload(cursor=cursor, player=player))
		return self._serve_static(parsed.path)

	def do_POST(self) -> None:
		parsed = urlparse(self.path)
		body = self._read_json_body()
		if parsed.path == "/api/train/ppo":
			params = self._build_ppo_train_params(body)
			job = APP_STATE.start_job("ppo-train", params)
			return self._write_json(job.to_payload(), HTTPStatus.ACCEPTED)
		if parsed.path == "/api/eval/ppo":
			params = self._build_ppo_eval_params(body)
			job = APP_STATE.start_job("ppo-eval", params)
			return self._write_json(job.to_payload(), HTTPStatus.ACCEPTED)
		if parsed.path == "/api/battles":
			battle = APP_STATE.create_battle(body)
			player = cast(PlayerId | None, body.get("human_player"))
			if player not in ("p1", "p2"):
				player = cast(PlayerId, "p1")
			return self._write_json(battle.snapshot_payload(player=player))
		if parsed.path.startswith("/api/battles/"):
			parts = [part for part in parsed.path.split("/") if part]
			if len(parts) < 4:
				return self._write_error(HTTPStatus.NOT_FOUND, "Unknown API route")
			battle_id = parts[2]
			action = parts[3]
			try:
				battle = APP_STATE.get_battle(battle_id)
			except KeyError as exc:
				return self._write_error(HTTPStatus.NOT_FOUND, str(exc))
			if action == "advance":
				battle.advance(
					mode=cast(Literal["action", "turn", "battle"], body.get("mode", "turn")),
					count=int(body.get("count", 1)),
				)
				return self._write_json(battle.snapshot_payload(player=cast(PlayerId | None, body.get("player"))))
			if action == "choose":
				player = cast(PlayerId | None, body.get("player")) or battle.human_player
				if player not in ("p1", "p2"):
					return self._write_error(HTTPStatus.BAD_REQUEST, "player must be supplied for choose")
				battle.choose(player, str(body.get("action", "")))
				return self._write_json(battle.snapshot_payload(player=player))
			if action == "close":
				APP_STATE.close_battle(battle_id)
				return self._write_json({"closed": battle_id})
			if action == "restore":
				cursor = int(body.get("cursor", battle.battle.cursor))
				battle.battle.restore(cursor)
				return self._write_json(battle.snapshot_payload(player=cast(PlayerId | None, body.get("player"))))
			if action == "undo":
				steps = max(1, int(body.get("steps", 1)))
				battle.battle.undo(steps)
				return self._write_json(battle.snapshot_payload(player=cast(PlayerId | None, body.get("player"))))
			if action == "redo":
				steps = max(1, int(body.get("steps", 1)))
				battle.battle.redo(steps)
				return self._write_json(battle.snapshot_payload(player=cast(PlayerId | None, body.get("player"))))
		return self._write_error(HTTPStatus.NOT_FOUND, "Unknown API route")

	def log_message(self, format: str, *args: Any) -> None:
		return

	def _build_ppo_train_params(self, body: dict[str, Any]) -> dict[str, Any]:
		return {
			"repo_root": REPO_ROOT,
			"output": _resolve_repo_path(cast(str | None, body.get("output"))) or (CHECKPOINT_ROOT / "ppo_ui.pt"),
			"architecture": str(body.get("architecture", "deepset")),
			"hidden_dim": int(body.get("hidden_dim", 192)),
			"formatid": str(body.get("formatid", "gen9ou")),
			"player_id": str(body.get("player_id", "p1")),
			"simple_state": bool(body.get("simple_state", True)),
			"max_actions": int(body.get("max_actions", 128)),
			"iterations": int(body.get("iterations", 50)),
			"rollout_steps": int(body.get("rollout_steps", 1024)),
			"ppo_epochs": int(body.get("ppo_epochs", 4)),
			"minibatch_size": int(body.get("minibatch_size", 256)),
			"gamma": float(body.get("gamma", 0.995)),
			"gae_lambda": float(body.get("gae_lambda", 0.95)),
			"clip_ratio": float(body.get("clip_ratio", 0.2)),
			"lr": float(body.get("lr", 3e-4)),
			"value_coef": float(body.get("value_coef", 0.5)),
			"entropy_coef": float(body.get("entropy_coef", 0.01)),
			"max_grad_norm": float(body.get("max_grad_norm", 1.0)),
			"shaping_scale": float(body.get("shaping_scale", 0.4)),
			"device": str(body.get("device", "cpu")),
			"seed": int(body.get("seed", 0)),
			"strategy": str(body.get("strategy", "sample")),
			"team_selection_policy": str(body.get("team_selection_policy", "fixed")),
			"p1team": _resolve_team_input(cast(str | None, body.get("p1team_path")), cast(str | None, body.get("p1team_text"))),
			"p2team": _resolve_team_input(cast(str | None, body.get("p2team_path")), cast(str | None, body.get("p2team_text"))),
			"opponent_refresh_interval": int(body.get("opponent_refresh_interval", 0)),
			"opponent_checkpoint_dir": _resolve_repo_path(cast(str | None, body.get("opponent_checkpoint_dir"))),
		}

	def _build_ppo_eval_params(self, body: dict[str, Any]) -> dict[str, Any]:
		checkpoint = _coerce_path(cast(str | None, body.get("checkpoint")))
		if checkpoint is None:
			raise ValueError("checkpoint is required")
		return {
			"repo_root": REPO_ROOT,
			"checkpoint": checkpoint,
			"episodes": int(body.get("episodes", 20)),
			"max_actions": int(body.get("max_actions", 128)),
			"formatid": str(body.get("formatid", "gen9ou")),
			"player_id": str(body.get("player_id", "p1")),
			"simple_state": bool(body.get("simple_state", True)),
			"planning_top_k": int(body.get("planning_top_k", 4)),
			"planning_depth": max(1, int(body.get("planning_depth", 3))),
			"planning_gamma": float(body.get("planning_gamma", 0.99)),
			"strategy": str(body.get("strategy", "greedy")),
			"team_selection_policy": str(body.get("team_selection_policy", "fixed")),
			"p1team": _resolve_team_input(cast(str | None, body.get("p1team_path")), cast(str | None, body.get("p1team_text"))),
			"p2team": _resolve_team_input(cast(str | None, body.get("p2team_path")), cast(str | None, body.get("p2team_text"))),
		}

	def _read_json_body(self) -> dict[str, Any]:
		length = int(self.headers.get("Content-Length", "0"))
		if length <= 0:
			return {}
		raw = self.rfile.read(length)
		if not raw:
			return {}
		return cast(dict[str, Any], json.loads(raw.decode("utf-8")))

	def _serve_static(self, raw_path: str) -> None:
		path = "/index.html" if raw_path in {"", "/"} else raw_path
		file_path = (STATIC_ROOT / path.lstrip("/")).resolve()
		if STATIC_ROOT.resolve() not in file_path.parents and file_path != STATIC_ROOT.resolve():
			return self._write_error(HTTPStatus.FORBIDDEN, "Forbidden")
		if not file_path.exists() or not file_path.is_file():
			return self._write_error(HTTPStatus.NOT_FOUND, f"Static asset not found: {path}")
		content_type, _ = mimetypes.guess_type(str(file_path))
		self.send_response(HTTPStatus.OK)
		self.send_header("Content-Type", content_type or "application/octet-stream")
		self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
		self.send_header("Pragma", "no-cache")
		self.send_header("Expires", "0")
		self.end_headers()
		self.wfile.write(file_path.read_bytes())

	def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
		data = json.dumps(payload, indent=2).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json; charset=utf-8")
		self.send_header("Content-Length", str(len(data)))
		self.end_headers()
		self.wfile.write(data)

	def _write_error(self, status: HTTPStatus, message: str) -> None:
		self._write_json({"error": message, "status": int(status)}, status)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run the local browser UI for Pokemon Showdown RL tooling.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8765)
	parser.add_argument("--reload", action="store_true", help="Restart server automatically when backend/ui files change.")
	parser.add_argument("--reload-interval", type=float, default=0.8, help="Polling interval in seconds for reload mode.")
	parser.add_argument(
		"--reload-path",
		action="append",
		default=[],
		help="Additional repository-relative path to watch (can be passed multiple times).",
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	if args.reload:
		return _run_with_reloader(args)
	server = ThreadingHTTPServer((args.host, args.port), RLUIRequestHandler)
	print(f"RL UI listening on http://{args.host}:{args.port}")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()
		for battle_id in list(APP_STATE.battles):
			APP_STATE.close_battle(battle_id)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())