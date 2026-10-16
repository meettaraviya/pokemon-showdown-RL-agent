"""Offline/batch RL and planning entrypoint for Pokemon Showdown self-play."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TEAM_POOL_ROOT = REPO_ROOT / "tools/python/sample_teams"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ..ou_selfplay_client import OUSelfPlayClient, TeamSpec, _resolve_player_id
    from .action_features import ACTION_FEATURE_DIM, encode_legal_actions
    from .environment import SingleAgentBattleEnv
    from .features import build_battle_features, heuristic_position_score
    from .offline_dataset import OfflineDatasetBuilder, load_offline_batch, save_offline_batch
    from .planning import pick_action_with_model
except ImportError:  # pragma: no cover - allows script-style execution
    from tools.python.ou_selfplay_client import OUSelfPlayClient, TeamSpec, _resolve_player_id
    from tools.python.rl.action_features import ACTION_FEATURE_DIM, encode_legal_actions
    from tools.python.rl.environment import SingleAgentBattleEnv
    from tools.python.rl.features import build_battle_features, heuristic_position_score
    from tools.python.rl.offline_dataset import OfflineDatasetBuilder, load_offline_batch, save_offline_batch
    from tools.python.rl.planning import pick_action_with_model


@dataclass(frozen=True)
class CollectConfig:
    """CollectConfig helper type/container used by the RL self-play tooling."""
    repo_root: Path
    output: Path
    episodes: int = 200
    max_actions: int = 128
    formatid: str = "gen9ou"
    player_id: str = "p1"
    simple_state: bool = True
    strategy: str = "random"
    shaping_scale: float = 0.4
    seed: int = 0
    team_selection_policy: str = "fixed"
    p1team: TeamSpec | None = None
    p2team: TeamSpec | None = None


@dataclass(frozen=True)
class TrainConfig:
    """TrainConfig helper type/container used by the RL self-play tooling."""
    dataset: Path
    output: Path
    architecture: str = "deepset"
    hidden_dim: int = 192
    epochs: int = 25
    batch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.995
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    device: str = "cpu"
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    """EvalConfig helper type/container used by the RL self-play tooling."""
    repo_root: Path
    checkpoint: Path
    episodes: int = 20
    max_actions: int = 128
    formatid: str = "gen9ou"
    player_id: str = "p1"
    simple_state: bool = True
    planning_top_k: int = 4
    planning_depth: int = 3
    planning_gamma: float = 0.99
    strategy: str = "planned"
    team_selection_policy: str = "fixed"
    p1team: TeamSpec | None = None
    p2team: TeamSpec | None = None


@dataclass(frozen=True)
class PPOTrainConfig:
    """PPOTrainConfig helper type/container used by the RL self-play tooling."""
    repo_root: Path
    output: Path
    architecture: str = "deepset"
    hidden_dim: int = 192
    formatid: str = "gen9ou"
    player_id: str = "p1"
    simple_state: bool = True
    max_actions: int = 128
    iterations: int = 50
    rollout_steps: int = 1024
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    shaping_scale: float = 0.4
    device: str = "cpu"
    seed: int = 0
    strategy: str = "sample"
    team_selection_policy: str = "fixed"
    p1team: TeamSpec | None = None
    p2team: TeamSpec | None = None
    opponent_refresh_interval: int = 0
    opponent_checkpoint_dir: Path | None = None


@dataclass(frozen=True)
class PPOEvalConfig:
    """PPOEvalConfig helper type/container used by the RL self-play tooling."""
    repo_root: Path
    checkpoint: Path
    episodes: int = 20
    max_actions: int = 128
    formatid: str = "gen9ou"
    player_id: str = "p1"
    simple_state: bool = True
    planning_top_k: int = 4
    planning_depth: int = 3
    planning_gamma: float = 0.99
    strategy: str = "greedy"
    team_selection_policy: str = "fixed"
    p1team: TeamSpec | None = None
    p2team: TeamSpec | None = None


class _LazyTorch:
    """_LazyTorch helper type/container used by the RL self-play tooling."""
    @staticmethod
    def require() -> tuple[Any, Any, Any, Any]:
        try:
            import torch
            import torch.nn.functional as functional
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for train/eval mode. Install it in your environment first."
            ) from exc
        return torch, functional, nn, (DataLoader, TensorDataset)


def _random_action_index(legal_count: int, rng: random.Random, max_actions: int) -> int:
    """_random_action_index utility used by the RL self-play workflow."""
    if legal_count <= 0:
        raise ValueError("No legal actions available")
    return rng.randrange(0, min(legal_count, max_actions))


def _first_action_index(legal_count: int, max_actions: int) -> int:
    """_first_action_index utility used by the RL self-play workflow."""
    if legal_count <= 0:
        raise ValueError("No legal actions available")
    return 0 if max_actions > 0 else 0


def _team_pool_paths(team_root: Path = TEAM_POOL_ROOT) -> list[Path]:
    if not team_root.exists():
        return []
    return [path for path in sorted(team_root.rglob("*.txt")) if not path.name.startswith("_")]


def _select_battle_teams(
    *,
    team_selection_policy: str,
    p1team: TeamSpec | None,
    p2team: TeamSpec | None,
    rng: random.Random | None = None,
) -> tuple[TeamSpec | None, TeamSpec | None]:
    if team_selection_policy == "fixed":
        return p1team, p2team
    if team_selection_policy != "random":
        raise ValueError(f"Unsupported team_selection_policy: {team_selection_policy}")
    team_pool = _team_pool_paths()
    if not team_pool:
        raise ValueError(f"No team files found under {TEAM_POOL_ROOT}")
    chooser = rng or random
    return chooser.choice(team_pool), chooser.choice(team_pool)


def _load_model_for_inference(checkpoint: dict[str, Any], device: str) -> tuple[Any, Any]:
    """_load_model_for_inference utility used by the RL self-play workflow."""
    torch, _, _, _ = _LazyTorch.require()
    try:
        from .models import ModelConfig, OfflinePolicyValueNet
    except ImportError:  # pragma: no cover
        from tools.python.rl.models import ModelConfig, OfflinePolicyValueNet

    model_config = ModelConfig(
        scalar_dim=int(checkpoint["scalar_dim"]),
        party_feature_dim=int(checkpoint["party_feature_dim"]),
        action_feature_dim=int(checkpoint["action_feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model = OfflinePolicyValueNet(model_config, architecture=str(checkpoint["architecture"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch.device(device))
    model.eval()
    return model, torch.device(device)


def _save_policy_checkpoint(
    *,
    path: Path,
    model: Any,
    architecture: str,
    scalar_dim: int,
    party_feature_dim: int,
    action_feature_dim: int,
    hidden_dim: int,
    metadata: dict[str, Any],
) -> None:
    """_save_policy_checkpoint utility used by the RL self-play workflow."""
    torch, _, _, _ = _LazyTorch.require()
    checkpoint = {
        "mode": "ppo",
        "architecture": architecture,
        "scalar_dim": scalar_dim,
        "party_feature_dim": party_feature_dim,
        "action_feature_dim": action_feature_dim,
        "hidden_dim": hidden_dim,
        "model_state": model.state_dict(),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _build_model_from_dims(
    *,
    architecture: str,
    hidden_dim: int,
    scalar_dim: int,
    party_feature_dim: int,
    action_feature_dim: int,
) -> Any:
    """_build_model_from_dims utility used by the RL self-play workflow."""
    try:
        from .models import ModelConfig, OfflinePolicyValueNet
    except ImportError:  # pragma: no cover
        from tools.python.rl.models import ModelConfig, OfflinePolicyValueNet

    model_config = ModelConfig(
        scalar_dim=scalar_dim,
        party_feature_dim=party_feature_dim,
        action_feature_dim=action_feature_dim,
        hidden_dim=hidden_dim,
    )
    return OfflinePolicyValueNet(model_config, architecture=architecture)


def _policy_action_index(
    *,
    model: Any,
    device: Any,
    observation: Any,
    max_actions: int,
    sample: bool,
) -> tuple[int, float, float]:
    """_policy_action_index utility used by the RL self-play workflow."""
    torch, _, _, _ = _LazyTorch.require()
    legal_actions = tuple(observation.legal_actions)
    if not legal_actions:
        raise ValueError("No legal actions available")
    encoded = encode_legal_actions(legal_actions, max_actions=max_actions, request=observation.request)

    scalar = torch.from_numpy(observation.scalar_features).float().unsqueeze(0).to(device)
    party = torch.from_numpy(observation.party_features).float().unsqueeze(0).to(device)
    action_vectors = torch.from_numpy(encoded.vectors).float().unsqueeze(0).to(device)
    action_mask = torch.from_numpy(encoded.mask).bool().unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = model(scalar, party, action_vectors, action_mask)
    valid_count = min(len(legal_actions), max_actions)
    valid_logits = logits[0, :valid_count]
    dist = torch.distributions.Categorical(logits=valid_logits)
    if sample:
        action = dist.sample()
    else:
        action = torch.argmax(valid_logits)
    action_index = int(action.item())
    return action_index, float(dist.log_prob(action).item()), float(value.item())


def _frozen_opponent_policy(
    *,
    model: Any,
    device: Any,
    max_actions: int,
    sample: bool,
):
    """_frozen_opponent_policy utility used by the RL self-play workflow."""
    def _policy(state: Any, player: str, legal_actions: tuple[Any, ...]) -> int:
        if not legal_actions:
            raise ValueError("Opponent policy received no legal actions")
        observation = build_battle_features(state, player)
        action_index, _, _ = _policy_action_index(
            model=model,
            device=device,
            observation=observation,
            max_actions=max_actions,
            sample=sample,
        )
        return action_index

    return _policy


def _compute_gae(
    rewards: list[float],
    values: list[float],
    next_values: list[float],
    dones: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """_compute_gae utility used by the RL self-play workflow."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * (1.0 - dones[index]) * next_values[index] - values[index]
        gae = delta + gamma * gae_lambda * (1.0 - dones[index]) * gae
        advantages[index] = gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def _trend_over_time(curve: list[dict[str, float]], key: str) -> float | None:
    samples = [point for point in curve if isinstance(point.get(key), (int, float))]
    if len(samples) < 2:
        return None
    xs = np.asarray([float(point["elapsed_seconds"]) for point in samples], dtype=np.float64)
    ys = np.asarray([float(point[key]) for point in samples], dtype=np.float64)
    if float(xs[-1] - xs[0]) <= 1e-9:
        return None
    slope, _ = np.polyfit(xs, ys, deg=1)
    return float(slope)


def collect_offline_data(config: CollectConfig) -> dict[str, Any]:
    """collect_offline_data utility used by the RL self-play workflow."""
    rng = random.Random(config.seed)
    builder = OfflineDatasetBuilder()
    player_id = _resolve_player_id(config.player_id)

    client = OUSelfPlayClient(repo_root=config.repo_root)
    try:
        total_steps = 0
        terminal_rewards: list[float] = []
        for _ in range(config.episodes):
            p1team, p2team = _select_battle_teams(
                team_selection_policy=config.team_selection_policy,
                p1team=config.p1team,
                p2team=config.p2team,
                rng=rng,
            )
            battle = client.create_battle(
                formatid=config.formatid,
                p1team=p1team,
                p2team=p2team,
                simple_state=config.simple_state,
            )
            env = SingleAgentBattleEnv(battle=battle, player_id=player_id)
            try:
                step = env.reset()
                while not step.done:
                    legal_actions = env.legal_actions()
                    if not legal_actions:
                        break
                    if config.strategy == "first":
                        action_index = _first_action_index(len(legal_actions), config.max_actions)
                    else:
                        action_index = _random_action_index(len(legal_actions), rng, config.max_actions)

                    curr_obs = step.observation
                    curr_encoded = encode_legal_actions(curr_obs.legal_actions, config.max_actions, request=curr_obs.request)
                    next_step = env.step(action_index)
                    next_obs = next_step.observation
                    next_encoded = encode_legal_actions(next_obs.legal_actions, config.max_actions, request=next_obs.request)

                    shaped_reward = next_step.reward
                    shaped_reward += config.shaping_scale * (
                        heuristic_position_score(next_obs) - heuristic_position_score(curr_obs)
                    )

                    builder.append(
                        scalar=curr_obs.scalar_features,
                        party=curr_obs.party_features,
                        action_vectors=curr_encoded.vectors,
                        action_mask=curr_encoded.mask,
                        action_index=action_index,
                        reward=shaped_reward,
                        next_scalar=next_obs.scalar_features,
                        next_party=next_obs.party_features,
                        next_action_vectors=next_encoded.vectors,
                        next_action_mask=next_encoded.mask,
                        done=next_step.done,
                    )
                    step = next_step
                    total_steps += 1
                terminal_rewards.append(step.reward)
            finally:
                battle.close()

        batch = builder.build()
        save_offline_batch(config.output, batch)
        return {
            "dataset_path": str(config.output),
            "episodes": config.episodes,
            "samples": batch.size,
            "steps": total_steps,
            "terminal_reward_mean": float(np.mean(terminal_rewards)) if terminal_rewards else 0.0,
            "max_actions": config.max_actions,
            "action_feature_dim": ACTION_FEATURE_DIM,
            "team_selection_policy": config.team_selection_policy,
        }
    finally:
        client.close()


def train_offline(config: TrainConfig) -> dict[str, Any]:
    """train_offline utility used by the RL self-play workflow."""
    torch, functional, _, loaders = _LazyTorch.require()
    DataLoader, TensorDataset = loaders
    try:
        from .models import ModelConfig, OfflinePolicyValueNet
    except ImportError:  # pragma: no cover
        from tools.python.rl.models import ModelConfig, OfflinePolicyValueNet

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    batch = load_offline_batch(config.dataset)
    model_config = ModelConfig(
        scalar_dim=int(batch.scalar.shape[-1]),
        party_feature_dim=int(batch.party.shape[-1]),
        action_feature_dim=int(batch.action_vectors.shape[-1]),
        hidden_dim=config.hidden_dim,
    )
    model = OfflinePolicyValueNet(model_config, architecture=config.architecture)
    device = torch.device(config.device)
    model.to(device)

    dataset = TensorDataset(
        torch.from_numpy(batch.scalar).float(),
        torch.from_numpy(batch.party).float(),
        torch.from_numpy(batch.action_vectors).float(),
        torch.from_numpy(batch.action_mask).bool(),
        torch.from_numpy(batch.action_index).long(),
        torch.from_numpy(batch.reward).float(),
        torch.from_numpy(batch.next_scalar).float(),
        torch.from_numpy(batch.next_party).float(),
        torch.from_numpy(batch.next_action_vectors).float(),
        torch.from_numpy(batch.next_action_mask).bool(),
        torch.from_numpy(batch.done.astype(np.float32)).float(),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    for _ in range(config.epochs):
        model.train()
        for mini_batch in loader:
            (
                scalar,
                party,
                action_vectors,
                action_mask,
                action_index,
                reward,
                next_scalar,
                next_party,
                next_action_vectors,
                next_action_mask,
                done,
            ) = [tensor.to(device) for tensor in mini_batch]

            logits, value = model(scalar, party, action_vectors, action_mask)
            with torch.no_grad():
                _, next_value = model(next_scalar, next_party, next_action_vectors, next_action_mask)
                target_value = reward + config.gamma * (1.0 - done) * next_value

            policy_loss = functional.cross_entropy(logits, action_index)
            value_loss = functional.mse_loss(value, target_value)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))

    checkpoint = {
        "architecture": config.architecture,
        "scalar_dim": model_config.scalar_dim,
        "party_feature_dim": model_config.party_feature_dim,
        "action_feature_dim": model_config.action_feature_dim,
        "hidden_dim": model_config.hidden_dim,
        "model_state": model.state_dict(),
        "train_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "gamma": config.gamma,
        },
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, config.output)

    return {
        "checkpoint": str(config.output),
        "dataset": str(config.dataset),
        "samples": int(batch.size),
        "loss_mean": float(np.mean(losses)) if losses else 0.0,
        "policy_loss_mean": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss_mean": float(np.mean(value_losses)) if value_losses else 0.0,
    }


def evaluate_agent(config: EvalConfig) -> dict[str, Any]:
    """evaluate_agent utility used by the RL self-play workflow."""
    torch, _, _, _ = _LazyTorch.require()
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    model, device = _load_model_for_inference(checkpoint, device="cpu")
    player_id = _resolve_player_id(config.player_id)

    client = OUSelfPlayClient(repo_root=config.repo_root)
    wins = 0
    ties = 0
    turns: list[int] = []
    try:
        for _ in range(config.episodes):
            p1team, p2team = _select_battle_teams(
                team_selection_policy=config.team_selection_policy,
                p1team=config.p1team,
                p2team=config.p2team,
            )
            battle = client.create_battle(
                formatid=config.formatid,
                p1team=p1team,
                p2team=p2team,
                simple_state=config.simple_state,
            )
            env = SingleAgentBattleEnv(battle=battle, player_id=player_id)
            try:
                step = env.reset()
                while not step.done:
                    legal_actions = env.legal_actions()
                    if not legal_actions:
                        break
                    if config.strategy == "greedy":
                        obs = step.observation
                        encoded = encode_legal_actions(obs.legal_actions, config.max_actions, request=obs.request)
                        scalar = torch.from_numpy(obs.scalar_features).float().unsqueeze(0).to(device)
                        party = torch.from_numpy(obs.party_features).float().unsqueeze(0).to(device)
                        action_vectors = torch.from_numpy(encoded.vectors).float().unsqueeze(0).to(device)
                        action_mask = torch.from_numpy(encoded.mask).bool().unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits, _ = model(scalar, party, action_vectors, action_mask)
                        valid_count = min(len(obs.legal_actions), config.max_actions)
                        action_index = int(torch.argmax(logits[0, :valid_count]).item())
                    else:
                        planned = pick_action_with_model(
                            env=env,
                            model=model,
                            device=device,
                            max_actions=config.max_actions,
                            top_k=config.planning_top_k,
                            depth=max(1, config.planning_depth),
                            gamma=config.planning_gamma,
                        )
                        action_index = planned.action_index
                    step = env.step(action_index)

                winner = battle.state.winner
                if winner == player_id:
                    wins += 1
                elif winner == "tie":
                    ties += 1
                turns.append(int(battle.state.turns))
            finally:
                battle.close()
    finally:
        client.close()

    return {
        "episodes": config.episodes,
        "wins": wins,
        "ties": ties,
        "win_rate": float(wins / max(1, config.episodes)),
        "average_turns": float(np.mean(turns)) if turns else 0.0,
        "checkpoint": str(config.checkpoint),
        "strategy": config.strategy,
        "team_selection_policy": config.team_selection_policy,
    }


def train_on_policy_ppo(config: PPOTrainConfig, progress_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    """train_on_policy_ppo utility used by the RL self-play workflow."""

    def _log(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    _log(
        "ppo_train_start "
        f"iterations={config.iterations} rollout_steps={config.rollout_steps} "
        f"ppo_epochs={config.ppo_epochs} output={config.output}"
    )

    torch, functional, _, _ = _LazyTorch.require()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    rng = random.Random(config.seed)

    device = torch.device(config.device)
    player_id = _resolve_player_id(config.player_id)

    client = OUSelfPlayClient(repo_root=config.repo_root)
    try:
        probe_p1team, probe_p2team = _select_battle_teams(
            team_selection_policy=config.team_selection_policy,
            p1team=config.p1team,
            p2team=config.p2team,
            rng=rng,
        )
        probe_battle = client.create_battle(
            formatid=config.formatid,
            p1team=probe_p1team,
            p2team=probe_p2team,
            simple_state=config.simple_state,
        )
        try:
            probe_env = SingleAgentBattleEnv(battle=probe_battle, player_id=player_id)
            probe_step = probe_env.reset()
            probe_obs = probe_step.observation
            probe_encoded = encode_legal_actions(
                probe_obs.legal_actions,
                max_actions=config.max_actions,
                request=probe_obs.request,
            )
            scalar_dim = int(probe_obs.scalar_features.shape[-1])
            party_feature_dim = int(probe_obs.party_features.shape[-1])
            action_feature_dim = int(probe_encoded.vectors.shape[-1])
        finally:
            probe_battle.close()

        model = _build_model_from_dims(
            architecture=config.architecture,
            hidden_dim=config.hidden_dim,
            scalar_dim=scalar_dim,
            party_feature_dim=party_feature_dim,
            action_feature_dim=action_feature_dim,
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        opponent_model = None
        opponent_dir = config.opponent_checkpoint_dir
        if config.opponent_refresh_interval > 0:
            opponent_model = _build_model_from_dims(
                architecture=config.architecture,
                hidden_dim=config.hidden_dim,
                scalar_dim=scalar_dim,
                party_feature_dim=party_feature_dim,
                action_feature_dim=action_feature_dim,
            )
            opponent_model.to(device)
            opponent_model.load_state_dict(model.state_dict())
            if opponent_dir is None:
                opponent_dir = config.output.parent / "opponent_snapshots"

        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []
        rollout_rewards: list[float] = []
        learning_curve: list[dict[str, float]] = []

        total_steps = 0
        total_episodes = 0
        train_started = time.perf_counter()

        for iteration in range(config.iterations):
            loss_start = len(losses)
            policy_loss_start = len(policy_losses)
            value_loss_start = len(value_losses)
            entropy_start = len(entropy_values)
            reward_start = len(rollout_rewards)
            if opponent_model is not None and config.opponent_refresh_interval > 0:
                if iteration % config.opponent_refresh_interval == 0:
                    opponent_model.load_state_dict(model.state_dict())
                    if opponent_dir is not None:
                        snapshot_path = opponent_dir / f"opponent_iter_{iteration:05d}.pt"
                        _save_policy_checkpoint(
                            path=snapshot_path,
                            model=opponent_model,
                            architecture=config.architecture,
                            scalar_dim=scalar_dim,
                            party_feature_dim=party_feature_dim,
                            action_feature_dim=action_feature_dim,
                            hidden_dim=config.hidden_dim,
                            metadata={"iteration": iteration, "role": "frozen_opponent"},
                        )

            opponent_policy = None
            if opponent_model is not None:
                opponent_policy = _frozen_opponent_policy(
                    model=opponent_model,
                    device=device,
                    max_actions=config.max_actions,
                    sample=config.strategy == "sample",
                )

            scalars: list[np.ndarray] = []
            parties: list[np.ndarray] = []
            action_vectors: list[np.ndarray] = []
            action_masks: list[np.ndarray] = []
            action_indices: list[int] = []
            old_log_probs: list[float] = []
            rewards: list[float] = []
            values: list[float] = []
            next_values: list[float] = []
            dones: list[float] = []

            model.eval()
            collected_steps = 0
            while collected_steps < config.rollout_steps:
                p1team, p2team = _select_battle_teams(
                    team_selection_policy=config.team_selection_policy,
                    p1team=config.p1team,
                    p2team=config.p2team,
                    rng=rng,
                )
                battle = client.create_battle(
                    formatid=config.formatid,
                    p1team=p1team,
                    p2team=p2team,
                    simple_state=config.simple_state,
                )
                env = SingleAgentBattleEnv(battle=battle, player_id=player_id, opponent_policy=opponent_policy)
                total_episodes += 1
                try:
                    step = env.reset()
                    while not step.done and collected_steps < config.rollout_steps:
                        if not env.legal_actions():
                            break
                        obs = step.observation
                        encoded = encode_legal_actions(obs.legal_actions, config.max_actions, request=obs.request)
                        action_index, log_prob, value = _policy_action_index(
                            model=model,
                            device=device,
                            observation=obs,
                            max_actions=config.max_actions,
                            sample=config.strategy == "sample",
                        )

                        next_step = env.step(action_index)
                        next_obs = next_step.observation
                        shaped_reward = next_step.reward + config.shaping_scale * (
                            heuristic_position_score(next_obs) - heuristic_position_score(obs)
                        )

                        if next_step.done:
                            next_value = 0.0
                        else:
                            _, _, next_value = _policy_action_index(
                                model=model,
                                device=device,
                                observation=next_obs,
                                max_actions=config.max_actions,
                                sample=False,
                            )

                        scalars.append(obs.scalar_features)
                        parties.append(obs.party_features)
                        action_vectors.append(encoded.vectors)
                        action_masks.append(encoded.mask)
                        action_indices.append(action_index)
                        old_log_probs.append(log_prob)
                        rewards.append(float(shaped_reward))
                        values.append(float(value))
                        next_values.append(float(next_value))
                        dones.append(1.0 if next_step.done else 0.0)

                        rollout_rewards.append(float(shaped_reward))
                        collected_steps += 1
                        total_steps += 1
                        step = next_step
                finally:
                    battle.close()

            if not scalars:
                continue

            advantages_np, returns_np = _compute_gae(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

            scalar_t = torch.from_numpy(np.asarray(scalars, dtype=np.float32)).float().to(device)
            party_t = torch.from_numpy(np.asarray(parties, dtype=np.float32)).float().to(device)
            action_vectors_t = torch.from_numpy(np.asarray(action_vectors, dtype=np.float32)).float().to(device)
            action_masks_t = torch.from_numpy(np.asarray(action_masks, dtype=np.bool_)).bool().to(device)
            action_indices_t = torch.from_numpy(np.asarray(action_indices, dtype=np.int64)).long().to(device)
            old_log_probs_t = torch.from_numpy(np.asarray(old_log_probs, dtype=np.float32)).float().to(device)
            returns_t = torch.from_numpy(returns_np).float().to(device)
            advantages_t = torch.from_numpy(advantages_np).float().to(device)

            model.train()
            batch_size = int(action_indices_t.shape[0])
            for _ in range(config.ppo_epochs):
                permutation = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, config.minibatch_size):
                    end = min(start + config.minibatch_size, batch_size)
                    idx = permutation[start:end]

                    logits, value_pred = model(
                        scalar_t[idx],
                        party_t[idx],
                        action_vectors_t[idx],
                        action_masks_t[idx],
                    )
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(action_indices_t[idx])
                    ratio = torch.exp(new_log_probs - old_log_probs_t[idx])

                    unclipped = ratio * advantages_t[idx]
                    clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages_t[idx]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = functional.mse_loss(value_pred, returns_t[idx])
                    entropy = dist.entropy().mean()

                    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                    optimizer.step()

                    losses.append(float(loss.item()))
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropy_values.append(float(entropy.item()))

            iter_losses = losses[loss_start:]
            iter_policy_losses = policy_losses[policy_loss_start:]
            iter_value_losses = value_losses[value_loss_start:]
            iter_entropy = entropy_values[entropy_start:]
            iter_rewards = rollout_rewards[reward_start:]
            epoch_loss = float(np.mean(iter_losses)) if iter_losses else 0.0
            epoch_reward = float(np.mean(iter_rewards)) if iter_rewards else 0.0
            epoch_entropy = float(np.mean(iter_entropy)) if iter_entropy else 0.0
            learning_curve.append(
                {
                    "iteration": float(iteration + 1),
                    "elapsed_seconds": float(time.perf_counter() - train_started),
                    "steps_total": float(total_steps),
                    "episodes_total": float(total_episodes),
                    "steps_collected": float(collected_steps),
                    "loss_mean": epoch_loss,
                    "policy_loss_mean": float(np.mean(iter_policy_losses)) if iter_policy_losses else 0.0,
                    "value_loss_mean": float(np.mean(iter_value_losses)) if iter_value_losses else 0.0,
                    "entropy_mean": epoch_entropy,
                    "reward_mean": epoch_reward,
                }
            )
            _log(
                f"epoch {iteration + 1}/{config.iterations} "
                f"steps_total={total_steps} episodes_total={total_episodes} "
                f"loss_mean={epoch_loss:.6f} reward_mean={epoch_reward:.6f} entropy_mean={epoch_entropy:.6f}"
            )

        reward_slope = _trend_over_time(learning_curve, "reward_mean")
        loss_slope = _trend_over_time(learning_curve, "loss_mean")
        entropy_slope = _trend_over_time(learning_curve, "entropy_mean")
        reward_improvement = None
        if len(learning_curve) >= 2:
            reward_improvement = float(learning_curve[-1]["reward_mean"] - learning_curve[0]["reward_mean"])

        _save_policy_checkpoint(
            path=config.output,
            model=model,
            architecture=config.architecture,
            scalar_dim=scalar_dim,
            party_feature_dim=party_feature_dim,
            action_feature_dim=action_feature_dim,
            hidden_dim=config.hidden_dim,
            metadata={
                "trainer": "ppo",
                "iterations": config.iterations,
                "rollout_steps": config.rollout_steps,
                "total_steps": total_steps,
                "total_episodes": total_episodes,
                "gamma": config.gamma,
                "gae_lambda": config.gae_lambda,
                "clip_ratio": config.clip_ratio,
                "team_selection_policy": config.team_selection_policy,
                "opponent_refresh_interval": config.opponent_refresh_interval,
            },
        )
        _log(f"checkpoint_saved path={config.output}")

        result = {
            "checkpoint": str(config.output),
            "trainer": "ppo",
            "iterations": config.iterations,
            "total_steps": total_steps,
            "total_episodes": total_episodes,
            "loss_mean": float(np.mean(losses)) if losses else 0.0,
            "policy_loss_mean": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss_mean": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy_mean": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "rollout_reward_mean": float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
            "learning_curve": learning_curve,
            "learning_summary": {
                "reward_slope_per_second": reward_slope,
                "loss_slope_per_second": loss_slope,
                "entropy_slope_per_second": entropy_slope,
                "reward_improvement": reward_improvement,
                "elapsed_seconds": float(time.perf_counter() - train_started),
            },
            "team_selection_policy": config.team_selection_policy,
            "opponent_refresh_interval": config.opponent_refresh_interval,
        }
        _log("ppo_train_completed")
        return result
    finally:
        client.close()


def evaluate_ppo_agent(config: PPOEvalConfig) -> dict[str, Any]:
    """evaluate_ppo_agent utility used by the RL self-play workflow."""
    torch, _, _, _ = _LazyTorch.require()
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    model, device = _load_model_for_inference(checkpoint, device="cpu")
    player_id = _resolve_player_id(config.player_id)

    client = OUSelfPlayClient(repo_root=config.repo_root)
    wins = 0
    ties = 0
    turns: list[int] = []

    try:
        for _ in range(config.episodes):
            p1team, p2team = _select_battle_teams(
                team_selection_policy=config.team_selection_policy,
                p1team=config.p1team,
                p2team=config.p2team,
            )
            battle = client.create_battle(
                formatid=config.formatid,
                p1team=p1team,
                p2team=p2team,
                simple_state=config.simple_state,
            )
            env = SingleAgentBattleEnv(battle=battle, player_id=player_id)
            try:
                step = env.reset()
                while not step.done:
                    if not env.legal_actions():
                        break
                    if config.strategy == "planned":
                        planned = pick_action_with_model(
                            env=env,
                            model=model,
                            device=device,
                            max_actions=config.max_actions,
                            top_k=config.planning_top_k,
                            depth=max(1, config.planning_depth),
                            gamma=config.planning_gamma,
                        )
                        action_index = planned.action_index
                    else:
                        action_index, _, _ = _policy_action_index(
                            model=model,
                            device=device,
                            observation=step.observation,
                            max_actions=config.max_actions,
                            sample=config.strategy == "sample",
                        )
                    step = env.step(action_index)

                winner = battle.state.winner
                if winner == player_id:
                    wins += 1
                elif winner == "tie":
                    ties += 1
                turns.append(int(battle.state.turns))
            finally:
                battle.close()
    finally:
        client.close()

    return {
        "episodes": config.episodes,
        "wins": wins,
        "ties": ties,
        "win_rate": float(wins / max(1, config.episodes)),
        "average_turns": float(np.mean(turns)) if turns else 0.0,
        "checkpoint": str(config.checkpoint),
        "strategy": config.strategy,
        "planning_top_k": config.planning_top_k,
        "planning_depth": config.planning_depth,
        "planning_gamma": config.planning_gamma,
        "team_selection_policy": config.team_selection_policy,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """_build_arg_parser utility used by the RL self-play workflow."""
    parser = argparse.ArgumentParser(description="Offline/batch RL and planning tools for Pokemon Showdown.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    collect = subparsers.add_parser("collect", help="Generate offline dataset from self-play trajectories.")
    collect.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    collect.add_argument("--output", type=Path, default=REPO_ROOT / "tools/python/rl/data/offline_gen9ou.npz")
    collect.add_argument("--episodes", type=int, default=200)
    collect.add_argument("--max-actions", type=int, default=128)
    collect.add_argument("--formatid", type=str, default="gen9ou")
    collect.add_argument("--player-id", type=str, default="p1")
    collect.add_argument("--simple-state", action="store_true", default=True)
    collect.add_argument("--full-state", action="store_true", help="Use raw team-preview permutation state")
    collect.add_argument("--strategy", choices=["random", "first"], default="random")
    collect.add_argument("--shaping-scale", type=float, default=0.4)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--team-selection-policy", choices=["fixed", "random"], default="fixed")
    collect.add_argument("--p1team", type=Path)
    collect.add_argument("--p2team", type=Path)

    train = subparsers.add_parser("train", help="Train offline policy/value model from dataset.")
    train.add_argument("--dataset", type=Path, default=REPO_ROOT / "tools/python/rl/data/offline_gen9ou.npz")
    train.add_argument("--output", type=Path, default=REPO_ROOT / "tools/python/rl/checkpoints/policy_value.pt")
    train.add_argument("--architecture", choices=["deepset", "transformer"], default="deepset")
    train.add_argument("--hidden-dim", type=int, default=192)
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--gamma", type=float, default=0.995)
    train.add_argument("--value-coef", type=float, default=0.5)
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument("--device", type=str, default="cpu")
    train.add_argument("--seed", type=int, default=0)

    evaluate = subparsers.add_parser("eval", help="Evaluate saved checkpoint with optional planning.")
    evaluate.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--max-actions", type=int, default=128)
    evaluate.add_argument("--formatid", type=str, default="gen9ou")
    evaluate.add_argument("--player-id", type=str, default="p1")
    evaluate.add_argument("--simple-state", action="store_true", default=True)
    evaluate.add_argument("--full-state", action="store_true", help="Use raw team-preview permutation state")
    evaluate.add_argument("--planning-top-k", type=int, default=4)
    evaluate.add_argument("--planning-depth", type=int, default=3)
    evaluate.add_argument("--planning-gamma", type=float, default=0.99)
    evaluate.add_argument("--strategy", choices=["planned", "greedy"], default="planned")
    evaluate.add_argument("--team-selection-policy", choices=["fixed", "random"], default="fixed")
    evaluate.add_argument("--p1team", type=Path)
    evaluate.add_argument("--p2team", type=Path)

    ppo_train = subparsers.add_parser("ppo-train", help="Train on-policy PPO model from fresh rollouts.")
    ppo_train.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ppo_train.add_argument("--output", type=Path, default=REPO_ROOT / "tools/python/rl/checkpoints/ppo_policy_value.pt")
    ppo_train.add_argument("--architecture", choices=["deepset", "transformer"], default="deepset")
    ppo_train.add_argument("--hidden-dim", type=int, default=192)
    ppo_train.add_argument("--formatid", type=str, default="gen9ou")
    ppo_train.add_argument("--player-id", type=str, default="p1")
    ppo_train.add_argument("--simple-state", action="store_true", default=True)
    ppo_train.add_argument("--full-state", action="store_true", help="Use raw team-preview permutation state")
    ppo_train.add_argument("--max-actions", type=int, default=128)
    ppo_train.add_argument("--iterations", type=int, default=50)
    ppo_train.add_argument("--rollout-steps", type=int, default=1024)
    ppo_train.add_argument("--ppo-epochs", type=int, default=4)
    ppo_train.add_argument("--minibatch-size", type=int, default=256)
    ppo_train.add_argument("--gamma", type=float, default=0.995)
    ppo_train.add_argument("--gae-lambda", type=float, default=0.95)
    ppo_train.add_argument("--clip-ratio", type=float, default=0.2)
    ppo_train.add_argument("--lr", type=float, default=3e-4)
    ppo_train.add_argument("--value-coef", type=float, default=0.5)
    ppo_train.add_argument("--entropy-coef", type=float, default=0.01)
    ppo_train.add_argument("--max-grad-norm", type=float, default=1.0)
    ppo_train.add_argument("--shaping-scale", type=float, default=0.4)
    ppo_train.add_argument("--device", type=str, default="cpu")
    ppo_train.add_argument("--seed", type=int, default=0)
    ppo_train.add_argument("--strategy", choices=["sample", "greedy"], default="sample")
    ppo_train.add_argument("--team-selection-policy", choices=["fixed", "random"], default="fixed")
    ppo_train.add_argument("--opponent-refresh-interval", type=int, default=0)
    ppo_train.add_argument("--opponent-checkpoint-dir", type=Path)
    ppo_train.add_argument("--p1team", type=Path)
    ppo_train.add_argument("--p2team", type=Path)

    ppo_eval = subparsers.add_parser("ppo-eval", help="Evaluate PPO checkpoint without planning rollout.")
    ppo_eval.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ppo_eval.add_argument("--checkpoint", type=Path, required=True)
    ppo_eval.add_argument("--episodes", type=int, default=20)
    ppo_eval.add_argument("--max-actions", type=int, default=128)
    ppo_eval.add_argument("--formatid", type=str, default="gen9ou")
    ppo_eval.add_argument("--player-id", type=str, default="p1")
    ppo_eval.add_argument("--simple-state", action="store_true", default=True)
    ppo_eval.add_argument("--full-state", action="store_true", help="Use raw team-preview permutation state")
    ppo_eval.add_argument("--planning-top-k", type=int, default=4)
    ppo_eval.add_argument("--planning-depth", type=int, default=3)
    ppo_eval.add_argument("--planning-gamma", type=float, default=0.99)
    ppo_eval.add_argument("--strategy", choices=["greedy", "sample", "planned"], default="greedy")
    ppo_eval.add_argument("--team-selection-policy", choices=["fixed", "random"], default="fixed")
    ppo_eval.add_argument("--p1team", type=Path)
    ppo_eval.add_argument("--p2team", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    """main utility used by the RL self-play workflow."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == "collect":
            config = CollectConfig(
                repo_root=args.repo_root,
                output=args.output,
                episodes=args.episodes,
                max_actions=args.max_actions,
                formatid=args.formatid,
                player_id=args.player_id,
                simple_state=not args.full_state,
                strategy=args.strategy,
                shaping_scale=args.shaping_scale,
                seed=args.seed,
                team_selection_policy=args.team_selection_policy,
                p1team=args.p1team,
                p2team=args.p2team,
            )
            summary = collect_offline_data(config)
        elif args.mode == "train":
            config = TrainConfig(
                dataset=args.dataset,
                output=args.output,
                architecture=args.architecture,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                gamma=args.gamma,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                device=args.device,
                seed=args.seed,
            )
            summary = train_offline(config)
        elif args.mode == "eval":
            config = EvalConfig(
                repo_root=args.repo_root,
                checkpoint=args.checkpoint,
                episodes=args.episodes,
                max_actions=args.max_actions,
                formatid=args.formatid,
                player_id=args.player_id,
                simple_state=not args.full_state,
                planning_top_k=args.planning_top_k,
                planning_depth=max(1, args.planning_depth),
                planning_gamma=args.planning_gamma,
                strategy=args.strategy,
                team_selection_policy=args.team_selection_policy,
                p1team=args.p1team,
                p2team=args.p2team,
            )
            summary = evaluate_agent(config)
        elif args.mode == "ppo-train":
            config = PPOTrainConfig(
                repo_root=args.repo_root,
                output=args.output,
                architecture=args.architecture,
                hidden_dim=args.hidden_dim,
                formatid=args.formatid,
                player_id=args.player_id,
                simple_state=not args.full_state,
                max_actions=args.max_actions,
                iterations=args.iterations,
                rollout_steps=args.rollout_steps,
                ppo_epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_ratio=args.clip_ratio,
                lr=args.lr,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                shaping_scale=args.shaping_scale,
                device=args.device,
                seed=args.seed,
                strategy=args.strategy,
                team_selection_policy=args.team_selection_policy,
                p1team=args.p1team,
                p2team=args.p2team,
                opponent_refresh_interval=max(0, int(args.opponent_refresh_interval)),
                opponent_checkpoint_dir=args.opponent_checkpoint_dir,
            )
            summary = train_on_policy_ppo(config)
        else:
            config = PPOEvalConfig(
                repo_root=args.repo_root,
                checkpoint=args.checkpoint,
                episodes=args.episodes,
                max_actions=args.max_actions,
                formatid=args.formatid,
                player_id=args.player_id,
                simple_state=not args.full_state,
                planning_top_k=args.planning_top_k,
                planning_depth=max(1, args.planning_depth),
                planning_gamma=args.planning_gamma,
                strategy=args.strategy,
                team_selection_policy=args.team_selection_policy,
                p1team=args.p1team,
                p2team=args.p2team,
            )
            summary = evaluate_ppo_agent(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
