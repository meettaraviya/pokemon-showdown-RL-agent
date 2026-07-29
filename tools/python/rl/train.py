"""Offline/batch RL and planning entrypoint for Pokemon Showdown self-play."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ..ou_selfplay_client import OUSelfPlayClient, TeamSpec, _resolve_player_id
    from .action_features import ACTION_FEATURE_DIM, encode_legal_actions
    from .environment import SingleAgentBattleEnv
    from .features import heuristic_position_score
    from .offline_dataset import OfflineDatasetBuilder, load_offline_batch, save_offline_batch
    from .planning import pick_action_with_model
except ImportError:  # pragma: no cover - allows script-style execution
    from tools.python.ou_selfplay_client import OUSelfPlayClient, TeamSpec, _resolve_player_id
    from tools.python.rl.action_features import ACTION_FEATURE_DIM, encode_legal_actions
    from tools.python.rl.environment import SingleAgentBattleEnv
    from tools.python.rl.features import heuristic_position_score
    from tools.python.rl.offline_dataset import OfflineDatasetBuilder, load_offline_batch, save_offline_batch
    from tools.python.rl.planning import pick_action_with_model


@dataclass(frozen=True)
class CollectConfig:
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
    p1team: TeamSpec | None = None
    p2team: TeamSpec | None = None


@dataclass(frozen=True)
class TrainConfig:
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
    repo_root: Path
    checkpoint: Path
    episodes: int = 20
    max_actions: int = 128
    formatid: str = "gen9ou"
    player_id: str = "p1"
    simple_state: bool = True
    planning_top_k: int = 4
    planning_gamma: float = 0.99
    strategy: str = "planned"
    p1team: TeamSpec | None = None
    p2team: TeamSpec | None = None


class _LazyTorch:
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
    if legal_count <= 0:
        raise ValueError("No legal actions available")
    return rng.randrange(0, min(legal_count, max_actions))


def _first_action_index(legal_count: int, max_actions: int) -> int:
    if legal_count <= 0:
        raise ValueError("No legal actions available")
    return 0 if max_actions > 0 else 0


def _load_model_for_inference(checkpoint: dict[str, Any], device: str) -> tuple[Any, Any]:
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


def collect_offline_data(config: CollectConfig) -> dict[str, Any]:
    rng = random.Random(config.seed)
    builder = OfflineDatasetBuilder()
    player_id = _resolve_player_id(config.player_id)

    client = OUSelfPlayClient(repo_root=config.repo_root)
    try:
        total_steps = 0
        terminal_rewards: list[float] = []
        for _ in range(config.episodes):
            battle = client.create_battle(
                formatid=config.formatid,
                p1team=config.p1team,
                p2team=config.p2team,
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
                    curr_encoded = encode_legal_actions(curr_obs.legal_actions, config.max_actions)
                    next_step = env.step(action_index)
                    next_obs = next_step.observation
                    next_encoded = encode_legal_actions(next_obs.legal_actions, config.max_actions)

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
        }
    finally:
        client.close()


def train_offline(config: TrainConfig) -> dict[str, Any]:
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
            battle = client.create_battle(
                formatid=config.formatid,
                p1team=config.p1team,
                p2team=config.p2team,
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
                        encoded = encode_legal_actions(obs.legal_actions, config.max_actions)
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
    }


def _build_arg_parser() -> argparse.ArgumentParser:
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
    evaluate.add_argument("--planning-gamma", type=float, default=0.99)
    evaluate.add_argument("--strategy", choices=["planned", "greedy"], default="planned")
    evaluate.add_argument("--p1team", type=Path)
    evaluate.add_argument("--p2team", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
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
        else:
            config = EvalConfig(
                repo_root=args.repo_root,
                checkpoint=args.checkpoint,
                episodes=args.episodes,
                max_actions=args.max_actions,
                formatid=args.formatid,
                player_id=args.player_id,
                simple_state=not args.full_state,
                planning_top_k=args.planning_top_k,
                planning_gamma=args.planning_gamma,
                strategy=args.strategy,
                p1team=args.p1team,
                p2team=args.p2team,
            )
            summary = evaluate_agent(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())