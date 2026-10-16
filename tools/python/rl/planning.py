"""Planning helpers using branch simulation and a value model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .action_features import encode_legal_actions
    from .features import heuristic_position_score
except ImportError:  # pragma: no cover - allows script-style execution
    from tools.python.rl.action_features import encode_legal_actions
    from tools.python.rl.features import heuristic_position_score


@dataclass(frozen=True)
class PlannedAction:
    """PlannedAction helper type/container used by the RL self-play tooling."""
    action_index: int
    score: float


def _model_outputs(*, model: object, device: object, observation: object, max_actions: int):
    import torch

    legal_actions = tuple(observation.legal_actions)
    encoded = encode_legal_actions(legal_actions, max_actions=max_actions, request=observation.request)
    scalar = torch.from_numpy(observation.scalar_features).float().unsqueeze(0).to(device)
    party = torch.from_numpy(observation.party_features).float().unsqueeze(0).to(device)
    action_vectors = torch.from_numpy(encoded.vectors).float().unsqueeze(0).to(device)
    action_mask = torch.from_numpy(encoded.mask).bool().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, value = model(scalar, party, action_vectors, action_mask)
    return logits, value


def _terminal_reward(env: object) -> float:
    state = env.battle.state
    if not state.done:
        return 0.0
    if state.winner == env.player_id:
        return 1.0
    if state.winner in (None, "tie"):
        return 0.0
    return -1.0


def _leaf_score(*, env: object, model: object, device: object, max_actions: int, rollout_bonus: float) -> float:
    if env.battle.state.done:
        return _terminal_reward(env)
    observation = env.observe()
    _, value = _model_outputs(model=model, device=device, observation=observation, max_actions=max_actions)
    return float(value.item()) + rollout_bonus * heuristic_position_score(observation)


def _search_best_score(
    *,
    env: object,
    model: object,
    device: object,
    max_actions: int,
    top_k: int,
    depth: int,
    gamma: float,
    rollout_bonus: float,
) -> float:
    observation = env.observe()
    legal_actions = env.legal_actions()
    if not legal_actions:
        return _leaf_score(
            env=env,
            model=model,
            device=device,
            max_actions=max_actions,
            rollout_bonus=rollout_bonus,
        )

    logits, _ = _model_outputs(model=model, device=device, observation=observation, max_actions=max_actions)
    valid_count = min(len(legal_actions), max_actions)

    import torch

    branch_width = max(1, min(top_k, valid_count))
    top_indices = torch.topk(logits[0, :valid_count], k=branch_width).indices.tolist()

    best_score = float("-inf")
    for candidate in top_indices:
        branch = env.fork()
        try:
            step = branch.step(int(candidate))
            if step.done:
                candidate_score = step.reward
            elif depth <= 1:
                candidate_score = step.reward + gamma * _leaf_score(
                    env=branch,
                    model=model,
                    device=device,
                    max_actions=max_actions,
                    rollout_bonus=rollout_bonus,
                )
            else:
                candidate_score = step.reward + gamma * _search_best_score(
                    env=branch,
                    model=model,
                    device=device,
                    max_actions=max_actions,
                    top_k=top_k,
                    depth=depth - 1,
                    gamma=gamma,
                    rollout_bonus=rollout_bonus,
                )
            if candidate_score > best_score:
                best_score = candidate_score
        finally:
            branch.battle.close()
    return best_score


def pick_action_with_model(
    *,
    env: object,
    model: object,
    device: object,
    max_actions: int,
    top_k: int = 4,
    depth: int = 3,
    gamma: float = 0.99,
    rollout_bonus: float = 0.35,
) -> PlannedAction:
    # torch import is intentionally local so dataset collection can run without it.
    """pick_action_with_model utility used by the RL self-play workflow."""
    observation = env.observe()
    legal_actions = env.legal_actions()
    if not legal_actions:
        raise ValueError("No legal actions available")

    logits, _ = _model_outputs(model=model, device=device, observation=observation, max_actions=max_actions)

    valid_count = min(len(legal_actions), max_actions)
    top_k = max(1, min(top_k, valid_count))

    import torch

    top_indices = torch.topk(logits[0, :valid_count], k=top_k).indices.tolist()

    best_index = int(top_indices[0])
    best_score = float("-inf")
    for candidate in top_indices:
        branch = env.fork()
        try:
            step = branch.step(int(candidate))
            if step.done:
                score = step.reward
            elif depth <= 1:
                score = step.reward + gamma * _leaf_score(
                    env=branch,
                    model=model,
                    device=device,
                    max_actions=max_actions,
                    rollout_bonus=rollout_bonus,
                )
            else:
                score = step.reward + gamma * _search_best_score(
                    env=branch,
                    model=model,
                    device=device,
                    max_actions=max_actions,
                    top_k=top_k,
                    depth=depth - 1,
                    gamma=gamma,
                    rollout_bonus=rollout_bonus,
                )
        finally:
            branch.battle.close()
        if score > best_score:
            best_score = score
            best_index = int(candidate)
    return PlannedAction(action_index=best_index, score=best_score)
