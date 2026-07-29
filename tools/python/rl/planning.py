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
    action_index: int
    score: float


def pick_action_with_model(
    *,
    env: object,
    model: object,
    device: object,
    max_actions: int,
    top_k: int = 4,
    gamma: float = 0.99,
    rollout_bonus: float = 0.35,
) -> PlannedAction:
    # torch import is intentionally local so dataset collection can run without it.
    import torch

    observation = env.observe()
    legal_actions = env.legal_actions()
    if not legal_actions:
        raise ValueError("No legal actions available")

    encoded = encode_legal_actions(legal_actions, max_actions=max_actions)
    scalar = torch.from_numpy(observation.scalar_features).float().unsqueeze(0).to(device)
    party = torch.from_numpy(observation.party_features).float().unsqueeze(0).to(device)
    action_vectors = torch.from_numpy(encoded.vectors).float().unsqueeze(0).to(device)
    action_mask = torch.from_numpy(encoded.mask).bool().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(scalar, party, action_vectors, action_mask)

    valid_count = min(len(legal_actions), max_actions)
    top_k = max(1, min(top_k, valid_count))
    top_indices = torch.topk(logits[0, :valid_count], k=top_k).indices.tolist()

    best_index = int(top_indices[0])
    best_score = float("-inf")
    for candidate in top_indices:
        simulated = env.simulate(candidate)
        sim_actions = tuple(simulated.legal_actions)
        sim_encoded = encode_legal_actions(sim_actions, max_actions=max_actions)
        sim_scalar = torch.from_numpy(simulated.scalar_features).float().unsqueeze(0).to(device)
        sim_party = torch.from_numpy(simulated.party_features).float().unsqueeze(0).to(device)
        sim_action_vectors = torch.from_numpy(sim_encoded.vectors).float().unsqueeze(0).to(device)
        sim_action_mask = torch.from_numpy(sim_encoded.mask).bool().unsqueeze(0).to(device)
        with torch.no_grad():
            _, next_value = model(sim_scalar, sim_party, sim_action_vectors, sim_action_mask)
        score = float(next_value.item()) * gamma + rollout_bonus * heuristic_position_score(simulated)
        if score > best_score:
            best_score = score
            best_index = int(candidate)
    return PlannedAction(action_index=best_index, score=best_score)
