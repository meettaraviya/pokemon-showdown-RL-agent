"""Permutation-aware policy and value models for batch RL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    scalar_dim: int
    party_feature_dim: int
    action_feature_dim: int
    hidden_dim: int = 128
    party_slots: int = 6
    attention_heads: int = 4
    attention_layers: int = 2


class DeepSetStateEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.party_proj = nn.Sequential(
            nn.Linear(config.party_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(config.scalar_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, scalar: Tensor, party: Tensor) -> Tensor:
        # party: [batch, slots, features]
        party_latent = self.party_proj(party).sum(dim=1)
        scalar_latent = self.scalar_proj(scalar)
        return self.fusion(torch.cat([scalar_latent, party_latent], dim=-1))


class SetTransformerStateEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.scalar_token = nn.Linear(config.scalar_dim, config.hidden_dim)
        self.party_token = nn.Linear(config.party_feature_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.attention_layers)
        self.out = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, scalar: Tensor, party: Tensor) -> Tensor:
        scalar_token = self.scalar_token(scalar).unsqueeze(1)
        party_tokens = self.party_token(party)
        tokens = torch.cat([scalar_token, party_tokens], dim=1)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.out(pooled)


class ActionScoringHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.action_proj = nn.Sequential(
            nn.Linear(config.action_feature_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, state_latent: Tensor, action_vectors: Tensor) -> Tensor:
        # state_latent: [batch, hidden]
        # action_vectors: [batch, actions, action_dim]
        action_latent = self.action_proj(action_vectors)
        repeated_state = state_latent.unsqueeze(1).expand(-1, action_latent.size(1), -1)
        logits = self.scorer(torch.cat([repeated_state, action_latent], dim=-1)).squeeze(-1)
        return logits


class OfflinePolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig, architecture: str = "deepset") -> None:
        super().__init__()
        if architecture == "deepset":
            self.encoder: nn.Module = DeepSetStateEncoder(config)
        elif architecture == "transformer":
            self.encoder = SetTransformerStateEncoder(config)
        else:
            raise ValueError(f"Unknown architecture '{architecture}'")
        self.action_head = ActionScoringHead(config)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, scalar: Tensor, party: Tensor, action_vectors: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encoder(scalar, party)
        logits = self.action_head(latent, action_vectors)
        masked_logits = logits.masked_fill(~action_mask, float("-inf"))
        value = self.value_head(latent).squeeze(-1)
        return masked_logits, value
