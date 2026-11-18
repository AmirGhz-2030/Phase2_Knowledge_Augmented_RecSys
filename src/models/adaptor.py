"""
Hybrid-Expert Adaptor (MoE) Module for KAR Framework.

Based on Xi et al. (Section 4.3):
Transforms high-dimensional semantic knowledge vectors into compact recommendation-compatible
embeddings via shared and dedicated expert MLP networks with gating mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """A 2-layer MLP expert network as specified in the KAR paper."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridExpertAdaptor(nn.Module):
    """
    Hybrid-Expert Adaptor for transforming Reasoning & Factual Knowledge representations.
    
    Architecture:
    - Shared Experts: shared across both user and item domains.
    - Dedicated Experts: specialized for either user reasoning or item factual vectors.
    - Gating Network: computes dynamic soft attention weights over experts.
    """
    def __init__(
        self,
        input_dim: int = 384,  # e.g. sentence-transformer embedding dimension
        output_dim: int = 32,  # target recommendation embedding size
        num_shared_experts: int = 3,
        num_dedicated_experts: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_shared = num_shared_experts
        self.num_dedicated = num_dedicated_experts
        self.total_experts = num_shared_experts + num_dedicated_experts

        # Expert pools
        self.shared_experts = nn.ModuleList([
            ExpertMLP(input_dim, hidden_dim, output_dim, dropout) for _ in range(num_shared_experts)
        ])
        
        self.user_dedicated_experts = nn.ModuleList([
            ExpertMLP(input_dim, hidden_dim, output_dim, dropout) for _ in range(num_dedicated_experts)
        ])
        
        self.item_dedicated_experts = nn.ModuleList([
            ExpertMLP(input_dim, hidden_dim, output_dim, dropout) for _ in range(num_dedicated_experts)
        ])

        # Gating networks
        self.user_gate = nn.Linear(input_dim, self.total_experts)
        self.item_gate = nn.Linear(input_dim, self.total_experts)

        # Residual shortcut projection for gradient flow stability
        self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward_user(self, user_know_emb: torch.Tensor) -> torch.Tensor:
        """
        Transform user reasoning vector r_p into augmented vector r^_p (Eq. 2).
        Includes Residual skip-connection from input knowledge representation.
        """
        # Collect outputs from shared and user-dedicated experts
        expert_outputs = [expert(user_know_emb) for expert in self.shared_experts] + \
                         [expert(user_know_emb) for expert in self.user_dedicated_experts]
        
        # Stack into shape: (batch_size, total_experts, output_dim)
        expert_tensor = torch.stack(expert_outputs, dim=1)

        # Gate weights: (batch_size, total_experts, 1)
        gate_weights = F.softmax(self.user_gate(user_know_emb), dim=-1).unsqueeze(-1)

        # Weighted combination across experts + Residual Projection
        augmented_user_vec = torch.sum(gate_weights * expert_tensor, dim=1) + self.residual_proj(user_know_emb)
        return augmented_user_vec

    def forward_item(self, item_know_emb: torch.Tensor) -> torch.Tensor:
        """
        Transform item factual vector r_i into augmented vector r^_i.
        Includes Residual skip-connection from input knowledge representation.
        """
        # Collect outputs from shared and item-dedicated experts
        expert_outputs = [expert(item_know_emb) for expert in self.shared_experts] + \
                         [expert(item_know_emb) for expert in self.item_dedicated_experts]
        
        # Stack into shape: (batch_size, total_experts, output_dim)
        expert_tensor = torch.stack(expert_outputs, dim=1)

        # Gate weights: (batch_size, total_experts, 1)
        gate_weights = F.softmax(self.item_gate(item_know_emb), dim=-1).unsqueeze(-1)

        # Weighted combination across experts + Residual Projection
        augmented_item_vec = torch.sum(gate_weights * expert_tensor, dim=1) + self.residual_proj(item_know_emb)
        return augmented_item_vec
