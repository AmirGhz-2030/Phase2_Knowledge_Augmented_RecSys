"""
Knowledge-Augmented Neural Collaborative Filtering (KAR-NCF).

Combines:
1. Standard NeuMF (GMF branch + MLP branch) on User & Item IDs.
2. Knowledge Augmented Vectors (User Reasoning r^_p + Item Factual r^_i) via Hybrid-Expert Adaptor.
3. Feature interaction fusion layer for CTR prediction and Ranking.
"""

import torch
import torch.nn as nn
from src.models.adaptor import HybridExpertAdaptor


class KARNCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        knowledge_dim: int = 384,
        adaptor_out_dim: int = 32,
        mlp_hidden_dims: list[int] = None,
        dropout: float = 0.2,
        num_shared_experts: int = 3,
        num_dedicated_experts: int = 3,
        use_knowledge: bool = True
    ):
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64, 32]

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_knowledge = use_knowledge

        # 1. Classical ID Embeddings
        # GMF Branch
        self.gmf_user_emb = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(num_items, embedding_dim)

        # MLP Branch
        self.mlp_user_emb = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(num_items, embedding_dim)

        # 2. Hybrid-Expert Adaptor for Knowledge Vectors
        if self.use_knowledge:
            self.adaptor = HybridExpertAdaptor(
                input_dim=knowledge_dim,
                output_dim=adaptor_out_dim,
                num_shared_experts=num_shared_experts,
                num_dedicated_experts=num_dedicated_experts,
                hidden_dim=128,
                dropout=dropout
            )
            # Extra embedding size from augmented user & item knowledge vectors
            mlp_in_dim = (embedding_dim * 2) + (adaptor_out_dim * 2)
            gmf_in_dim = embedding_dim + adaptor_out_dim
        else:
            mlp_in_dim = embedding_dim * 2
            gmf_in_dim = embedding_dim

        # 3. MLP Layers
        layers = []
        in_dim = mlp_in_dim
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.mlp_layers = nn.Sequential(*layers)

        # 4. Final Prediction Layer (Fusion of GMF + MLP)
        final_in_dim = gmf_in_dim + mlp_hidden_dims[-1]
        self.prediction_layer = nn.Sequential(
            nn.Linear(final_in_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard NeuMF / KAR guidelines."""
        nn.init.normal_(self.gmf_user_emb.weight, std=0.01)
        nn.init.normal_(self.gmf_item_emb.weight, std=0.01)
        nn.init.normal_(self.mlp_user_emb.weight, std=0.01)
        nn.init.normal_(self.mlp_item_emb.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        for m in self.prediction_layer:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        user_knowledge_emb: torch.Tensor = None,
        item_knowledge_emb: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for KAR-NCF.
        user_indices: (batch_size,)
        item_indices: (batch_size,)
        user_knowledge_emb: (batch_size, knowledge_dim) or None
        item_knowledge_emb: (batch_size, knowledge_dim) or None
        """
        # GMF Branch ID Embeddings
        gmf_u = self.gmf_user_emb(user_indices)
        gmf_i = self.gmf_item_emb(item_indices)
        gmf_id_vector = gmf_u * gmf_i

        # MLP Branch ID Embeddings
        mlp_u = self.mlp_user_emb(user_indices)
        mlp_i = self.mlp_item_emb(item_indices)
        mlp_features = [mlp_u, mlp_i]

        if self.use_knowledge and user_knowledge_emb is not None and item_knowledge_emb is not None:
            # Transform knowledge vectors via MoE Adaptor
            aug_user_vec = self.adaptor.forward_user(user_knowledge_emb)
            aug_item_vec = self.adaptor.forward_item(item_knowledge_emb)
            
            # Interaction in GMF and concatenation in MLP
            gmf_know_vector = aug_user_vec * aug_item_vec
            gmf_vector = torch.cat([gmf_id_vector, gmf_know_vector], dim=-1)
            mlp_features.extend([aug_user_vec, aug_item_vec])
        else:
            gmf_vector = gmf_id_vector

        mlp_vector = torch.cat(mlp_features, dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)

        # Fusion
        fusion_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        prediction = self.prediction_layer(fusion_vector)
        return prediction.squeeze(-1)
