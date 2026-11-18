"""
Dataset and DataLoader for KAR Recommender System.

Handles:
- Loading train/test interactions.
- Loading offline cached user reasoning vectors (r_p) and item factual vectors (r_i).
- Fallback default vectors for unseen users/items during inference.
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class KARDataset(Dataset):
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        user_knowledge_dict: dict[int, np.ndarray] = None,
        item_knowledge_dict: dict[int, np.ndarray] = None,
        knowledge_dim: int = 384
    ):
        self.users = torch.tensor(interactions_df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(interactions_df["movie_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(interactions_df["label"].values, dtype=torch.float32)
        
        self.knowledge_dim = knowledge_dim
        self.user_knowledge_dict = user_knowledge_dict or {}
        self.item_knowledge_dict = item_knowledge_dict or {}
        
        # Default zero vector for cold start / fallback
        self.default_vector = np.zeros(knowledge_dim, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u_idx = int(self.users[idx].item())
        i_idx = int(self.items[idx].item())
        label = self.labels[idx]

        # Retrieve offline pre-computed vectors
        u_vec = self.user_knowledge_dict.get(u_idx, self.default_vector)
        i_vec = self.item_knowledge_dict.get(i_idx, self.default_vector)

        return (
            self.users[idx],
            self.items[idx],
            torch.tensor(u_vec, dtype=torch.float32),
            torch.tensor(i_vec, dtype=torch.float32),
            label
        )
