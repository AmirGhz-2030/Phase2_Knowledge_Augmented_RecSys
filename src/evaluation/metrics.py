"""
Evaluation Metrics for KAR Recommender System.

Includes:
- CTR Metrics: AUC, LogLoss (Binary Cross-Entropy)
- Top-K Ranking Metrics: Hit Rate (HR@K), NDCG@K, MRR@K
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def calculate_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate Area Under ROC Curve (AUC)."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def calculate_logloss(y_true: np.ndarray, y_score: np.ndarray, eps: float = 1e-7) -> float:
    """Calculate Binary Cross-Entropy (LogLoss)."""
    y_score = np.clip(y_score, eps, 1 - eps)
    try:
        return float(log_loss(y_true, y_score))
    except Exception:
        return 1.0


def calculate_hr_at_k(ranked_items: list[int], target_item: int, k: int = 10) -> float:
    """Hit Rate @ K."""
    return 1.0 if target_item in ranked_items[:k] else 0.0


def calculate_ndcg_at_k(ranked_items: list[int], target_item: int, k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    if target_item in ranked_items[:k]:
        rank = ranked_items[:k].index(target_item)
        return float(1.0 / np.log2(rank + 2))
    return 0.0


def calculate_mrr_at_k(ranked_items: list[int], target_item: int, k: int = 10) -> float:
    """Mean Reciprocal Rank @ K."""
    if target_item in ranked_items[:k]:
        rank = ranked_items[:k].index(target_item)
        return float(1.0 / (rank + 1))
    return 0.0
