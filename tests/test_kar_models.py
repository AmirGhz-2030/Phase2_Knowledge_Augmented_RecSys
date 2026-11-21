"""
Integration and unit tests for KAR Adaptor, Base/KAR Models, Dataset, and Evaluation Metrics.
"""

import numpy as np
import pandas as pd
import torch
from src.models.adaptor import HybridExpertAdaptor
from src.models.kar_ncf import KARNCF
from src.data.dataset import KARDataset
from src.evaluation.metrics import (
    calculate_auc,
    calculate_logloss,
    calculate_hr_at_k,
    calculate_ndcg_at_k,
    calculate_mrr_at_k
)
from unittest.mock import patch, MagicMock
from src.knowledge.text_encoder import TextEncoder
from src.knowledge.production_gemma_pipeline import query_gemma_clean


def test_hybrid_expert_adaptor():
    batch_size = 8
    input_dim = 384
    output_dim = 32

    adaptor = HybridExpertAdaptor(
        input_dim=input_dim,
        output_dim=output_dim,
        num_shared_experts=3,
        num_dedicated_experts=3
    )

    dummy_user_know = torch.randn(batch_size, input_dim)
    dummy_item_know = torch.randn(batch_size, input_dim)

    aug_u = adaptor.forward_user(dummy_user_know)
    aug_i = adaptor.forward_item(dummy_item_know)

    assert aug_u.shape == (batch_size, output_dim), f"Expected {(batch_size, output_dim)}, got {aug_u.shape}"
    assert aug_i.shape == (batch_size, output_dim), f"Expected {(batch_size, output_dim)}, got {aug_i.shape}"


def test_kar_ncf_forward():
    batch_size = 16
    num_users = 100
    num_items = 200
    knowledge_dim = 384

    model = KARNCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=32,
        knowledge_dim=knowledge_dim,
        adaptor_out_dim=32,
        use_knowledge=True
    )

    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    user_know = torch.randn(batch_size, knowledge_dim)
    item_know = torch.randn(batch_size, knowledge_dim)

    output = model(user_ids, item_ids, user_know, item_know)

    assert output.shape == (batch_size,), f"Expected {(batch_size,)}, got {output.shape}"
    assert torch.all((output >= 0.0) & (output <= 1.0)), "Probabilities must be in [0, 1]"


def test_base_ncf_forward():
    batch_size = 16
    num_users = 100
    num_items = 200

    model = KARNCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=32,
        use_knowledge=False
    )

    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))

    output = model(user_ids, item_ids)

    assert output.shape == (batch_size,), f"Expected {(batch_size,)}, got {output.shape}"
    assert torch.all((output >= 0.0) & (output <= 1.0)), "Probabilities must be in [0, 1]"


def test_kar_dataset_and_fallback():
    df = pd.DataFrame({
        "user_idx": [0, 1, 2],
        "movie_idx": [10, 20, 30],
        "label": [1.0, 0.0, 1.0]
    })
    user_dict = {0: np.ones(384, dtype=np.float32)}
    item_dict = {10: np.full(384, 0.5, dtype=np.float32)}

    ds = KARDataset(df, user_knowledge_dict=user_dict, item_knowledge_dict=item_dict, knowledge_dim=384)
    assert len(ds) == 3

    # Sample with known vectors
    u0, i0, uk0, ik0, l0 = ds[0]
    assert u0.item() == 0
    assert i0.item() == 10
    assert torch.all(uk0 == 1.0)
    assert torch.all(ik0 == 0.5)
    assert l0.item() == 1.0

    # Sample with missing vectors (fallback to zero vector)
    u1, i1, uk1, ik1, l1 = ds[1]
    assert u1.item() == 1
    assert i1.item() == 20
    assert torch.all(uk1 == 0.0)
    assert torch.all(ik1 == 0.0)
    assert l1.item() == 0.0


def test_evaluation_metrics():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.85, 0.75, 0.2])

    auc = calculate_auc(y_true, y_pred)
    logloss = calculate_logloss(y_true, y_pred)

    assert 0.5 <= auc <= 1.0, f"AUC out of expected range: {auc}"
    assert logloss >= 0.0, f"LogLoss must be non-negative: {logloss}"

    # Ranking metrics
    ranked = [101, 102, 103, 104, 105]
    assert calculate_hr_at_k(ranked, 101, k=5) == 1.0
    assert calculate_hr_at_k(ranked, 103, k=2) == 0.0
    assert calculate_hr_at_k(ranked, 999, k=5) == 0.0

    # NDCG@K
    assert calculate_ndcg_at_k(ranked, 101, k=5) == 1.0
    assert calculate_ndcg_at_k(ranked, 999, k=5) == 0.0

    # MRR@K
    assert calculate_mrr_at_k(ranked, 102, k=5) == 0.5
    assert calculate_mrr_at_k(ranked, 999, k=5) == 0.0


def test_text_encoder_load_embeddings():
    empty_dict = TextEncoder.load_embeddings("non_existent_file.pt")
    assert empty_dict == {}


def test_query_gemma_clean_null_and_fallback_handling():
    # Case 1: content is null, reasoning_content provides fallback answer
    mock_resp_fallback = MagicMock()
    mock_resp_fallback.status_code = 200
    mock_resp_fallback.json.return_value = {
        "choices": [{
            "message": {
                "content": None,
                "reasoning_content": "Step 1 analysis.\n\nA tense sci-fi thriller exploring identity and fate."
            }
        }]
    }
    with patch("src.knowledge.production_gemma_pipeline.requests.post", return_value=mock_resp_fallback):
        assert query_gemma_clean("test prompt") == "A tense sci-fi thriller exploring identity and fate."

    # Case 2: both content and reasoning_content are null (must not raise AttributeError)
    mock_resp_empty = MagicMock()
    mock_resp_empty.status_code = 200
    mock_resp_empty.json.return_value = {
        "choices": [{"message": {"content": None, "reasoning_content": None}}]
    }
    with patch("src.knowledge.production_gemma_pipeline.requests.post", return_value=mock_resp_empty):
        assert query_gemma_clean("test prompt") == ""

    # Case 3: normal clean content
    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_resp_ok.json.return_value = {
        "choices": [{"message": {"content": "A classic drama exploring resilience and friendship."}}]
    }
    with patch("src.knowledge.production_gemma_pipeline.requests.post", return_value=mock_resp_ok):
        assert query_gemma_clean("test prompt") == "A classic drama exploring resilience and friendship."


if __name__ == "__main__":
    test_hybrid_expert_adaptor()
    test_kar_ncf_forward()
    test_base_ncf_forward()
    test_kar_dataset_and_fallback()
    test_evaluation_metrics()
    test_text_encoder_load_embeddings()
    test_query_gemma_clean_null_and_fallback_handling()
    print("All enhanced KAR unit and integration tests passed successfully!")
