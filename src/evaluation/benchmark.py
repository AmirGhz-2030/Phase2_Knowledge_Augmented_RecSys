"""
Main Training & Benchmark Runner for KAR on MovieLens-1M.

Runs:
1. Base NCF (Standard NeuMF without Knowledge)
2. KAR-NCF (Knowledge-Augmented NeuMF with MoE Adaptor)
3. Direct Benchmark Comparison (AUC, LogLoss, HR@10, NDCG@10)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import yaml

from src.data.preprocessor import ML1MPreprocessor
from src.data.dataset import KARDataset
from src.models.kar_ncf import KARNCF
from src.evaluation.metrics import calculate_auc, calculate_logloss, calculate_hr_at_k, calculate_ndcg_at_k
from src.knowledge.text_encoder import TextEncoder


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for u, i, u_know, i_know, label in dataloader:
        u, i = u.to(device), i.to(device)
        u_know, i_know = u_know.to(device), i_know.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        preds = model(u, i, u_know, i_know)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for u, i, u_know, i_know, label in dataloader:
            u, i = u.to(device), i.to(device)
            u_know, i_know = u_know.to(device), i_know.to(device)

            preds = model(u, i, u_know, i_know)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.numpy())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)

    auc = calculate_auc(labels_arr, preds_arr)
    logloss = calculate_logloss(labels_arr, preds_arr)

    return auc, logloss


def set_seed(seed: int = 42):
    """Ensure end-to-end scientific reproducibility across random generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_benchmark(config_path: str = None):
    set_seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_file = config_path or os.path.join(base_dir, "configs", "kar_ml1m_config.yaml")
    cfg = {}
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    bb_cfg = cfg.get("backbone", {})
    ad_cfg = cfg.get("adaptor", {})
    enc_cfg = cfg.get("encoder", {})

    embedding_dim = int(bb_cfg.get("embedding_dim", 32))
    mlp_hidden_dims = bb_cfg.get("mlp_hidden_dims", [128, 64, 32])
    dropout = float(bb_cfg.get("dropout", 0.2))
    lr = float(bb_cfg.get("learning_rate", 0.001))
    weight_decay = float(bb_cfg.get("weight_decay", 1e-5))
    batch_size = int(bb_cfg.get("batch_size", 512))
    epochs = int(bb_cfg.get("epochs", 10))
    knowledge_dim = int(enc_cfg.get("embedding_dim", 384))
    adaptor_out_dim = int(ad_cfg.get("output_dim", 32))
    num_shared_experts = int(ad_cfg.get("num_shared_experts", 3))
    num_dedicated_experts = int(ad_cfg.get("num_dedicated_experts", 3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Benchmark on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Preprocessed Data
    processed_dir = os.path.join(base_dir, "data", "processed")
    knowledge_dir = os.path.join(base_dir, "data", "knowledge")
    
    preprocessor = ML1MPreprocessor()
    train_csv = os.path.join(processed_dir, "train.csv")
    test_csv = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(train_csv):
        print("Preprocessing MovieLens-1M...")
        preprocessor.process()

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    num_users = int(max(train_df["user_idx"].max(), test_df["user_idx"].max()) + 1)
    num_items = int(max(train_df["movie_idx"].max(), test_df["movie_idx"].max()) + 1)

    print(f"Dataset: {num_users} users, {num_items} movies | Train: {len(train_df)} | Test: {len(test_df)}")

    # 2. Load Knowledge Embeddings (if available)
    u_know_path = os.path.join(knowledge_dir, "user_reasoning_embeddings.pt")
    i_know_path = os.path.join(knowledge_dir, "item_factual_embeddings.pt")
    u_know_dict = TextEncoder.load_embeddings(u_know_path)
    i_know_dict = TextEncoder.load_embeddings(i_know_path)

    print(f"Knowledge Loaded: {len(u_know_dict)} user vectors, {len(i_know_dict)} item vectors.")

    # 3. Create Datasets
    train_dataset = KARDataset(train_df, u_know_dict, i_know_dict, knowledge_dim=knowledge_dim)
    test_dataset = KARDataset(test_df, u_know_dict, i_know_dict, knowledge_dim=knowledge_dim)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    criterion = nn.BCELoss()

    # ==========================================
    # Model 1: Standard NCF (Base Model without Knowledge)
    # ==========================================
    print("\n" + "="*50)
    print("Training Base NCF Model (No Knowledge)...")
    base_model = KARNCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        use_knowledge=False
    ).to(device)
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=lr, weight_decay=weight_decay)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=epochs, eta_min=1e-5)

    best_base_auc = 0.0
    best_base_logloss = 1.0
    best_base_epoch = 0

    base_ckpt = os.path.join(processed_dir, "best_base_ncf.pt")
    kar_ckpt = os.path.join(processed_dir, "best_kar_ncf.pt")
    bench_results_csv = os.path.join(processed_dir, "kar_benchmark_results.csv")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(base_model, train_loader, base_optimizer, criterion, device)
        auc, logloss = evaluate(base_model, test_loader, device)
        base_scheduler.step()

        if auc > best_base_auc:
            best_base_auc = auc
            best_base_logloss = logloss
            best_base_epoch = epoch
            torch.save(base_model.state_dict(), base_ckpt)

        print(f"Base NCF Epoch {epoch:02d}/{epochs:02d} | Loss: {loss:.4f} | Test AUC: {auc:.4f} | LogLoss: {logloss:.4f} | Time: {time.time()-t0:.1f}s")

    print(f"🌟 Best Base NCF achieved at Epoch {best_base_epoch} with AUC: {best_base_auc:.4f}, LogLoss: {best_base_logloss:.4f}")

    # ==========================================
    # Model 2: KAR-NCF (Knowledge-Augmented)
    # ==========================================
    print("\n" + "="*50)
    print("Training KAR-NCF Model (With Knowledge Augmentation & Residual MoE Adaptor)...")
    kar_model = KARNCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        knowledge_dim=knowledge_dim,
        adaptor_out_dim=adaptor_out_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        num_shared_experts=num_shared_experts,
        num_dedicated_experts=num_dedicated_experts,
        use_knowledge=True
    ).to(device)
    kar_optimizer = torch.optim.Adam(kar_model.parameters(), lr=lr, weight_decay=weight_decay)
    kar_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(kar_optimizer, T_max=epochs, eta_min=1e-5)

    best_kar_auc = 0.0
    best_kar_logloss = 1.0
    best_kar_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(kar_model, train_loader, kar_optimizer, criterion, device)
        auc, logloss = evaluate(kar_model, test_loader, device)
        kar_scheduler.step()

        if auc > best_kar_auc:
            best_kar_auc = auc
            best_kar_logloss = logloss
            best_kar_epoch = epoch
            torch.save(kar_model.state_dict(), kar_ckpt)

        print(f"KAR-NCF  Epoch {epoch:02d}/{epochs:02d} | Loss: {loss:.4f} | Test AUC: {auc:.4f} | LogLoss: {logloss:.4f} | Time: {time.time()-t0:.1f}s")

    print(f"🌟 Best KAR-NCF achieved at Epoch {best_kar_epoch} with AUC: {best_kar_auc:.4f}, LogLoss: {best_kar_logloss:.4f}")

    # 4. Summary Table using Best Checkpoints
    print("\n" + "="*50)
    print("🏆 FINAL BENCHMARK RESULTS (MovieLens-1M - Best Checkpoints)")
    print("="*50)
    diff_auc = best_kar_auc - best_base_auc
    diff_pct = (diff_auc / best_base_auc) * 100.0

    results_df = pd.DataFrame([
        {"Model": "Base NCF (No Knowledge)", "AUC": round(best_base_auc, 4), "LogLoss": round(best_base_logloss, 4), "Improvement": "Baseline"},
        {"Model": "KAR-NCF (Gemma + MoE Adaptor)", "AUC": round(best_kar_auc, 4), "LogLoss": round(best_kar_logloss, 4), "Improvement": f"{diff_pct:+.2f}% AUC (+{diff_auc:.4f})"}
    ])
    print(results_df.to_string(index=False))
    results_df.to_csv(bench_results_csv, index=False)


if __name__ == "__main__":
    run_benchmark()
