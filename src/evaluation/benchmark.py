"""
Main Training & Benchmark Runner for KAR on MovieLens-1M.

Runs:
1. Base NCF (Standard NeuMF without Knowledge)
2. KAR-NCF (Knowledge-Augmented NeuMF with Residual Hybrid-Expert MoE Adaptor)
3. Direct Benchmark Comparison (AUC, LogLoss) with leak-free chronological Validation
   checkpoint selection (default val_ratio=0.1 -> ~81% Train / 9% Val / 10% untouched Test).
   Pass `--val-ratio 0.0` to replicate the legacy 90/10 split protocol.
"""

import argparse
import copy
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml

from src.data.preprocessor import ML1MPreprocessor
from src.models.kar_ncf import KARNCF
from src.evaluation.metrics import calculate_auc, calculate_logloss
from src.knowledge.text_encoder import TextEncoder


def set_seed(seed: int = 42):
    """Ensure end-to-end scientific reproducibility across random generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_train_val_chronological(
    train_df: pd.DataFrame, val_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a per-user chronologically ordered training DataFrame into train_sub and val_df.

    Preserves strict temporal ordering per user (last `val_ratio` fraction of each user's
    training interactions goes to validation; users with <2 interactions remain in train).
    If val_ratio <= 0, returns (train_df, empty DataFrame).
    """
    if val_ratio <= 0.0 or len(train_df) == 0:
        return train_df.reset_index(drop=True), train_df.iloc[0:0].copy()

    ordered = (
        train_df.sort_values(by=["user_idx", "timestamp"]).reset_index(drop=True)
        if "timestamp" in train_df.columns
        else train_df.reset_index(drop=True)
    )
    counts = ordered.groupby("user_idx")["user_idx"].transform("count")
    rank = ordered.groupby("user_idx").cumcount()
    split_pos = np.ceil(counts * (1.0 - val_ratio)).astype(int)
    is_train = (counts < 2) | (rank < split_pos)

    tr_df = ordered[is_train].reset_index(drop=True)
    va_df = ordered[~is_train].reset_index(drop=True)
    return tr_df, va_df


def build_knowledge_matrix(
    know_dict: dict[int, np.ndarray],
    num_entities: int,
    knowledge_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack a {entity_idx: vector} dict into a dense (num_entities, knowledge_dim) tensor."""
    mat = torch.zeros((num_entities, knowledge_dim), dtype=torch.float32)
    for k, vec in know_dict.items():
        idx = int(k)
        if 0 <= idx < num_entities:
            mat[idx] = torch.as_tensor(vec, dtype=torch.float32)
    return mat.to(device)


def _df_to_tensors(
    df: pd.DataFrame, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(df["user_idx"].values, dtype=torch.long, device=device),
        torch.tensor(df["movie_idx"].values, dtype=torch.long, device=device),
        torch.tensor(df["label"].values, dtype=torch.float32, device=device),
    )


def train_epoch(
    model: nn.Module,
    u_tensor: torch.Tensor,
    i_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    u_know_mat: torch.Tensor,
    i_know_mat: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    use_knowledge: bool,
) -> float:
    model.train()
    n = len(u_tensor)
    perm = torch.randperm(n, device=u_tensor.device)
    total_loss = 0.0
    num_batches = 0

    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        bu, bi, by = u_tensor[idx], i_tensor[idx], y_tensor[idx]

        optimizer.zero_grad()
        if use_knowledge:
            preds = model(bu, bi, u_know_mat[bu], i_know_mat[bi])
        else:
            preds = model(bu, bi)
        loss = criterion(preds, by)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    u_tensor: torch.Tensor,
    i_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    u_know_mat: torch.Tensor,
    i_know_mat: torch.Tensor,
    batch_size: int,
    use_knowledge: bool,
) -> tuple[float, float]:
    model.eval()
    all_preds = []
    n = len(u_tensor)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            bu = u_tensor[start : start + batch_size]
            bi = i_tensor[start : start + batch_size]
            if use_knowledge:
                preds = model(bu, bi, u_know_mat[bu], i_know_mat[bi])
            else:
                preds = model(bu, bi)
            all_preds.append(preds.cpu().numpy())

    preds_arr = np.concatenate(all_preds)
    labels_arr = y_tensor.cpu().numpy()

    auc = calculate_auc(labels_arr, preds_arr)
    logloss = calculate_logloss(labels_arr, preds_arr)
    return auc, logloss


def run_benchmark(
    config_path: str = None,
    val_ratio: float = None,
    epochs_override: int = None,
):
    set_seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_file = config_path or os.path.join(base_dir, "configs", "kar_ml1m_config.yaml")
    cfg = {}
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    ds_cfg = cfg.get("dataset", {})
    bb_cfg = cfg.get("backbone", {})
    ad_cfg = cfg.get("adaptor", {})
    enc_cfg = cfg.get("encoder", {})

    effective_val_ratio = float(
        ds_cfg.get("val_ratio", 0.1) if val_ratio is None else val_ratio
    )
    embedding_dim = int(bb_cfg.get("embedding_dim", 32))
    mlp_hidden_dims = bb_cfg.get("mlp_hidden_dims", [128, 64, 32])
    dropout = float(bb_cfg.get("dropout", 0.2))
    lr = float(bb_cfg.get("learning_rate", 0.001))
    weight_decay = float(bb_cfg.get("weight_decay", 1e-5))
    batch_size = int(bb_cfg.get("batch_size", 512))
    epochs = int(epochs_override if epochs_override is not None else bb_cfg.get("epochs", 10))
    knowledge_dim = int(enc_cfg.get("embedding_dim", 384))
    adaptor_out_dim = int(ad_cfg.get("output_dim", 32))
    num_shared_experts = int(ad_cfg.get("num_shared_experts", 3))
    num_dedicated_experts = int(ad_cfg.get("num_dedicated_experts", 3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running Benchmark on Device: {device} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )

    # 1. Load Preprocessed Data
    processed_dir = os.path.join(base_dir, "data", "processed")
    knowledge_dir = os.path.join(base_dir, "data", "knowledge")

    preprocessor = ML1MPreprocessor()
    train_csv = os.path.join(processed_dir, "train.csv")
    test_csv = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(train_csv):
        print("Preprocessing MovieLens-1M...")
        preprocessor.process()

    full_train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    num_users = int(max(full_train_df["user_idx"].max(), test_df["user_idx"].max()) + 1)
    num_items = int(max(full_train_df["movie_idx"].max(), test_df["movie_idx"].max()) + 1)

    train_df, val_df = split_train_val_chronological(full_train_df, effective_val_ratio)
    use_val = len(val_df) > 0
    if use_val:
        print(
            f"Protocol: Leak-Free Chronological Train/Val/Test "
            f"(Train={len(train_df):,}, Val={len(val_df):,}, Untouched Test={len(test_df):,})"
        )
    else:
        print(
            f"Protocol: Legacy 90/10 Chronological Split "
            f"(Train={len(train_df):,}, Test={len(test_df):,})"
        )

    # 2. Load Knowledge Embeddings into GPU/CPU lookup matrices
    u_know_path = os.path.join(knowledge_dir, "user_reasoning_embeddings.pt")
    i_know_path = os.path.join(knowledge_dir, "item_factual_embeddings.pt")
    u_know_dict = TextEncoder.load_embeddings(u_know_path)
    i_know_dict = TextEncoder.load_embeddings(i_know_path)
    print(f"Knowledge Loaded: {len(u_know_dict)} user vectors, {len(i_know_dict)} item vectors.")

    u_know_mat = build_knowledge_matrix(u_know_dict, num_users, knowledge_dim, device)
    i_know_mat = build_knowledge_matrix(i_know_dict, num_items, knowledge_dim, device)

    tr_u, tr_i, tr_y = _df_to_tensors(train_df, device)
    te_u, te_i, te_y = _df_to_tensors(test_df, device)
    if use_val:
        va_u, va_i, va_y = _df_to_tensors(val_df, device)
    else:
        va_u, va_i, va_y = te_u, te_i, te_y

    criterion = nn.BCELoss()
    eval_batch_size = max(batch_size * 4, 2048)

    base_ckpt = os.path.join(processed_dir, "best_base_ncf.pt")
    kar_ckpt = os.path.join(processed_dir, "best_kar_ncf.pt")
    bench_results_csv = os.path.join(processed_dir, "kar_benchmark_results.csv")

    def train_and_select(model_name: str, use_knowledge: bool, ckpt_path: str):
        set_seed(42)
        print("\n" + "=" * 60)
        print(f"Training {model_name}...")
        model = KARNCF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            knowledge_dim=knowledge_dim,
            adaptor_out_dim=adaptor_out_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            num_shared_experts=num_shared_experts,
            num_dedicated_experts=num_dedicated_experts,
            use_knowledge=use_knowledge,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_sel_auc = 0.0
        best_epoch = 0
        best_state = None
        sel_label = "Val" if use_val else "Test"

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = train_epoch(
                model, tr_u, tr_i, tr_y, u_know_mat, i_know_mat,
                batch_size, optimizer, criterion, use_knowledge
            )
            sel_auc, sel_ll = evaluate(
                model, va_u, va_i, va_y, u_know_mat, i_know_mat,
                eval_batch_size, use_knowledge
            )
            scheduler.step()

            if sel_auc > best_sel_auc:
                best_sel_auc = sel_auc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"{model_name:<8} Epoch {epoch:02d}/{epochs:02d} | Loss: {loss:.4f} | "
                f"{sel_label} AUC: {sel_auc:.4f} | {sel_label} LogLoss: {sel_ll:.4f} | "
                f"Time: {time.time() - t0:.1f}s",
                flush=True,
            )

        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)
        test_auc, test_ll = evaluate(
            model, te_u, te_i, te_y, u_know_mat, i_know_mat,
            eval_batch_size, use_knowledge
        )
        print(
            f"Best {model_name} selected at Epoch {best_epoch} "
            f"({sel_label} AUC: {best_sel_auc:.4f}) -> "
            f"Held-Out Test AUC: {test_auc:.4f}, Test LogLoss: {test_ll:.4f}"
        )
        return test_auc, test_ll, best_sel_auc, best_epoch

    base_test_auc, base_test_ll, _, _ = train_and_select(
        "Base NCF", use_knowledge=False, ckpt_path=base_ckpt
    )
    kar_test_auc, kar_test_ll, _, _ = train_and_select(
        "KAR-NCF", use_knowledge=True, ckpt_path=kar_ckpt
    )

    print("\n" + "=" * 60)
    print("FINAL BENCHMARK RESULTS (MovieLens-1M - Held-Out Test Set)")
    print("=" * 60)
    diff_auc = kar_test_auc - base_test_auc
    diff_pct = (diff_auc / base_test_auc) * 100.0

    results_df = pd.DataFrame(
        [
            {
                "Model": "Base NCF (No Knowledge)",
                "AUC": round(base_test_auc, 4),
                "LogLoss": round(base_test_ll, 4),
                "Improvement": "Baseline",
            },
            {
                "Model": "KAR-NCF (Gemma + MoE Adaptor)",
                "AUC": round(kar_test_auc, 4),
                "LogLoss": round(kar_test_ll, 4),
                "Improvement": f"{diff_pct:+.2f}% AUC ({diff_auc:+.4f})",
            },
        ]
    )
    print(results_df.to_string(index=False))
    results_df.to_csv(bench_results_csv, index=False)
    print(f"Saved benchmark results to {bench_results_csv}")


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark Base NCF vs KAR-NCF.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Chronological per-user validation fraction carved from train.csv (default: config value 0.1; 0.0 = legacy 90/10).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    args = parser.parse_args()
    run_benchmark(config_path=args.config, val_ratio=args.val_ratio, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
