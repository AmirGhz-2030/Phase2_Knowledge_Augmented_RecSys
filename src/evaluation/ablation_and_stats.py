"""
Reproducible Adversarial Ablation, Statistical Significance & Latency Verification for KAR-NCF.

Evaluates saved checkpoints (`best_base_ncf.pt`, `best_kar_ncf.pt`) on the held-out
MovieLens-1M test set (`data/processed/test.csv`) across:
1. Base NCF (ID-only baseline)
2. KAR-NCF (Full Hybrid Gemma + Sentence-Transformer Knowledge)
3. Adversarial Ablations on KAR-NCF:
   - Zeroed Knowledge Vectors (tests reliance on semantic vectors)
   - Matched-Variance Gaussian Noise (tests semantic signal vs. magnitude/norm)
   - Entity-Shuffled Knowledge Vectors (tests semantic user/item alignment)
4. Two-sided paired t-test (`scipy.stats.ttest_rel`) on per-sample BCE losses
   between Base NCF and KAR-NCF over the full held-out test set.
5. Real-time Top-K full-catalog inference latency across all 3,883 movies.

Saves verifiable results to `data/processed/ablation_stats_results.csv`.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from src.evaluation.benchmark import build_knowledge_matrix
from src.evaluation.metrics import calculate_auc, calculate_logloss
from src.knowledge.text_encoder import TextEncoder
from src.models.kar_ncf import KARNCF


def compute_per_sample_bce(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7
) -> np.ndarray:
    """Compute per-sample Binary Cross-Entropy loss vector."""
    p = np.clip(y_pred, eps, 1.0 - eps)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def compute_paired_bce_ttest(
    y_true: np.ndarray, base_preds: np.ndarray, kar_preds: np.ndarray
) -> dict:
    """Run a two-sided paired t-test on per-sample BCE losses (Base NCF vs. KAR-NCF)."""
    bce_base = compute_per_sample_bce(y_true, base_preds)
    bce_kar = compute_per_sample_bce(y_true, kar_preds)
    t_stat, p_val = stats.ttest_rel(bce_base, bce_kar)
    diff = bce_base - bce_kar  # positive means KAR-NCF has lower loss
    return {
        "n_samples": int(len(y_true)),
        "mean_bce_base": float(np.mean(bce_base)),
        "mean_bce_kar": float(np.mean(bce_kar)),
        "mean_bce_reduction": float(np.mean(diff)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
    }


def shuffle_knowledge_dict(
    know_dict: dict[int, np.ndarray], seed: int = 42
) -> dict[int, np.ndarray]:
    """Randomly permute knowledge vectors across entity IDs deterministically."""
    rng = np.random.default_rng(seed)
    keys = sorted(know_dict.keys())
    vals = [know_dict[k] for k in keys]
    perm = rng.permutation(len(vals))
    return {keys[i]: vals[perm[i]] for i in range(len(keys))}


def make_matched_noise_embeddings(
    know_dict: dict[int, np.ndarray], knowledge_dim: int = 384, seed: int = 42
) -> dict[int, np.ndarray]:
    """Generate Gaussian noise vectors matching the empirical per-coordinate std of embeddings."""
    rng = np.random.default_rng(seed)
    keys = sorted(know_dict.keys())
    if not keys:
        return {}
    emp_std = float(np.std(np.stack([know_dict[k] for k in keys])))
    return {
        k: rng.normal(loc=0.0, scale=emp_std, size=(knowledge_dim,)).astype(np.float32)
        for k in keys
    }


def predict_all(
    model: torch.nn.Module,
    u_tensor: torch.Tensor,
    i_tensor: torch.Tensor,
    u_know_mat: torch.Tensor,
    i_know_mat: torch.Tensor,
    use_knowledge: bool,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(u_tensor), batch_size):
            bu = u_tensor[start : start + batch_size]
            bi = i_tensor[start : start + batch_size]
            if use_knowledge:
                p = model(bu, bi, u_know_mat[bu], i_know_mat[bi])
            else:
                p = model(bu, bi)
            preds.append(p.cpu().numpy())
    return np.concatenate(preds)


def benchmark_catalog_latency(
    model: torch.nn.Module,
    u_know_mat: torch.Tensor,
    i_know_mat: torch.Tensor,
    num_items: int,
    device: torch.device,
    user_idx: int = 0,
    warmup: int = 5,
    runs: int = 50,
) -> dict:
    """Measure full-catalog single-user inference latency across all items."""
    model.eval()
    all_items = torch.arange(num_items, dtype=torch.long, device=device)
    user_ids = torch.full((num_items,), user_idx, dtype=torch.long, device=device)
    u_vecs = u_know_mat[user_idx].unsqueeze(0).expand(num_items, -1)
    i_vecs = i_know_mat[:num_items]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(user_ids, all_items, u_vecs, i_vecs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(user_ids, all_items, u_vecs, i_vecs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "num_items": int(num_items),
        "runs": int(runs),
        "mean_ms": float(np.mean(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "max_ms": float(np.max(times_ms)),
    }


def run_ablation_and_stats(config_path: str = None, seed: int = 42):
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
    knowledge_dim = int(enc_cfg.get("embedding_dim", 384))
    adaptor_out_dim = int(ad_cfg.get("output_dim", 32))
    num_shared_experts = int(ad_cfg.get("num_shared_experts", 3))
    num_dedicated_experts = int(ad_cfg.get("num_dedicated_experts", 3))

    processed_dir = os.path.join(base_dir, "data", "processed")
    knowledge_dir = os.path.join(base_dir, "data", "knowledge")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    num_users = int(max(train_df["user_idx"].max(), test_df["user_idx"].max()) + 1)
    num_items = int(max(train_df["movie_idx"].max(), test_df["movie_idx"].max()) + 1)

    u_know_dict = TextEncoder.load_embeddings(
        os.path.join(knowledge_dir, "user_reasoning_embeddings.pt")
    )
    i_know_dict = TextEncoder.load_embeddings(
        os.path.join(knowledge_dir, "item_factual_embeddings.pt")
    )

    u_mat = build_knowledge_matrix(u_know_dict, num_users, knowledge_dim, device)
    i_mat = build_knowledge_matrix(i_know_dict, num_items, knowledge_dim, device)

    te_u = torch.tensor(test_df["user_idx"].values, dtype=torch.long, device=device)
    te_i = torch.tensor(test_df["movie_idx"].values, dtype=torch.long, device=device)
    y_true = test_df["label"].values.astype(np.float64)

    base_ckpt = os.path.join(processed_dir, "best_base_ncf.pt")
    kar_ckpt = os.path.join(processed_dir, "best_kar_ncf.pt")
    if not (os.path.exists(base_ckpt) and os.path.exists(kar_ckpt)):
        raise FileNotFoundError(
            "Trained checkpoints not found. Run `python -m src.evaluation.benchmark` first."
        )

    base_model = KARNCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        use_knowledge=False,
    ).to(device)
    base_model.load_state_dict(torch.load(base_ckpt, map_location=device, weights_only=True))

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
        use_knowledge=True,
    ).to(device)
    kar_model.load_state_dict(torch.load(kar_ckpt, map_location=device, weights_only=True))

    # 1. Core comparison
    base_preds = predict_all(base_model, te_u, te_i, u_mat, i_mat, use_knowledge=False)
    kar_preds = predict_all(kar_model, te_u, te_i, u_mat, i_mat, use_knowledge=True)

    # 2. Adversarial Ablations
    zero_u = torch.zeros_like(u_mat)
    zero_i = torch.zeros_like(i_mat)
    zero_preds = predict_all(kar_model, te_u, te_i, zero_u, zero_i, use_knowledge=True)

    noise_u = build_knowledge_matrix(
        make_matched_noise_embeddings(u_know_dict, knowledge_dim, seed=seed),
        num_users, knowledge_dim, device,
    )
    noise_i = build_knowledge_matrix(
        make_matched_noise_embeddings(i_know_dict, knowledge_dim, seed=seed + 1),
        num_items, knowledge_dim, device,
    )
    noise_preds = predict_all(kar_model, te_u, te_i, noise_u, noise_i, use_knowledge=True)

    shuf_u = build_knowledge_matrix(
        shuffle_knowledge_dict(u_know_dict, seed=seed), num_users, knowledge_dim, device
    )
    shuf_i = build_knowledge_matrix(
        shuffle_knowledge_dict(i_know_dict, seed=seed + 1), num_items, knowledge_dim, device
    )
    shuf_preds = predict_all(kar_model, te_u, te_i, shuf_u, shuf_i, use_knowledge=True)

    # 3. Statistical Significance (Paired t-test on per-sample BCE)
    ttest_res = compute_paired_bce_ttest(y_true, base_preds, kar_preds)

    # 4. Full-Catalog Real-Time Latency Benchmark
    latency_res = benchmark_catalog_latency(kar_model, u_mat, i_mat, num_items, device)

    base_auc = calculate_auc(y_true, base_preds)
    rows = []
    for label_name, preds in [
        ("Base NCF (No Knowledge)", base_preds),
        ("KAR-NCF (Full Semantic Knowledge)", kar_preds),
        ("Ablation: Zeroed Knowledge", zero_preds),
        ("Ablation: Matched-Variance Gaussian Noise", noise_preds),
        ("Ablation: Shuffled Knowledge (Entity Permuted)", shuf_preds),
    ]:
        auc = calculate_auc(y_true, preds)
        ll = calculate_logloss(y_true, preds)
        rows.append(
            {
                "Condition": label_name,
                "Test_AUC": round(auc, 4),
                "Test_LogLoss": round(ll, 4),
                "Delta_AUC_vs_Base": round(auc - base_auc, 4),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(processed_dir, "ablation_stats_results.csv")
    out_df.to_csv(out_csv, index=False)

    print("=" * 72)
    print(f"ABLATION & STATISTICAL SIGNIFICANCE REPORT (Held-Out Test N={ttest_res['n_samples']:,})")
    print("=" * 72)
    print(out_df.to_string(index=False))
    print("-" * 72)
    print(
        f"Paired t-test (Per-Sample BCE: Base NCF vs. KAR-NCF): "
        f"t = {ttest_res['t_statistic']:.2f}, p = {ttest_res['p_value']:.2e} "
        f"(mean BCE reduction = {ttest_res['mean_bce_reduction']:+.6f})"
    )
    print(
        f"Full-Catalog Inference Latency ({latency_res['num_items']:,} items on {latency_res['device']}, "
        f"{latency_res['runs']} runs): mean = {latency_res['mean_ms']:.2f} ms, "
        f"p95 = {latency_res['p95_ms']:.2f} ms, max = {latency_res['max_ms']:.2f} ms"
    )
    print(f"Saved ablation table to {out_csv}")
    return out_df, ttest_res, latency_res


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial knowledge ablations, paired t-test, and latency benchmark."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise/shuffle ablations.")
    args = parser.parse_args()
    run_ablation_and_stats(config_path=args.config, seed=args.seed)


if __name__ == "__main__":
    main()
