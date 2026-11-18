"""
Build Embeddings directly from Gemma 4 E2B generated texts and fallback profiles.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.knowledge.text_encoder import TextEncoder
from src.knowledge.build_embeddings import build_item_knowledge_texts, build_user_reasoning_texts


def build_hybrid_gemma_embeddings():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    knowledge_dir = os.path.join(base_dir, "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)

    movies_df = pd.read_csv(os.path.join(processed_dir, "movies_clean.csv"))
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))

    # Load Gemma generated files
    gemma_items_path = os.path.join(knowledge_dir, "gemma_item_knowledge.json")
    gemma_users_path = os.path.join(knowledge_dir, "gemma_user_reasoning.json")

    gemma_items = {}
    if os.path.exists(gemma_items_path):
        with open(gemma_items_path, "r", encoding="utf-8") as f:
            gemma_items = json.load(f)

    gemma_users = {}
    if os.path.exists(gemma_users_path):
        with open(gemma_users_path, "r", encoding="utf-8") as f:
            gemma_users = json.load(f)

    print(f"Loaded {len(gemma_items)} Gemma film knowledge entries and {len(gemma_users)} Gemma user reasoning entries.")

    # Build base texts
    base_item_texts = build_item_knowledge_texts(movies_df)
    base_user_texts = build_user_reasoning_texts(train_df)

    # Blend: Prioritize Gemma 4 E2B texts where available
    final_item_texts = {}
    for m_idx, text in base_item_texts.items():
        if m_idx in gemma_items and len(gemma_items[m_idx]) > 20:
            final_item_texts[m_idx] = f"Gemma Analysis: {gemma_items[m_idx]} | Context: {text}"
        else:
            final_item_texts[m_idx] = text

    final_user_texts = {}
    for u_idx, text in base_user_texts.items():
        if u_idx in gemma_users and len(gemma_users[u_idx]) > 20:
            final_user_texts[u_idx] = f"Gemma Taste Reasoning: {gemma_users[u_idx]} | Profile: {text}"
        else:
            final_user_texts[u_idx] = text

    # Encode with Sentence-Transformers on GPU
    encoder = TextEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\nEncoding Final Blended Item Embeddings...")
    item_embs = encoder.encode_dict(final_item_texts, batch_size=128)
    encoder.save_embeddings(item_embs, os.path.join(knowledge_dir, "item_factual_embeddings.pt"))

    print("\nEncoding Final Blended User Reasoning Embeddings...")
    user_embs = encoder.encode_dict(final_user_texts, batch_size=128)
    encoder.save_embeddings(user_embs, os.path.join(knowledge_dir, "user_reasoning_embeddings.pt"))

    print("\n✅ Knowledge Embeddings successfully updated with real Gemma 4 E2B reasoning!")


if __name__ == "__main__":
    build_hybrid_gemma_embeddings()
