"""
Direct Embedding Pipeline for MovieLens-1M Metadata & User History.

Generates:
1. Item Factual Embeddings from Movie Titles + Genres.
2. User Reasoning Embeddings from aggregated viewing history & genre distributions.
Saves PyTorch embedding dictionaries to data/knowledge/.
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.knowledge.text_encoder import TextEncoder


def build_item_knowledge_texts(movies_df: pd.DataFrame) -> dict[str, str]:
    """Create rich descriptive text for each movie."""
    item_texts = {}
    for _, row in movies_df.iterrows():
        m_idx = str(int(row["movie_idx"]))
        title = row["title"]
        genres = str(row["genres"]).replace("|", ", ")
        text = f"Movie: {title}. Genres: {genres}. Cinematic Context: Classic film featuring narrative elements of {genres}."
        item_texts[m_idx] = text
    return item_texts


def build_user_reasoning_texts(train_df: pd.DataFrame) -> dict[str, str]:
    """Create aggregated user taste reasoning text from liked movies."""
    user_texts = {}
    liked_df = train_df[train_df["label"] == 1]

    for u_idx, group in tqdm(liked_df.groupby("user_idx"), desc="Building User Profiles"):
        u_str = str(int(u_idx))
        # Top genres and recent movies
        recent_movies = group.tail(15)["title"].tolist()
        movie_list_str = "; ".join(recent_movies[:10])

        all_genres = []
        for g_str in group["genres"].dropna():
            all_genres.extend(str(g_str).split("|"))

        top_genres = pd.Series(all_genres).value_counts().head(3).index.tolist() if all_genres else ["General"]
        top_genres_str = ", ".join(top_genres)

        text = (
            f"User Taste Profile: Strongly prefers {top_genres_str} films. "
            f"Demonstrates consistent engagement with thematic narratives. "
            f"Recent favored titles include: {movie_list_str}."
        )
        user_texts[u_str] = text

    return user_texts


def generate_all_embeddings():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    knowledge_dir = os.path.join(base_dir, "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)

    movies_df = pd.read_csv(os.path.join(processed_dir, "movies_clean.csv"))
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))

    print("Generating Knowledge Texts for Items & Users...")
    item_texts = build_item_knowledge_texts(movies_df)
    user_texts = build_user_reasoning_texts(train_df)

    print(f"Generated {len(item_texts)} item descriptions and {len(user_texts)} user profile texts.")

    # Initialize Sentence-Transformers Encoder
    encoder = TextEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\nEncoding Item Embeddings...")
    item_embeddings = encoder.encode_dict(item_texts, batch_size=128)
    encoder.save_embeddings(item_embeddings, os.path.join(knowledge_dir, "item_factual_embeddings.pt"))

    print("\nEncoding User Reasoning Embeddings...")
    user_embeddings = encoder.encode_dict(user_texts, batch_size=128)
    encoder.save_embeddings(user_embeddings, os.path.join(knowledge_dir, "user_reasoning_embeddings.pt"))

    print("\n✅ All Knowledge Embeddings Generated and Cached Successfully!")


if __name__ == "__main__":
    generate_all_embeddings()
