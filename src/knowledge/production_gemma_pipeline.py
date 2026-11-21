"""
Production-Grade Gemma 4 E2B Knowledge Extractor.
Uses 600 max_tokens to accommodate reasoning tokens + clean final content.
Directly extracts message.content.
Runs with 4 parallel threads.
"""

import os
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def query_gemma_clean(user_prompt: str, max_tokens: int = 600) -> str:
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "google/gemma-4-e2b",
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(url, json=payload, timeout=90)
        if resp.status_code == 200:
            res_json = resp.json()
            msg = res_json["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            # If content exists, it is the clean final answer
            if content and len(content) > 20:
                return content
            # Fallback if content was empty or null
            reasoning = msg.get("reasoning_content") or ""
            return reasoning.split("\n\n")[-1].strip() if reasoning else ""
        return ""
    except Exception:
        return ""


def run_full_clean_pipeline(num_movies: int = 100, num_users: int = 349):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "processed")
    knowledge_dir = os.path.join(base_dir, "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)

    # 1. Process Movie Knowledge
    movies_df = pd.read_csv(os.path.join(data_dir, "movies_clean.csv"))
    item_json_path = os.path.join(knowledge_dir, "gemma_item_knowledge.json")
    
    existing_items = {}
    if os.path.exists(item_json_path):
        with open(item_json_path, "r", encoding="utf-8") as f:
            existing_items = json.load(f)

    movies_subset = movies_df.head(num_movies)
    print(f"🎬 Generating {len(movies_subset)} Clean Movie Factual Descriptions (4 Threads, 600 Tokens)...")

    def process_movie(row):
        prompt = (
            f"Provide a concise 2-sentence cinematic encyclopedia summary of the movie '{row['title']}' "
            f"(Genres: {row['genres']}). Focus on its core thematic premise, cinematic tone, and emotional atmosphere."
        )
        ans = query_gemma_clean(prompt)
        return str(row["movie_idx"]), ans

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_movie, row): row for _, row in movies_subset.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Clean Movies"):
            m_idx, text = future.result()
            if text and len(text) > 25:
                existing_items[m_idx] = text

    with open(item_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_items, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(existing_items)} clean movies to {item_json_path}")

    # 2. Process User Preference Reasoning
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    user_json_path = os.path.join(knowledge_dir, "gemma_user_reasoning.json")
    
    existing_users = {}
    if os.path.exists(user_json_path):
        with open(user_json_path, "r", encoding="utf-8") as f:
            existing_users = json.load(f)

    max_uid = int(train_df["user_idx"].max()) + 1
    target_uids = list(range(min(num_users, max_uid)))
    print(f"👤 Generating {len(target_uids)} Clean User Preference Profiles (4 Threads, 600 Tokens)...")

    def process_user(u_idx):
        u_history = train_df[(train_df["user_idx"] == u_idx) & (train_df["label"] == 1)].tail(8)
        if len(u_history) == 0:
            return str(u_idx), ""
        liked_movies = ", ".join([f"{row['title']} ({row['genres']})" for _, row in u_history.iterrows()])
        prompt = (
            f"A user highly rates the following movies: {liked_movies}. "
            f"Write a concise 2-3 sentence profile describing this user's underlying movie taste, favored genres, "
            f"and the psychological tone or narrative themes they enjoy."
        )
        ans = query_gemma_clean(prompt)
        return str(u_idx), ans

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_user, uid): uid for uid in target_uids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Clean Users"):
            u_idx, text = future.result()
            if text and len(text) > 25:
                existing_users[u_idx] = text

    with open(user_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_users, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(existing_users)} clean user profiles to {user_json_path}")


if __name__ == "__main__":
    run_full_clean_pipeline()
