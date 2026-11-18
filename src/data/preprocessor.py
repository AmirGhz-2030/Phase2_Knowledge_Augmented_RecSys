"""
Data Preprocessor for MovieLens-1M (and MovieLens-20M) following KAR Paper Protocol.

Protocol from Xi et al. (Section 5.1.1):
1. Ratings of 4 and 5 are labeled as positive (1), and ratings < 4 as negative (0).
2. Data is split into 90% training and 10% testing.
3. User & item features extracted for downstream CTR (AUC/LogLoss) and Top-K ranking models.
"""

import os
import pandas as pd
import numpy as np


class ML1MPreprocessor:
    def __init__(self, raw_dir: str = None, processed_dir: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_dir = raw_dir or os.path.join(base_dir, "data", "raw", "ml-1m")
        self.processed_dir = processed_dir or os.path.join(base_dir, "data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load MovieLens-1M .dat files."""
        # movies.dat: MovieID::Title::Genres
        movies = pd.read_csv(
            os.path.join(self.raw_dir, "movies.dat"),
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1"
        )

        # users.dat: UserID::Gender::Age::Occupation::Zip-code
        users = pd.read_csv(
            os.path.join(self.raw_dir, "users.dat"),
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python",
            encoding="latin-1"
        )

        # ratings.dat: UserID::MovieID::Rating::Timestamp
        ratings = pd.read_csv(
            os.path.join(self.raw_dir, "ratings.dat"),
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
            encoding="latin-1"
        )

        return movies, users, ratings

    def process(self, train_ratio: float = 0.9, positive_threshold: float = 4.0) -> dict:
        """Process MovieLens-1M into train/test splits according to KAR protocol."""
        movies, users, ratings = self.load_raw_data()

        # Binary label for CTR prediction (KAR protocol: 4-5 is positive)
        ratings["label"] = (ratings["rating"] >= positive_threshold).astype(int)

        # Create continuous 0-indexed mappings for user and movie IDs
        unique_users = sorted(ratings["user_id"].unique())
        unique_movies = sorted(movies["movie_id"].unique())

        user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        movie2idx = {mid: idx for idx, mid in enumerate(unique_movies)}

        ratings["user_idx"] = ratings["user_id"].map(user2idx)
        ratings["movie_idx"] = ratings["movie_id"].map(movie2idx)
        movies["movie_idx"] = movies["movie_id"].map(movie2idx)
        users["user_idx"] = users["user_id"].map(user2idx)

        # Drop any unmatched records
        ratings = ratings.dropna(subset=["user_idx", "movie_idx"])
        ratings["user_idx"] = ratings["user_idx"].astype(int)
        ratings["movie_idx"] = ratings["movie_idx"].astype(int)

        # Merge item and user metadata into interaction dataframe
        full_df = ratings.merge(
            movies[["movie_idx", "title", "genres"]], on="movie_idx", how="left"
        ).merge(
            users[["user_idx", "gender", "age", "occupation"]], on="user_idx", how="left"
        )

        # Sort by timestamp for realistic evaluation / Leave-Time split
        full_df = full_df.sort_values(by=["user_idx", "timestamp"]).reset_index(drop=True)

        # 90/10 train-test split by interactions per user (or global 90/10)
        train_list = []
        test_list = []

        for uid, user_group in full_df.groupby("user_idx"):
            n = len(user_group)
            if n < 2:
                train_list.append(user_group)
                continue
            split_idx = int(np.ceil(n * train_ratio))
            train_list.append(user_group.iloc[:split_idx])
            test_list.append(user_group.iloc[split_idx:])

        train_df = pd.concat(train_list).reset_index(drop=True)
        test_df = pd.concat(test_list).reset_index(drop=True)

        # Save processed files
        train_df.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False)
        movies.to_csv(os.path.join(self.processed_dir, "movies_clean.csv"), index=False)
        users.to_csv(os.path.join(self.processed_dir, "users_clean.csv"), index=False)

        stats = {
            "num_users": len(user2idx),
            "num_movies": len(movie2idx),
            "num_interactions": len(full_df),
            "num_train": len(train_df),
            "num_test": len(test_df),
            "pos_ratio_train": float(train_df["label"].mean()),
            "pos_ratio_test": float(test_df["label"].mean()),
        }

        return stats


if __name__ == "__main__":
    preprocessor = ML1MPreprocessor()
    stats = preprocessor.process()
    print("MovieLens-1M Preprocessing Complete!")
    for k, v in stats.items():
        print(f"  {k}: {v}")
