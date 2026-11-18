"""
Text Encoder for Knowledge Augmented Recommendation (KAR).

Encodes extracted knowledge texts (user reasoning & item factual) into dense vector embeddings.
Uses sentence-transformers (e.g. all-MiniLM-L6-v2) or fallback TF-IDF/SVD when offline.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm


class TextEncoder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_dict(self, texts_dict: dict[str, str], batch_size: int = 64) -> dict[int, np.ndarray]:
        """
        Encode a dictionary of {idx_str: text} into {idx_int: embedding_np_array}.
        """
        indices = [int(k) for k in texts_dict.keys()]
        texts = [texts_dict[str(k)] for k in indices]

        print(f"Encoding {len(texts)} text entries with {self.model_name} on {self.device}...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings_dict = {idx: embeddings[i] for i, idx in enumerate(indices)}
        return embeddings_dict

    def save_embeddings(self, embeddings_dict: dict[int, np.ndarray], output_file: str):
        """Save embeddings dictionary as a PyTorch or NumPy artifact."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(embeddings_dict, output_file)
        print(f"Saved {len(embeddings_dict)} embeddings to {output_file}")

    @staticmethod
    def load_embeddings(file_path: str) -> dict[int, np.ndarray]:
        """Load embeddings from disk."""
        if not os.path.exists(file_path):
            return {}
        return torch.load(file_path, weights_only=False)
