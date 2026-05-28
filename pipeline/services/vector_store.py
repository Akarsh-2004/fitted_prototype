import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pipeline.config import settings

class VectorStore:
    def __init__(self):
        self.index_dir = settings.vector_index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "embeddings.json"
        
        # In-memory mapping of item_id -> embedding list
        self.embeddings: Dict[str, List[float]] = {}
        self.load()

    def add_item(self, item_id: str, embedding: List[float]) -> None:
        """Adds or updates the embedding for a wardrobe item."""
        self.embeddings[item_id] = embedding
        self.save()

    def delete_item(self, item_id: str) -> None:
        """Deletes the embedding for a wardrobe item."""
        if item_id in self.embeddings:
            del self.embeddings[item_id]
            self.save()

    def search(self, query_embedding: List[float], top_k: int = 8) -> List[Tuple[str, float]]:
        """
        Performs a cosine similarity search against all stored wardrobe items.
        Returns a list of tuples containing (item_id, similarity_score).
        """
        if not self.embeddings:
            return []
            
        # Convert query embedding to NumPy array
        query = np.array(query_embedding, dtype=np.float32)
        norm_query = np.linalg.norm(query)
        if norm_query == 0:
            return []
            
        results = []
        for item_id, emb in self.embeddings.items():
            candidate = np.array(emb, dtype=np.float32)
            norm_cand = np.linalg.norm(candidate)
            if norm_cand == 0:
                continue
                
            # Cosine similarity calculation
            similarity = float(np.dot(query, candidate) / (norm_query * norm_cand))
            results.append((item_id, similarity))
            
        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self) -> None:
        """Saves the current index map to disk in JSON format."""
        try:
            with open(self.index_path, "w") as f:
                json.dump(self.embeddings, f)
        except Exception as e:
            print(f"Error saving vector store embeddings index: {e}")

    def load(self) -> None:
        """Loads the index map from disk if it exists."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    self.embeddings = json.load(f)
            except Exception as e:
                print(f"Error loading vector store embeddings index: {e}. Starting fresh.")
                self.embeddings = {}

# Global vector store instance
vector_store = VectorStore()
