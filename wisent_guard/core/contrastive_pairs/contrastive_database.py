"""
This module implements a vector database for storing and retrieving
synthetically generated contrastive pairs using FAISS and a JSON metadata store.
"""

import json
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from ..response import NegativeResponse, PositiveResponse
from .contrastive_pair import ContrastivePair
from .contrastive_pair_set import ContrastivePairSet


class ContrastivePairDatabase:
    """
    Manages a database of contrastive pairs, using FAISS for semantic search
    and JSON for metadata storage.
    """

    def __init__(self, db_path: str, embedding_dim: int = 384) -> None:
        """
        Initializes the database.

        Args:
            db_path: The directory to store the database files (index and metadata).
            embedding_dim: The dimension of the embeddings (e.g., 384 for all-MiniLM-L6-v2).
        """
        self.db_path: Path = Path(db_path)
        self.index_path: Path = self.db_path / "traits.index"
        self.metadata_path: Path = self.db_path / "metadata.json"
        self.pairs_dir: Path = self.db_path / "pair_sets"

        self.embedding_dim: int = embedding_dim
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Loads the database from disk or creates it if it doesn't exist."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.pairs_dir.mkdir(exist_ok=True)

        if self.index_path.exists() and self.metadata_path.exists():
            print(f"📂 Loading existing database from {self.db_path}")
            self.index: faiss.Index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata: dict = json.load(f)
        else:
            print(f"✨ Creating new database at {self.db_path}")
            # Use IndexFlatL2 for simple, exact search.
            # For very large datasets, a more complex index like IndexIVFFlat would be better.
            self.index: faiss.Index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata: dict[str, Any] = {"next_set_id": 0, "sets": {}}
            self._save()

    def _save(self):
        """Saves the FAISS index and metadata to disk."""
        print(f"💾 Saving database to {self.db_path}...")
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def add_set(
        self, pair_set: ContrastivePairSet, trait_embedding: np.ndarray
    ) -> None:
        """
        Adds a new contrastive pair set to the database.

        Args:
            pair_set: The ContrastivePairSet to add.
            trait_embedding: The embedding of the trait description for this set.
        """
        # --- Input Validation and Normalization ---
        # Ensure trait_embedding is a numpy array
        if not isinstance(trait_embedding, np.ndarray):
            raise TypeError(
                f"trait_embedding must be a numpy array, but got {type(trait_embedding)}"
            )

        # If the embedding is wrapped in a list or extra dimension (e.g., shape (1, 384)), squeeze it.
        if trait_embedding.ndim > 1:
            trait_embedding: np.ndarray = np.squeeze(trait_embedding)

        # Final check for correct shape
        if trait_embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Trait embedding must be a 1D array of shape ({self.embedding_dim},), but got {trait_embedding.shape}"
            )

        set_id: int = self.metadata["next_set_id"]

        # 1. Add the trait embedding to the FAISS index
        self.index.add(np.array([trait_embedding], dtype=np.float32))

        # 2. Save the actual pair set to its own JSON file
        pair_set_path: Path = self.pairs_dir / f"{set_id}.json"

        data_to_save: dict[str, Any] = {
            "name": pair_set.name,
            "task_type": pair_set.task_type,
            "pairs": [
                {
                    "prompt": p.prompt,
                    "positive_response": p.positive_response.text,
                    "negative_response": p.negative_response.text,
                }
                for p in pair_set.pairs
            ],
        }
        with open(pair_set_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)

        # 3. Update and save metadata
        self.metadata["sets"][str(self.index.ntotal - 1)] = {
            "set_id": set_id,
            "name": pair_set.name,
            "trait_description": (
                pair_set.pairs[0].trait_description if pair_set.pairs else "N/A"
            ),
            "pair_count": len(pair_set.pairs),
        }
        self.metadata["next_set_id"] += 1
        self._save()
        print(f"✅ Added new pair set with ID {set_id} to the database.")

    def search_for_trait(
        self, trait_embedding: np.ndarray, threshold: float
    ) -> Optional[int]:
        """
        Searches for a semantically similar trait in the database.

        Args:
            trait_embedding: The embedding of the trait to search for.
            threshold: The similarity threshold (e.g., 0.95). A higher value means a stricter match.

        Returns:
            The ID of the found pair set, or None if no similar set is found.
        """
        if self.index.ntotal == 0:
            return None

        # --- Input Validation and Normalization ---
        # Ensure trait_embedding is a numpy array
        if not isinstance(trait_embedding, np.ndarray):
            raise TypeError(
                f"trait_embedding must be a numpy array, but got {type(trait_embedding)}"
            )

        # If the embedding is wrapped in a list or extra dimension (e.g., shape (1, 384)), squeeze it.
        if trait_embedding.ndim > 1:
            trait_embedding: np.ndarray = np.squeeze(trait_embedding)

        # Final check for correct shape
        if trait_embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Trait embedding must be a 1D array of shape ({self.embedding_dim},), but got {trait_embedding.shape}"
            )

        # FAISS search returns distances (L2 squared) and indices (labels)
        # k=1 means we only search for the single best match
        distances: np.ndarray
        labels: np.ndarray
        # Perform the search
        print(f"🔍 Searching for similar traits with threshold {threshold}...")
        distances, labels = self.index.search(
            np.array([trait_embedding], dtype=np.float32), k=1
        )

        best_match_label: int = labels[0][0]
        best_match_distance: float = distances[0][0]

        # Convert L2 distance to cosine similarity for a more intuitive threshold
        # This is an approximation but works well for normalized embeddings.
        similarity: float = 1 - (best_match_distance / 2)

        print(
            f"🔍 DB Search: Best match has similarity {similarity:.4f} (Label: {best_match_label})"
        )

        if similarity >= threshold:
            found_set_info: dict[str, Any] = self.metadata["sets"][
                str(best_match_label)
            ]
            print(
                f"🎉 Found similar trait in cache! (Set ID: {found_set_info['set_id']})"
            )
            return found_set_info

        return None

    def get_set_by_id(self, set_id: int) -> ContrastivePairSet:
        """
        Retrieves a contrastive pair set from the database by its ID.
        Args:
            set_id: The ID of the pair set to retrieve.
        Returns:
            The ContrastivePairSet object corresponding to the given ID.
        """
        pair_set_path: Path = self.pairs_dir / f"{set_id}.json"
        with open(pair_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pair_set: ContrastivePairSet = ContrastivePairSet(
            name=data.get("name"), task_type=data.get("task_type")
        )

        for pair_data in data["pairs"]:
            pair = ContrastivePair(
                prompt=pair_data["prompt"],
                positive_response=PositiveResponse(text=pair_data["positive_response"]),
                negative_response=NegativeResponse(text=pair_data["negative_response"]),
            )
            pair_set.pairs.append(pair)

        print(f"📦 Loaded pair set ID {set_id} from cache.")
        return pair_set
