"""
Vector index management using HNSW.
"""

import os
import json
import logging

import hnswlib
import numpy as np

from financial_chart_analyzer.config import config

logger = logging.getLogger(__name__)


class VectorIndex:
    """HNSW vector index for fast similarity search."""

    def __init__(self, dim=None, ef_construction=None, M=None, ef=None):
        """
        Initialize vector index.

        Args:
            dim: Embedding dimension (default from config)
            ef_construction: HNSW ef_construction parameter
            M: HNSW M parameter
            ef: HNSW ef parameter for query
        """
        self.dim = dim or config.retrieval.embedding_dim
        self.ef_construction = ef_construction or config.retrieval.ef_construction
        self.M = M or config.retrieval.ef_construction // 12  # Default M based on ef_construction
        self.ef = ef or config.retrieval.ef

        self.index = None
        self.num_elements = 0

    def create_index(self, num_elements):
        """Create a new index."""
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(max_elements=num_elements,
                           ef_construction=self.ef_construction,
                           M=self.M)
        self.index.set_ef(self.ef)
        self.num_elements = num_elements
        logger.info(f"Created index with {num_elements} elements")

    def add_items(self, embeddings, labels=None):
        """
        Add embeddings to index.

        Args:
            embeddings: Numpy array of shape (n, dim)
            labels: Optional label array of shape (n,)
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")

        if labels is None:
            labels = np.arange(len(embeddings))

        self.index.add_items(embeddings.astype(np.float32), labels)
        logger.info(f"Added {len(embeddings)} items to index")

    def save_index(self, path):
        """Save index to file."""
        if self.index is None:
            raise ValueError("No index to save")

        self.index.save_index(path)
        logger.info(f"Saved index to {path}")

    def load_index(self, path):
        """Load index from file."""
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.load_index(path)
        self.index.set_ef(self.ef)
        logger.info(f"Loaded index from {path}")

    def knn_query(self, query_vector, k=3):
        """
        Find k nearest neighbors.

        Args:
            query_vector: Query embedding vector
            k: Number of neighbors to return

        Returns:
            Tuple of (labels, distances)
        """
        if self.index is None:
            raise ValueError("Index not loaded")

        return self.index.knn_query(query_vector.astype(np.float32), k=k)
