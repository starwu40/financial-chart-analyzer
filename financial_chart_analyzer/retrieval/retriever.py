"""
Chart retriever for vector similarity search.
"""

import os
import json
import logging

import numpy as np

from financial_chart_analyzer.retrieval.embeddings import JinaEmbeddings
from financial_chart_analyzer.retrieval.index import VectorIndex
from financial_chart_analyzer.config import config

logger = logging.getLogger(__name__)


class ChartRetriever:
    """Retrieve charts using vector similarity search."""

    def __init__(self, metadata_path, index_path="chart_index.bin", alpha=None):
        """
        Initialize chart retriever.

        Args:
            metadata_path: Path to charts_index.json
            index_path: Path to vector index file
            alpha: Weight for combining features (default from config)
        """
        self.metadata_path = metadata_path
        self.index_path = index_path
        self.alpha = alpha if alpha is not None else config.retrieval.alpha

        # Initialize components
        self.embeddings = JinaEmbeddings()
        self.index = VectorIndex()

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Initialize or load index
        if os.path.exists(index_path):
            logger.info(f"Loading existing index: {index_path}")
            self._load_index()
        else:
            logger.info("Building index...")
            self._build_index()
            self._save_index()

    def _build_index(self):
        """Build vector index from metadata."""
        dim = config.retrieval.embedding_dim
        num_elements = len(self.metadata)

        self.index.create_index(num_elements)

        embeddings = []
        for i, chart in enumerate(self.metadata):
            img_path = chart["path"]
            try:
                ocr_text = chart.get("ocr_text", "")
                numbers = chart.get("numbers", [])

                # Build enhanced text description
                enhanced_text = ocr_text
                if numbers:
                    enhanced_text += " key_data: " + " ".join(str(n) for n in numbers[:15])

                feat = self.embeddings.encode_text(enhanced_text, task="retrieval.passage")
                feat = feat.flatten()

                if feat.shape[0] != dim:
                    logger.warning(f"Dimension mismatch: got {feat.shape[0]}, expected {dim}")
                    feat = np.zeros(dim, dtype=np.float32)

                embeddings.append(feat)

                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i+1}/{num_elements} charts")

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                embeddings.append(np.zeros(dim, dtype=np.float32))

        embeddings = np.vstack(embeddings).astype(np.float32)
        logger.info(f"Index built with {len(embeddings)} vectors, shape: {embeddings.shape}")
        self.index.add_items(embeddings, np.arange(len(self.metadata)))

    def _save_index(self):
        """Save index to disk."""
        self.index.save_index(self.index_path)
        meta_out = os.path.splitext(self.index_path)[0] + "_meta.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Index saved to {self.index_path}")

    def _load_index(self):
        """Load index from disk."""
        self.index.load_index(self.index_path)

    def search(self, query, k=3):
        """
        Search for charts matching the query.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            List of (score, metadata) tuples
        """
        # Encode query
        q_emb = self.embeddings.encode_text(query, task="retrieval.query")
        feat = q_emb.astype(np.float32)

        labels, distances = self.index.knn_query(feat, k=k)

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            meta = self.metadata[idx].copy()
            if "ocr_text" in meta:
                del meta["ocr_text"]  # Remove long text
            results.append((1 - dist, meta))
        return results
