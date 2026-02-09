"""
Embeddings encoding module using Jina API.
"""

import base64
import logging
import numpy as np
import requests

from financial_chart_analyzer.config import config

logger = logging.getLogger(__name__)


class JinaEmbeddings:
    """Jina embeddings API client."""

    def __init__(self, api_key=None, api_url=None, model_name=None):
        """
        Initialize Jina embeddings client.

        Args:
            api_key: Jina API key (default from config)
            api_url: API endpoint URL
            model_name: Model name
        """
        self.api_key = api_key or config.api.jina_api_key
        if not self.api_key:
            raise ValueError("JINA_API_KEY not set")

        self.api_url = api_url or config.api.jina_api_url
        self.model_name = model_name or config.api.jina_model_name
        self.embedding_dim = config.retrieval.embedding_dim

    def encode_image(self, image_path, timeout=60):
        """
        Encode an image to embedding vector.

        Args:
            image_path: Path to image file
            timeout: Request timeout in seconds

        Returns:
            Numpy array of shape (1, embedding_dim)
        """
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model_name,
                "task": "retrieval.passage",
                "input": [
                    {"image": img_base64}
                ]
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            result = response.json()
            embedding = np.array(result["data"][0]["embedding"], dtype=np.float32)
            return embedding.reshape(1, -1)

        except requests.exceptions.HTTPError as e:
            logger.error(f"Image encoding HTTP error: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text}")
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

    def encode_text(self, text, task="retrieval.passage", timeout=30):
        """
        Encode text to embedding vector.

        Args:
            text: Text to encode
            task: Task type (retrieval.passage or retrieval.query)
            timeout: Request timeout in seconds

        Returns:
            Numpy array of shape (1, embedding_dim)
        """
        if not text.strip():
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model_name,
                "task": task,
                "input": [
                    {"text": text}
                ]
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            result = response.json()
            embedding = np.array(result["data"][0]["embedding"], dtype=np.float32)
            return embedding.reshape(1, -1)

        except requests.exceptions.HTTPError as e:
            logger.error(f"Text encoding HTTP error: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text}")
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
