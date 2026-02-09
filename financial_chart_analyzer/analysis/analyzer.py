"""
LLM-based chart analyzer for answering questions about charts.
"""

import os
import io
import base64
import logging
import time
from typing import List, Tuple
import requests
from PIL import Image

from financial_chart_analyzer.analysis.prompts import SYSTEM_PROMPT, CLOSING_INSTRUCTION
from financial_chart_analyzer.config import config

logger = logging.getLogger(__name__)


class ChartAnalyzer:
    """
    Analyze charts using OpenAI/LLM API to answer questions.
    """

    def __init__(self, api_key=None, api_url=None, model_name=None):
        """
        Initialize chart analyzer.

        Args:
            api_key: OpenAI API key (default from config)
            api_url: API endpoint URL (default from config)
            model_name: Model name (default from config)
        """
        self.api_key = api_key or config.api.openai_api_key
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. LLM analysis will not work.")

        self.api_url = api_url or config.api.openai_api_url
        self.model_name = model_name or config.api.openai_model
        logger.info(f"Initialized LLM client with model: {self.model_name}")

    @staticmethod
    def _encode_image(image, max_size=1536, quality=75):
        """
        Encode PIL Image to base64 string.

        Args:
            image: PIL Image object
            max_size: Maximum dimension
            quality: JPEG quality (0-100)

        Returns:
            Base64 encoded string
        """
        image.thumbnail((max_size, max_size))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.debug(f"Encoded image size: {len(img_str) / 1024:.2f} KB")
        return img_str

    def analyze_and_answer(self, query: str, retrieved_charts: List[Tuple[float, dict]]) -> str:
        """
        Analyze retrieved charts and answer the query using LLM.

        Args:
            query: User's question
            retrieved_charts: List of (score, metadata) tuples

        Returns:
            LLM-generated answer
        """
        if not self.api_key:
            return "Error: LLM analyzer not initialized. Please set OPENAI_API_KEY."

        if not retrieved_charts:
            return "No relevant charts found to answer this question."

        logger.info("Preparing prompt for LLM...")

        # Build user content
        user_question_text = f"**User Question:** {query}\n\nPlease carefully analyze the following charts and answer the above question:"

        user_content = [
            {"type": "text", "text": user_question_text}
        ]

        # Add images
        for i, (score, meta) in enumerate(retrieved_charts):
            chart_path = meta.get("path")
            if chart_path and os.path.exists(chart_path):
                try:
                    logger.info(f"Loading and encoding chart {i+1}: {chart_path}")

                    with open(chart_path, "rb") as img_file:
                        img_data = img_file.read()
                    base64_image = base64.b64encode(img_data).decode('utf-8')
                    logger.debug(f"Original size: {len(img_data) / 1024:.2f} KB, Base64: {len(base64_image) / 1024:.2f} KB")

                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
                    user_content.append({
                        "type": "text",
                        "text": f"[Chart {i+1}]"
                    })

                except Exception as e:
                    logger.warning(f"Failed to load/encode image {chart_path}: {e}")

        user_content.append({
            "type": "text",
            "text": CLOSING_INSTRUCTION
        })

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
            "stream": False
        }

        # Send request with retry logic
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            logger.info(f"Sending request to LLM API (attempt {attempt + 1}/{max_retries})... This may take some time...")
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=180,
                    verify=False
                )

                logger.info(f"API response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    answer = result['choices'][0]['message']['content']

                    logger.debug(f"API Response Success! Answer length: {len(answer)} characters")
                    return answer

                elif 400 <= response.status_code < 500:
                    logger.error(f"API client error: {response.text}")
                    return f"API request failed (client error), status: {response.status_code}, response: {response.text}"
                else:
                    logger.error(f"API error response: {response.text}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        return f"API request failed, status: {response.status_code}, response: {response.text}"

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error communicating with model: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return f"Serious error communicating with model, multiple retries failed: {e}"

        return "Error: All retry attempts failed."
