"""
Configuration management for Financial Chart Analyzer.
"""

import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class APIConfig:
    """Configuration for API keys and endpoints."""

    def __init__(self):
        # Jina API
        self.jina_api_key = os.getenv("JINA_API_KEY", "")
        self.jina_api_url = os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
        self.jina_model_name = os.getenv("JINA_MODEL_NAME", "jina-embeddings-v4")

        # OpenAI API
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        #you can use other LLM models from other providers as well by changing the api_url and model_name accordingly

class TesseractConfig:
    """Configuration for Tesseract OCR."""

    def __init__(self):
        self.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
        self.ocr_languages = os.getenv("OCR_LANGUAGES", "chi_sim+eng")

    @property
    def tesseract_cmd_path(self) -> str:
        """Return the Tesseract command path"""
        if os.path.exists(r"D:\tesseract\tesseract.exe"):
            return r"D:\tesseract\tesseract.exe"
        #this is an example for windows, modify it according to your OS and installation path   
        return self.tesseract_cmd


class ChartDetectionConfig:
    """Configuration for chart detection parameters."""

    def __init__(self):
        # Size constraints
        self.min_aspect_ratio = float(os.getenv("CHART_MIN_AR", "0.25"))
        self.max_aspect_ratio = float(os.getenv("CHART_MAX_AR", "4.0"))
        self.min_width = int(os.getenv("CHART_MIN_WIDTH", "150"))
        self.min_height = int(os.getenv("CHART_MIN_HEIGHT", "150"))
        self.min_absolute_area = int(os.getenv("CHART_MIN_ABSOLUTE_AREA", "30000"))
        self.min_relative_area = float(os.getenv("CHART_MIN_REL_AREA", "0.005"))

        # Detection thresholds
        self.min_edge_density = float(os.getenv("CHART_MIN_EDGE_DENSITY", "0.002"))
        self.min_quality_threshold = float(os.getenv("CHART_MIN_QUALITY_THRESHOLD", "0.4"))
        self.max_charts_per_page = int(os.getenv("MAX_CHARTS_PER_PAGE", "8"))

        # Image processing
        self.dpi = int(os.getenv("PDF_DPI", "220"))
        self.canny_low = int(os.getenv("CANNY_LOW", "50"))
        self.canny_high = int(os.getenv("CANNY_HIGH", "150"))
        self.morph_kernel_size = int(os.getenv("MORPH_KERNEL_SIZE", "5"))
        self.dilate_iterations = int(os.getenv("DILATE_ITERATIONS", "2"))

        # Padding ratios
        self.padding_w_ratio = float(os.getenv("PADDING_W_RATIO", "0.20"))
        self.padding_h_top_ratio = float(os.getenv("PADDING_H_TOP_RATIO", "0.25"))
        self.padding_h_bottom_ratio = float(os.getenv("PADDING_H_BOTTOM_RATIO", "0.20"))

        # Axis detection
        self.axis_strip_ratio = float(os.getenv("AXIS_STRIP_RATIO", "0.25"))
        self.ocr_upscale_axis = float(os.getenv("OCR_UPSCALE_AXIS", "2.5"))
        self.axis_extend_ratio = float(os.getenv("AXIS_EXTEND_RATIO", "0.30"))


class RetrievalConfig:
    """Configuration for vector search retrieval."""

    def __init__(self):
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "2048"))
        self.max_results = int(os.getenv("MAX_RESULTS", "3"))
        self.ef_construction = int(os.getenv("EF_CONSTRUCTION", "200"))
        self.ef = int(os.getenv("EF", "50"))
        self.alpha = float(os.getenv("RETRIEVAL_ALPHA", "0.6"))


class AppConfig:
    """Main application configuration."""

    def __init__(self):
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Directory paths (relative to current working directory)
        base_path = Path.cwd()
        self.cache_dir = Path(os.getenv("CACHE_DIR", base_path / "cache"))
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", base_path / "uploads"))
        self.processed_dir = Path(os.getenv("PROCESSED_DIR", base_path / "processed"))
        self.preprocess_dir = Path(os.getenv("PREPROCESS_DIR", base_path / "preprocessed"))
        self.preprocess_cache_dir = Path(os.getenv("PREPROCESS_CACHE_DIR", base_path / "preprocessed_cache"))

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.preprocess_dir.mkdir(parents=True, exist_ok=True)
        self.preprocess_cache_dir.mkdir(parents=True, exist_ok=True)

        # Sub-configurations
        self.api = APIConfig()
        self.tesseract = TesseractConfig()
        self.chart_detection = ChartDetectionConfig()
        self.retrieval = RetrievalConfig()

        # Processing options
        self.use_multiprocessing = os.getenv("USE_MULTIPROCESSING", "false").lower() == "true"
        self.max_workers = int(os.getenv("MAX_WORKERS", "0")) or None


# Global configuration instance
config = AppConfig()
