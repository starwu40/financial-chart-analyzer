"""
Financial Chart Analyzer - AI-powered financial document chart analysis.

This package provides tools for:
- Extracting charts from PDF documents
- Performing OCR on chart elements
- Indexing and retrieving charts using vector embeddings
- Analyzing charts with LLM to answer questions
- Web interface for interactive analysis
"""

__version__ = "0.1.0"
__author__ = "Financial Chart Analyzer"

from financial_chart_analyzer.config import config

__all__ = ["__version__", "__author__", "config"]
