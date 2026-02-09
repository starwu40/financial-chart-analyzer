# Financial Chart Analyzer

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An AI-powered financial document chart analysis system that automatically extracts, indexes, and analyzes charts from PDF financial reports.

## Features

- **Automated Chart Detection**: Uses computer vision to identify and extract charts from PDF documents
- **OCR Integration**: Extracts text and numerical data from chart elements using Tesseract OCR
- **Vector Search**: Indexes charts using Jina embeddings for semantic search and retrieval
- **LLM Analysis**: Uses OpenAI/DeepSeek models to analyze charts and answer natural language questions
- **Web Interface**: Interactive Streamlit-based UI for easy document upload and querying
- **Caching**: Smart caching to avoid reprocessing the same documents
- **Multiprocessing**: Optional parallel processing for faster PDF analysis

## Installation

### Prerequisites

- Python 3.9 or higher
- Tesseract OCR engine
  - **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`

### Install from source

```bash
cd financial-chart-analyzer
pip install -r requirements.txt
```

## Quick Start

### 1. Configure API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
JINA_API_KEY=your_jina_api_key_here
TESSERACT_CMD=/path/to/tesseract
```

### 2. Preprocess a PDF Document

```python
from financial_chart_analyzer.preprocessing import preprocess_pdf_charts

preprocess_pdf_charts(
    "financial_report.pdf",
    out_dir="./processed",
    use_multiprocessing=True
)
```

### 3. Analyze with a Query

```python
from financial_chart_analyzer.retrieval import ChartRetriever
from financial_chart_analyzer.analysis import ChartAnalyzer

# Initialize retriever
retriever = ChartRetriever(
    metadata_path="./processed/charts_index.json"
)

# Search for relevant charts
results = retriever.search("revenue trends", k=3)

# Analyze with LLM
analyzer = ChartAnalyzer()
answer = analyzer.analyze_and_answer("revenue trends", results)
print(answer)
```

### 4. Launch Web Interface

```bash
streamlit run financial_chart_analyzer/web/app.py
```

Then open your browser to `http://localhost:8501`

## Configuration

Configuration can be done through environment variables:

| Variable | Description | Default |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `JINA_API_KEY` | Jina API key | (required) |
| `TESSERACT_CMD` | Tesseract executable path | `tesseract` |
| `OPENAI_API_URL` | OpenAI API endpoint | `https://api.openai.com/v1/chat/completions` |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `JINA_API_URL` | Jina API endpoint | `https://api.jina.ai/v1/embeddings` |
| `JINA_MODEL_NAME` | Jina model name | `jina-embeddings-v4` |
| `MAX_CHARTS_PER_PAGE` | Maximum charts per page | `8` |
| `MIN_QUALITY_THRESHOLD` | Minimum quality score | `0.4` |
| `PDF_DPI` | PDF rendering DPI | `220` |
| `USE_MULTIPROCESSING` | Enable multiprocessing | `false` |

## Architecture

```
financial_chart_analyzer/
├── preprocessing/    # PDF processing & chart detection
├── retrieval/        # Vector search & embeddings
├── analysis/         # LLM-based analysis
├── web/              # Streamlit web interface
├── utils/            # Utility functions
└── config.py         # Configuration management
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
