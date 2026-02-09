"""
PDF to image conversion module.
"""

import os
import fitz  # PyMuPDF
import logging

from financial_chart_analyzer.config import config

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path, out_dir, dpi=None):
    """
    Convert PDF pages to PNG images.

    Args:
        pdf_path: Path to the PDF file
        out_dir: Output directory for images
        dpi: DPI for rendering (default from config)

    Returns:
        List of paths to the generated images
    """
    if dpi is None:
        dpi = config.chart_detection.dpi

    os.makedirs(out_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        paths = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            img_path = os.path.join(out_dir, f"page_{i+1}.png")
            pix.save(img_path)
            paths.append(img_path)
            logger.debug(f"Converted page {i+1} to {img_path}")

        doc.close()
        logger.info(f"Converted {len(paths)} pages from {pdf_path}")
        return paths

    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise
