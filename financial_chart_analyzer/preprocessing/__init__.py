"""
PDF and chart preprocessing module.
"""

import os
import json
import shutil
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from financial_chart_analyzer.preprocessing.pdf_processor import pdf_to_images
from financial_chart_analyzer.preprocessing.chart_detector import detect_charts_final_filtered
from financial_chart_analyzer.preprocessing.models import ChartDetectionConfig, get_config
from financial_chart_analyzer.config import config

logger = logging.getLogger(__name__)


def clear_dir(dir_path):
    """Clear directory contents."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def _process_single_page(args):
    """Process a single page for multiprocessing."""
    idx, page_img, charts_dir, max_charts_per_page, start_count = args
    page_id = f"p_{idx+1:04d}"
    charts, new_count = detect_charts_final_filtered(
        page_img, charts_dir,
        max_charts_per_page=max_charts_per_page,
        start_count=start_count
    )
    for m in charts:
        m["page_id"] = page_id
    return idx, charts, new_count


def preprocess_pdf_charts(pdf_path, out_dir="preprocessed",
                          max_charts_per_page=None,
                          use_multiprocessing=False,
                          max_workers=None):
    """
    Process PDF to extract and analyze charts.

    Args:
        pdf_path: Path to PDF file
        out_dir: Output directory
        max_charts_per_page: Maximum charts per page
        use_multiprocessing: Use multiprocessing for faster processing
        max_workers: Number of worker processes

    Returns:
        None (results saved to out_dir)
    """
    if max_charts_per_page is None:
        max_charts_per_page = config.chart_detection.max_charts_per_page

    clear_dir(out_dir)
    pages_dir = os.path.join(out_dir, "pages")
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    logger.info(f"Starting PDF processing: {pdf_path}")
    pages = pdf_to_images(pdf_path, pages_dir, dpi=config.chart_detection.dpi)
    logger.info(f"Converted {len(pages)} pages")

    all_meta = []

    if use_multiprocessing and len(pages) > 1:
        if max_workers is None:
            max_workers = min(os.cpu_count() or 1, len(pages))

        logger.info(f"Using {max_workers} processes for parallel processing")
        tasks = [(i, p, charts_dir, max_charts_per_page, i * max_charts_per_page)
                 for i, p in enumerate(pages)]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single_page, task): task[0]
                      for task in tasks}

            for future in as_completed(futures):
                try:
                    idx, charts, _ = future.result()
                    all_meta.extend(charts)
                    logger.info(f"Page {idx+1} completed, detected {len(charts)} charts")
                except Exception as e:
                    logger.error(f"Page processing failed: {e}")
    else:
        total = 0
        for idx, page_img in enumerate(pages):
            page_id = f"p_{idx+1:04d}"
            charts, total = detect_charts_final_filtered(
                page_img, charts_dir,
                max_charts_per_page=max_charts_per_page,
                start_count=total
            )
            for m in charts:
                m["page_id"] = page_id
            all_meta.extend(charts)
            logger.info(f"Page {page_id} completed, added {len(charts)} charts, total {total}")

    # Sort by quality score
    all_meta.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

    # Save metadata
    meta_path = os.path.join(out_dir, "charts_index.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    # Generate statistics
    if all_meta:
        stats = {
            "total_pages": len(pages),
            "total_charts": len(all_meta),
            "avg_quality_score": round(sum(m.get('quality_score', 0) for m in all_meta) / len(all_meta), 3),
            "quality_distribution": {
                "excellent (>=0.7)": sum(1 for m in all_meta if m.get('quality_score', 0) >= 0.7),
                "good (0.5-0.7)": sum(1 for m in all_meta if 0.5 <= m.get('quality_score', 0) < 0.7),
                "acceptable (0.4-0.5)": sum(1 for m in all_meta if 0.4 <= m.get('quality_score', 0) < 0.5),
                "poor (<0.4)": sum(1 for m in all_meta if m.get('quality_score', 0) < 0.4)
            }
        }
    else:
        stats = {
            "total_pages": len(pages),
            "total_charts": 0,
            "message": "No charts detected"
        }

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Processing complete! Total charts: {len(all_meta)}")
    logger.info(f"Index file: {meta_path}")
    logger.info(f"Stats file: {stats_path}")


__all__ = [
    "pdf_to_images",
    "detect_charts_final_filtered",
    "ChartDetectionConfig",
    "preprocess_pdf_charts",
]
