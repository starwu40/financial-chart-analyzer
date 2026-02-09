"""
Chart detection module using computer vision.
"""

import os
import re
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple
import uuid

from financial_chart_analyzer.preprocessing.ocr_processor import (
    enhance_image, _ocr_image, _numbers_extract
)
from financial_chart_analyzer.preprocessing.models import get_config
from financial_chart_analyzer.config import config

logger = logging.getLogger(__name__)

_CONFIG = get_config()


def _calculate_chart_quality(roi: np.ndarray, ocr_text: str) -> float:
    """Calculate chart quality score (0-1)."""
    score = 0.0
    h, w = roi.shape[:2]

    # 1. Size score (0-0.25)
    if h >= 400 and w >= 400:
        score += 0.25
    elif h >= 300 and w >= 300:
        score += 0.20
    elif h >= 200 and w >= 200:
        score += 0.15
    elif h >= 150 and w >= 150:
        score += 0.10
    else:
        score += 0.05

    # 2. Edge density score (0-0.25)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, _CONFIG.canny_low, _CONFIG.canny_high)
    edge_density = edges.sum() / 255.0 / (h * w + 1e-6)
    if edge_density > 0.015:
        score += 0.25
    elif edge_density > 0.01:
        score += 0.20
    elif edge_density > 0.005:
        score += 0.15
    else:
        score += 0.05

    # 3. Contour complexity (0-0.15)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 20:
        score += 0.15
    elif len(contours) > 10:
        score += 0.10
    else:
        score += 0.05

    # 4. OCR text quality (0-0.20)
    text_len = len(ocr_text.strip())
    if text_len > 100:
        score += 0.20
    elif text_len > 50:
        score += 0.15
    elif text_len > 20:
        score += 0.10
    else:
        score += 0.05

    # 5. Number density (0-0.15)
    numbers = re.findall(r'\d+', ocr_text)
    if len(numbers) > 10:
        score += 0.15
    elif len(numbers) > 5:
        score += 0.10
    elif len(numbers) > 2:
        score += 0.05

    return min(score, 1.0)


def _enhanced_edge_detection(gray: np.ndarray) -> np.ndarray:
    """Enhanced edge detection for chart boundaries."""
    edges1 = cv2.Canny(gray, _CONFIG.canny_low, _CONFIG.canny_high)
    edges2 = cv2.Canny(gray, _CONFIG.canny_low - 20, _CONFIG.canny_high - 30)
    edges = cv2.bitwise_or(edges1, edges2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (_CONFIG.morph_kernel_size, _CONFIG.morph_kernel_size))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.dilate(edges, kernel, iterations=_CONFIG.dilate_iterations)

    return edges


def _detect_axis_regions(roi: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
    """Detect axis regions using projection analysis."""
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    h_proj = np.sum(binary, axis=1)
    v_proj = np.sum(binary, axis=0)

    ext = _CONFIG.axis_extend_ratio

    top_region = int(h * ext)
    bottom_start = int(h * (1 - ext))
    left_region = int(w * ext)
    right_start = int(w * (1 - ext))

    return {
        "top": (0, 0, w, top_region),
        "bottom": (0, bottom_start, w, h),
        "left": (0, 0, left_region, h),
        "right": (right_start, 0, w, h)
    }


def _is_likely_chart(roi: np.ndarray) -> bool:
    """Check if region contains chart-like features."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=30, maxLineGap=10)

    if lines is None or len(lines) < 3:
        return False

    h_lines, v_lines = 0, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 10 or angle > 170:
            h_lines += 1
        elif 80 < angle < 100:
            v_lines += 1

    return h_lines >= 1 and v_lines >= 1


def _calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def detect_charts_final_filtered(img_path, out_dir="charts",
                                 max_charts_per_page=None,
                                 min_rel_area=None, start_count=0):
    """Detect and extract charts from an image."""
    if max_charts_per_page is None:
        max_charts_per_page = _CONFIG.max_charts_per_page
    if min_rel_area is None:
        min_rel_area = _CONFIG.min_rel_area

    os.makedirs(out_dir, exist_ok=True)

    try:
        page = cv2.imread(img_path)
        if page is None:
            logger.warning(f"Cannot read page image: {img_path}")
            return [], start_count

        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        edges = _enhanced_edge_detection(gray)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        H, W = page.shape[:2]
        area_page = H * W
        cand, seen = [], set()

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h

            if area < area_page * min_rel_area:
                continue

            if area < _CONFIG.min_absolute_area:
                continue

            if w < _CONFIG.min_width or h < _CONFIG.min_height:
                continue

            ar = w / (h + 1e-6)
            if ar < _CONFIG.min_ar or ar > _CONFIG.max_ar:
                continue

            base_pad_w = int(w * _CONFIG.padding_w_ratio)
            base_pad_h_top = int(h * _CONFIG.padding_h_top_ratio)
            base_pad_h_bottom = int(h * _CONFIG.padding_h_bottom_ratio)

            if w < 300 or h < 300:
                base_pad_w = int(base_pad_w * 1.5)
                base_pad_h_bottom = int(base_pad_h_bottom * 1.5)

            x1 = max(0, x - base_pad_w)
            y1 = max(0, y - base_pad_h_top)
            x2 = min(W, x + w + base_pad_w)
            y2 = min(H, y + h + base_pad_h_bottom)

            key = (x1, y1, x2, y2)
            if key in seen:
                continue
            seen.add(key)

            roi = page[y1:y2, x1:x2]

            edge_roi = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                                _CONFIG.canny_low, _CONFIG.canny_high)
            density = edge_roi.sum() / 255.0 / (roi.shape[0]*roi.shape[1] + 1e-6)
            if density < _CONFIG.min_edge_density:
                continue

            if not _is_likely_chart(roi):
                continue

            cand.append({
                "bbox": [x1, y1, x2, y2],
                "area": area,
                "ar": ar,
                "density": density
            })

        cand.sort(key=lambda z: z["area"], reverse=True)

        # Remove overlapping regions
        filtered_cand = []
        for c1 in cand:
            is_duplicate = False
            for c2 in filtered_cand:
                iou = _calculate_iou(c1["bbox"], c2["bbox"])
                if iou > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_cand.append(c1)
                if len(filtered_cand) >= max_charts_per_page * 2:
                    break

        cand = filtered_cand

        # Generate candidates with quality scores
        candidates_with_quality = []
        base = os.path.splitext(os.path.basename(img_path))[0]

        for i, c in enumerate(cand):
            x1, y1, x2, y2 = c["bbox"]
            roi = enhance_image(page[y1:y2, x1:x2])

            temp_path = os.path.join(out_dir, f"{base}_temp_{i}.png")
            cv2.imwrite(temp_path, roi, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            ocr_text = _ocr_image(temp_path)
            numbers = _numbers_extract(ocr_text)
            quality = _calculate_chart_quality(roi, ocr_text)

            candidates_with_quality.append({
                "index": i,
                "bbox": c["bbox"],
                "roi": roi,
                "temp_path": temp_path,
                "ocr_text": ocr_text,
                "numbers": numbers,
                "quality": quality,
                "ar": c["ar"],
                "density": c["density"],
                "area": c["area"]
            })

        # Filter by quality
        candidates_with_quality.sort(key=lambda x: x["quality"], reverse=True)
        high_quality_candidates = [
            c for c in candidates_with_quality
            if c["quality"] >= _CONFIG.min_quality_threshold
        ]
        high_quality_candidates = high_quality_candidates[:max_charts_per_page]

        # Save final charts
        metas = []
        current_count = start_count

        for idx, c in enumerate(high_quality_candidates):
            x1, y1, x2, y2 = c["bbox"]
            chart_path = os.path.join(out_dir, f"{base}_chart_{idx+1}.png")

            if os.path.exists(c["temp_path"]):
                os.rename(c["temp_path"], chart_path)
            else:
                cv2.imwrite(chart_path, c["roi"], [cv2.IMWRITE_PNG_COMPRESSION, 3])

            current_count += 1
            logger.info(f"Generated chart {current_count}: {chart_path} (quality: {c['quality']:.2f})")

            metas.append({
                "chart_id": f"c_{uuid.uuid4().hex[:8]}",
                "path": chart_path,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "absolute_area": c["area"],
                "aspect_ratio": round(c["ar"], 3),
                "edge_density": round(c["density"], 4),
                "ocr_text": c["ocr_text"],
                "numbers": c["numbers"],
                "quality_score": round(c["quality"], 3)
            })

        # Cleanup temp files
        for c in candidates_with_quality:
            if os.path.exists(c["temp_path"]) and c not in high_quality_candidates:
                try:
                    os.remove(c["temp_path"])
                except:
                    pass

        return metas, current_count

    except Exception as e:
        logger.error(f"Chart detection failed {img_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [], start_count
