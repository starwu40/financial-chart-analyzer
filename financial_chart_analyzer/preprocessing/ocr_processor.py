"""
OCR processing module for extracting text from charts.
"""

import os
import re
import logging
import hashlib
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance

from financial_chart_analyzer.config import config

logger = logging.getLogger(__name__)

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = config.tesseract.tesseract_cmd_path

# OCR cache
_OCR_CACHE = {}


def _hash_path(path: str) -> str:
    """Calculate hash of a file path for caching."""
    st = os.stat(path)
    return hashlib.md5(f"{path}:{int(st.st_mtime)}".encode()).hexdigest()


def enhance_image(img_bgr):
    """Enhance image contrast and sharpness."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Contrast(pil).enhance(1.3)
    pil = ImageEnhance.Sharpness(pil).enhance(1.5)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _deskew(gray):
    """Correct image skew/rotation."""
    coords = np.column_stack(np.where(gray > 0))
    if coords.size == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _preprocess_for_ocr(img_bgr):
    """Preprocess image for OCR."""
    try:
        if img_bgr.shape[0] < 400 or img_bgr.shape[1] < 400:
            img_bgr = cv2.resize(img_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        blur = cv2.bilateralFilter(img_bgr, 5, 55, 55)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        gray = _deskew(gray)
        thr = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        return morph
    except Exception as e:
        logger.error(f"OCR preprocessing failed: {e}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _numbers_extract(text: str) -> list:
    """Extract numbers from text."""
    nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?%?', text)
    nums = [n.replace(',', '') for n in nums]
    years = re.findall(r'\b20\d{2}\b', text)
    all_nums = list(set(nums + years))
    try:
        return sorted(all_nums, key=lambda x: float(re.sub(r'[^\d.-]', '', x)))
    except:
        return sorted(all_nums)


def _lang_guess(text: str) -> str:
    """Guess OCR language from text content."""
    zh = len(re.findall(r'[\u4e00-\u9fa5]', text))
    en = len(re.findall(r'[A-Za-z]', text))
    return "chi_sim+eng" if zh >= en else "eng+chi_sim"


def _ocr_blocks(proc_img):
    """OCR in blocks for better accuracy."""
    H, W = proc_img.shape
    step = max(400, W // 3)
    segs = []
    for x0 in range(0, W, step):
        crop = proc_img[:, x0:min(W, x0 + step)]
        seg = pytesseract.image_to_string(crop, lang=config.tesseract.ocr_languages)
        segs.append(seg)
    text = "\n".join(segs)
    lang2 = _lang_guess(text)
    if lang2 != config.tesseract.ocr_languages:
        retry = pytesseract.image_to_string(proc_img, lang=lang2)
        if len(retry) > len(text) * 0.8:
            text = retry
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    return text.strip()


def _ocr_raw(img_gray, lang="chi_sim+eng", psm=6):
    """Low-level OCR call."""
    cfg = f"--psm {psm} -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(img_gray, lang=lang, config=cfg)


def _ocr_axis_patches(roi_bgr):
    """OCR axis patches for tick labels."""
    h, w = roi_bgr.shape[:2]
    if h < 30 or w < 30:
        return ""
    r = 0.25
    scale = 2.5
    roi_up = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    hu, wu = roi_up.shape[:2]
    bw = int(wu * r)
    bh = int(hu * r)
    patches = [
        roi_up[0:bh, :],
        roi_up[hu - bh:hu, :],
        roi_up[:, 0:bw],
        roi_up[:, wu - bw:wu]
    ]

    texts = []
    for p in patches:
        gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        proc = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kern, iterations=1)
        t = _ocr_raw(proc, psm=7)
        t = re.sub(r'[^\S\r\n]+', ' ', t).strip()
        if t:
            texts.append(t)
    merged = " ".join(texts)
    merged = re.sub(r'\s{2,}', ' ', merged)
    return merged.strip()


def _ocr_axis_enhanced(roi_bgr):
    """Enhanced axis OCR with smart region detection."""
    h, w = roi_bgr.shape[:2]
    if h < 30 or w < 30:
        return ""

    ext = 0.30
    scale = 2.5
    roi_up = cv2.resize(roi_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)

    regions = {
        "top": (0, 0, w, int(h * ext)),
        "bottom": (0, int(h * (1 - ext)), w, h),
        "left": (0, 0, int(w * ext), h),
        "right": (int(w * (1 - ext)), 0, w, h)
    }

    all_texts = []

    for region_name, (x1, y1, x2, y2) in regions.items():
        x1_s, y1_s = int(x1 * scale), int(y1 * scale)
        x2_s, y2_s = int(x2 * scale), int(y2 * scale)
        patch = roi_up[y1_s:y2_s, x1_s:x2_s]

        if patch.size == 0:
            continue

        texts = []
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        t1 = _ocr_raw(binary, psm=11)
        if t1.strip():
            texts.append(t1)

        inv = cv2.bitwise_not(binary)
        t2 = _ocr_raw(inv, psm=11)
        if t2.strip() and t2 != t1:
            texts.append(t2)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t3 = _ocr_raw(otsu, psm=6)
        if t3.strip() and len(t3) > len(t1):
            texts.append(t3)

        region_text = " ".join(texts)
        region_text = re.sub(r'[^\S\r\n]+', ' ', region_text).strip()
        if region_text:
            all_texts.append(f"[{region_name}] {region_text}")

    merged = " ".join(all_texts)
    merged = re.sub(r'\s{2,}', ' ', merged)
    return merged.strip()


def _ocr_image(path: str) -> str:
    """OCR with caching and error handling."""
    try:
        h = _hash_path(path)
        if h in _OCR_CACHE:
            return _OCR_CACHE[h]

        if not os.path.exists(path):
            logger.warning(f"Image file not found: {path}")
            _OCR_CACHE[h] = ""
            return ""

        img = cv2.imread(path)
        if img is None:
            logger.warning(f"Cannot read image: {path}")
            _OCR_CACHE[h] = ""
            return ""

        proc = _preprocess_for_ocr(img)
        main_text = _ocr_blocks(proc)
        axis_text = _ocr_axis_enhanced(img)
        combined = (main_text + " " + axis_text).strip()
        combined = re.sub(r'\s{2,}', ' ', combined)
        _OCR_CACHE[h] = combined
        return combined
    except Exception as e:
        logger.error(f"OCR processing failed {path}: {e}")
        _OCR_CACHE[h] = ""
        return ""
