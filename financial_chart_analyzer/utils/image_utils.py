"""
Image utility functions.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def enhance_image(img_bgr):
    """
    Enhance image contrast and sharpness.

    Args:
        img_bgr: BGR image array

    Returns:
        Enhanced BGR image array
    """
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Contrast(pil).enhance(1.3)
    pil = ImageEnhance.Sharpness(pil).enhance(1.5)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def encode_image_to_base64(image, max_size=1536, quality=75):
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
    import io
    import base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str
