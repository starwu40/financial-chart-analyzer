"""
Data models for chart preprocessing.
"""

from dataclasses import dataclass
from financial_chart_analyzer.config import ChartDetectionConfig


@dataclass
class ChartDetectionConfig:
    """Chart detection configuration class."""
    min_ar: float = 0.25          # Minimum aspect ratio
    max_ar: float = 4.0           # Maximum aspect ratio
    axis_strip_ratio: float = 0.25 # Axis text region ratio
    ocr_upscale_axis: float = 2.5 # Axis patch upscale factor
    min_rel_area: float = 0.005   # Minimum relative area (0.5%)
    min_edge_density: float = 0.002 # Minimum edge density
    padding_w_ratio: float = 0.20 # Horizontal padding ratio
    padding_h_top_ratio: float = 0.25 # Top padding ratio
    padding_h_bottom_ratio: float = 0.20 # Bottom padding ratio
    canny_low: int = 50            # Canny lower threshold
    canny_high: int = 150          # Canny upper threshold
    dpi: int = 220
    max_charts_per_page: int = 8
    morph_kernel_size: int = 5     # Morphology kernel size
    dilate_iterations: int = 2     # Dilation iterations
    axis_extend_ratio: float = 0.30 # Axis extension ratio
    min_width: int = 150           # Minimum width (pixels)
    min_height: int = 150          # Minimum height (pixels)
    min_absolute_area: int = 30000 # Minimum absolute area (pixels^2)
    min_quality_threshold: float = 0.4  # Minimum quality score threshold


def get_config() -> ChartDetectionConfig:
    """Get chart detection configuration from app config."""
    app_config = ChartDetectionConfig()
    return ChartDetectionConfig(
        min_ar=app_config.min_aspect_ratio,
        max_ar=app_config.max_aspect_ratio,
        axis_strip_ratio=app_config.axis_strip_ratio,
        ocr_upscale_axis=app_config.ocr_upscale_axis,
        min_rel_area=app_config.min_relative_area,
        min_edge_density=app_config.min_edge_density,
        padding_w_ratio=app_config.padding_w_ratio,
        padding_h_top_ratio=app_config.padding_h_top_ratio,
        padding_h_bottom_ratio=app_config.padding_h_bottom_ratio,
        canny_low=app_config.canny_low,
        canny_high=app_config.canny_high,
        dpi=app_config.dpi,
        max_charts_per_page=app_config.max_charts_per_page,
        morph_kernel_size=app_config.morph_kernel_size,
        dilate_iterations=app_config.dilate_iterations,
        axis_extend_ratio=app_config.axis_extend_ratio,
        min_width=app_config.min_width,
        min_height=app_config.min_height,
        min_absolute_area=app_config.min_absolute_area,
        min_quality_threshold=app_config.min_quality_threshold,
    )
