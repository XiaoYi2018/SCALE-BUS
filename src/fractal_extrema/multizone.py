from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from skimage.filters import gaussian
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    disk,
    local_maxima,
    local_minima,
)

from .features import (
    DEFAULT_SCALES,
    _box_dimension,
    _component_stats,
    _linear_fit,
    _prepare_roi,
)


def _zone_extrema_features(
    image: np.ndarray,
    zone: np.ndarray,
    zone_name: str,
    scales: Iterable[float] = DEFAULT_SCALES,
) -> dict[str, float]:
    zone_area = float(zone.sum())
    if zone_area < 32:
        raise ValueError(f"{zone_name} zone is too small")
    result: dict[str, float] = {}
    counts: list[float] = []
    scale_values: list[float] = []

    for sigma in scales:
        sigma = float(sigma)
        smooth = image if sigma <= 0 else gaussian(
            image,
            sigma=sigma,
            preserve_range=True,
        )
        maxima = local_maxima(smooth, connectivity=2) & zone
        minima = local_minima(smooth, connectivity=2) & zone
        max_stats = _component_stats(maxima, smooth)
        min_stats = _component_stats(minima, smooth)
        tag = str(int(sigma)) if sigma.is_integer() else str(sigma).replace(".", "p")
        prefix = f"zone_{zone_name}_s{tag}"
        result[f"{prefix}_max_density"] = (
            max_stats["count"] * 10000.0 / zone_area
        )
        result[f"{prefix}_min_density"] = (
            min_stats["count"] * 10000.0 / zone_area
        )
        total = max_stats["count"] + min_stats["count"]
        result[f"{prefix}_balance"] = (
            (max_stats["count"] - min_stats["count"]) / max(total, 1.0)
        )
        result[f"{prefix}_intensity_gap"] = (
            max_stats["intensity_mean"] - min_stats["intensity_mean"]
        )
        if sigma == 1.0:
            max_dimension, _ = _box_dimension(maxima)
            min_dimension, _ = _box_dimension(minima)
            result[f"zone_{zone_name}_s1_max_boxdim"] = max_dimension
            result[f"zone_{zone_name}_s1_min_boxdim"] = min_dimension
        counts.append(total * 10000.0 / zone_area)
        scale_values.append(sigma + 1.0)

    slope, r2 = _linear_fit(
        np.log(np.asarray(scale_values)),
        np.log1p(np.asarray(counts)),
    )
    result[f"zone_{zone_name}_scale_count_slope"] = slope
    result[f"zone_{zone_name}_scale_count_r2"] = r2
    result[f"zone_{zone_name}_density_auc"] = float(
        np.trapezoid(np.asarray(counts), np.log(np.asarray(scale_values)))
    )
    return result


def extract_multizone_feature_dict(
    image: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Extract fixed lesion, inner-margin, and outer-margin extrema signatures."""
    prepared_image, roi = _prepare_roi(image, mask)
    lesion = binary_erosion(roi, disk(2))
    if lesion.sum() < 32:
        lesion = roi
    inner = roi & ~binary_erosion(roi, disk(8))
    outer = binary_dilation(roi, disk(12)) & ~binary_dilation(roi, disk(2))
    zones = {
        "lesion": lesion,
        "inner": inner,
        "outer": outer,
    }
    result: dict[str, float] = {}
    for zone_name, zone in zones.items():
        result.update(_zone_extrema_features(prepared_image, zone, zone_name))
    for key, value in result.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite multizone feature {key}={value}")
    return result
