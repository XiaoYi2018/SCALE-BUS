from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from skimage.morphology import (
    area_closing,
    area_opening,
    binary_dilation,
    binary_erosion,
    disk,
    max_tree,
)
from skimage.util import invert

from .features import _prepare_roi


DEFAULT_AREA_THRESHOLDS = (16, 64, 256)
ACTIVE_EPSILON = 0.5 / 255.0


def _component_tree_zone_features(
    image_u8: np.ndarray,
    zone: np.ndarray,
    zone_name: str,
    area_thresholds: Iterable[int] = DEFAULT_AREA_THRESHOLDS,
) -> dict[str, float]:
    if int(zone.sum()) < 32:
        raise ValueError(f"{zone_name} zone is too small")

    zone_values = image_u8[zone]
    fill_value = int(np.rint(np.median(zone_values)))
    filled = image_u8.copy()
    filled[~zone] = fill_value
    filled_float = filled.astype(np.float64) / 255.0
    opening_parent, opening_traverser = max_tree(filled, connectivity=2)
    closing_parent, closing_traverser = max_tree(
        invert(filled),
        connectivity=2,
    )

    result: dict[str, float] = {}
    for area in area_thresholds:
        area = int(area)
        opened = area_opening(
            filled,
            area_threshold=area,
            connectivity=2,
            parent=opening_parent,
            tree_traverser=opening_traverser,
        )
        closed = area_closing(
            filled,
            area_threshold=area,
            connectivity=2,
            parent=closing_parent,
            tree_traverser=closing_traverser,
        )
        bright = np.maximum(
            filled_float - opened.astype(np.float64) / 255.0,
            0.0,
        )[zone]
        dark = np.maximum(
            closed.astype(np.float64) / 255.0 - filled_float,
            0.0,
        )[zone]
        prefix = f"ct_{zone_name}_a{area}"
        result[f"{prefix}_bright_mean"] = float(bright.mean())
        result[f"{prefix}_dark_mean"] = float(dark.mean())
        result[f"{prefix}_bright_q95"] = float(np.quantile(bright, 0.95))
        result[f"{prefix}_dark_q95"] = float(np.quantile(dark, 0.95))
        result[f"{prefix}_bright_fraction"] = float(
            np.mean(bright > ACTIVE_EPSILON)
        )
        result[f"{prefix}_dark_fraction"] = float(
            np.mean(dark > ACTIVE_EPSILON)
        )
    return result


def extract_component_tree_feature_dict(
    image: np.ndarray,
    mask: np.ndarray,
    area_thresholds: Iterable[int] = DEFAULT_AREA_THRESHOLDS,
) -> dict[str, float]:
    """Extract a fixed 54-variable connected-morphology comparator.

    The features are grayscale area-opening/closing residual summaries in the
    same lesion, inner-margin, and outer-margin supports as the primary method.
    """
    prepared_image, roi = _prepare_roi(image, mask)
    image_u8 = np.clip(
        np.rint(prepared_image * 255.0),
        0,
        255,
    ).astype(np.uint8)

    lesion = binary_erosion(roi, disk(2))
    if lesion.sum() < 32:
        lesion = roi
    inner = roi & ~binary_erosion(roi, disk(8))
    outer = binary_dilation(roi, disk(12)) & ~binary_dilation(roi, disk(2))

    result: dict[str, float] = {}
    for zone_name, zone in {
        "lesion": lesion,
        "inner": inner,
        "outer": outer,
    }.items():
        result.update(
            _component_tree_zone_features(
                image_u8,
                zone,
                zone_name,
                area_thresholds,
            )
        )
    if len(result) != 54:
        raise RuntimeError(
            f"expected 54 component-tree features, found {len(result)}"
        )
    for key, value in result.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite component-tree feature {key}={value}")
    return result
