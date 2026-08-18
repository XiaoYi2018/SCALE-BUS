from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pywt
from skimage.filters import sobel
from skimage.measure import find_contours, label, perimeter, regionprops
from skimage.morphology import binary_dilation, binary_erosion, disk

from .features import EPS, _linear_fit, _prepare_roi


BOX_SIZES = (4, 8, 16, 32)
WAVELET = "db2"
WAVELET_LEVEL = 3


def _zone_masks(roi: np.ndarray) -> dict[str, np.ndarray]:
    lesion = binary_erosion(roi, disk(2))
    if lesion.sum() < 32:
        lesion = roi.copy()
    inner = roi & ~binary_erosion(roi, disk(8))
    outer = binary_dilation(roi, disk(12)) & ~binary_dilation(roi, disk(2))
    zones = {"lesion": lesion, "inner": inner, "outer": outer}
    for name, zone in zones.items():
        if zone.sum() < 32:
            raise ValueError(f"{name} zone is too small for advanced features")
    return zones


def _block_masses(
    measure: np.ndarray,
    support: np.ndarray,
    sizes: Iterable[int] = BOX_SIZES,
) -> tuple[np.ndarray, list[np.ndarray]]:
    valid_sizes: list[float] = []
    probabilities: list[np.ndarray] = []
    height, width = measure.shape
    for size_value in sizes:
        size = int(size_value)
        if size > min(height, width):
            continue
        pad_h = (-height) % size
        pad_w = (-width) % size
        padded_measure = np.pad(
            measure,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
        )
        padded_support = np.pad(
            support,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
        )
        measure_blocks = padded_measure.reshape(
            padded_measure.shape[0] // size,
            size,
            padded_measure.shape[1] // size,
            size,
        )
        support_blocks = padded_support.reshape(
            padded_support.shape[0] // size,
            size,
            padded_support.shape[1] // size,
            size,
        )
        mass = measure_blocks.sum(axis=(1, 3)).ravel()
        occupied = support_blocks.any(axis=(1, 3)).ravel()
        mass = mass[occupied]
        mass = mass[mass > 0]
        if mass.size < 2:
            continue
        probability = mass / mass.sum()
        valid_sizes.append(float(size))
        probabilities.append(probability)
    return np.asarray(valid_sizes, dtype=np.float64), probabilities


def _generalized_dimensions(
    measure: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    sizes, probabilities = _block_masses(measure, support)
    if sizes.size < 3:
        return {
            "d0": 0.0,
            "d1": 0.0,
            "d2": 0.0,
            "spread": 0.0,
            "r2_mean": 0.0,
        }
    x = np.log(1.0 / sizes)
    y0 = np.asarray([np.log(len(p)) for p in probabilities])
    y1 = np.asarray([-np.sum(p * np.log(p + EPS)) for p in probabilities])
    y2 = np.asarray([-np.log(np.sum(np.square(p)) + EPS) for p in probabilities])
    d0, r0 = _linear_fit(x, y0)
    d1, r1 = _linear_fit(x, y1)
    d2, r2 = _linear_fit(x, y2)
    return {
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "spread": float(np.ptp([d0, d1, d2])),
        "r2_mean": float(np.mean([r0, r1, r2])),
    }


def _multifractal_zone_features(
    image: np.ndarray,
    zone: np.ndarray,
    zone_name: str,
) -> dict[str, float]:
    values = image[zone]
    shifted = np.zeros_like(image, dtype=np.float64)
    shifted[zone] = values - float(values.min()) + 1.0 / 255.0
    gradient = sobel(image)
    gradient_measure = np.zeros_like(image, dtype=np.float64)
    gradient_measure[zone] = gradient[zone] + 1.0 / 255.0

    result: dict[str, float] = {}
    for signal_name, measure in (
        ("intensity", shifted),
        ("gradient", gradient_measure),
    ):
        dimensions = _generalized_dimensions(measure, zone)
        for statistic, value in dimensions.items():
            result[
                f"advanced_mf_{zone_name}_{signal_name}_{statistic}"
            ] = value
    return result


def _coefficient_entropy(coefficients: np.ndarray) -> float:
    magnitude = np.abs(np.asarray(coefficients, dtype=np.float64)).ravel()
    total = float(magnitude.sum())
    if total <= EPS:
        return 0.0
    probability = magnitude / total
    return float(-np.sum(probability * np.log2(probability + EPS)))


def _wavelet_zone_features(
    image: np.ndarray,
    zone: np.ndarray,
    zone_name: str,
) -> dict[str, float]:
    filled = np.asarray(image, dtype=np.float64).copy()
    filled[~zone] = float(np.median(filled[zone]))
    coefficients = pywt.wavedec2(
        filled,
        wavelet=WAVELET,
        level=WAVELET_LEVEL,
        mode="symmetric",
    )
    total_energy = float(
        sum(np.square(np.asarray(item)).sum() for detail in coefficients[1:] for item in detail)
    )
    result: dict[str, float] = {}
    for level, (horizontal, vertical, diagonal) in zip(
        range(WAVELET_LEVEL, 0, -1),
        coefficients[1:],
    ):
        energy_h = float(np.square(horizontal).sum())
        energy_v = float(np.square(vertical).sum())
        energy_d = float(np.square(diagonal).sum())
        detail_energy = energy_h + energy_v + energy_d
        detail_vector = np.concatenate(
            [
                np.asarray(horizontal).ravel(),
                np.asarray(vertical).ravel(),
                np.asarray(diagonal).ravel(),
            ]
        )
        prefix = f"advanced_wavelet_{zone_name}_l{level}"
        result[f"{prefix}_energy"] = detail_energy / max(total_energy, EPS)
        result[f"{prefix}_entropy"] = _coefficient_entropy(detail_vector)
        result[f"{prefix}_hv_logratio"] = float(
            np.log((energy_h + EPS) / (energy_v + EPS))
        )
        result[f"{prefix}_diagonal_fraction"] = (
            energy_d / max(detail_energy, EPS)
        )
    return result


def _resample_closed_contour(
    contour: np.ndarray,
    samples: int = 128,
) -> np.ndarray:
    points = np.asarray(contour, dtype=np.float64)
    if points.shape[0] < 4:
        return np.zeros((samples, 2), dtype=np.float64)
    points = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if cumulative[-1] <= EPS:
        return np.repeat(points[:1], samples, axis=0)
    targets = np.linspace(0.0, cumulative[-1], samples, endpoint=False)
    row = np.interp(targets, cumulative, points[:, 0])
    column = np.interp(targets, cumulative, points[:, 1])
    return np.column_stack([row, column])


def _entropy_histogram(values: np.ndarray, bins: int = 16) -> float:
    histogram, _ = np.histogram(values, bins=bins)
    probability = histogram.astype(np.float64)
    probability /= max(float(probability.sum()), 1.0)
    nonzero = probability[probability > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _boundary_features(roi: np.ndarray) -> dict[str, float]:
    labels = label(roi, connectivity=2)
    properties = regionprops(labels)
    if not properties:
        raise ValueError("lesion mask contains no component")
    region = max(properties, key=lambda item: item.area)
    component = labels == region.label
    area = float(region.area)
    boundary_length = float(perimeter(component, neighborhood=8))
    contours = find_contours(component.astype(np.float64), 0.5)
    if not contours:
        raise ValueError("lesion mask has no contour")
    contour = _resample_closed_contour(max(contours, key=len))
    centroid = np.asarray(region.centroid, dtype=np.float64)
    radial = np.linalg.norm(contour - centroid[None, :], axis=1)
    segments = np.diff(np.vstack([contour, contour[:2]]), axis=0)
    directions = np.arctan2(segments[:, 0], segments[:, 1])
    turning = np.angle(np.exp(1j * np.diff(directions)))
    minor = max(float(region.minor_axis_length), EPS)
    return {
        "advanced_shape_perimeter_normalized": boundary_length / np.sqrt(area),
        "advanced_shape_compactness": (
            4.0 * np.pi * area / max(boundary_length * boundary_length, EPS)
        ),
        "advanced_shape_solidity": float(region.solidity),
        "advanced_shape_eccentricity": float(region.eccentricity),
        "advanced_shape_axis_ratio": float(region.major_axis_length) / minor,
        "advanced_shape_convex_area_ratio": float(region.convex_area) / area,
        "advanced_shape_radial_cv": float(radial.std()) / max(float(radial.mean()), EPS),
        "advanced_shape_radial_entropy": _entropy_histogram(radial),
        "advanced_shape_turning_abs_mean": float(np.mean(np.abs(turning))),
        "advanced_shape_turning_std": float(np.std(turning)),
    }


def extract_advanced_feature_dict(
    image: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Extract the frozen 76-variable multifractal, wavelet, and shape block."""
    prepared_image, roi = _prepare_roi(image, mask)
    zones = _zone_masks(roi)
    result: dict[str, float] = {}
    for zone_name, zone in zones.items():
        result.update(
            _multifractal_zone_features(prepared_image, zone, zone_name)
        )
        result.update(_wavelet_zone_features(prepared_image, zone, zone_name))
    result.update(_boundary_features(roi))
    if len(result) != 76:
        raise RuntimeError(
            f"advanced feature count is {len(result)}, expected 76"
        )
    for key, value in result.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite advanced feature {key}={value}")
    return result
