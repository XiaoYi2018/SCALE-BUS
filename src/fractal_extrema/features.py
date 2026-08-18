from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import binary_erosion, disk, local_maxima, local_minima
from skimage.transform import resize


EPS = np.finfo(np.float64).eps
DEFAULT_SCALES = (0.0, 1.0, 2.0, 4.0)


FEATURE_GROUPS = {
    "basic": ("basic_",),
    "fractal": ("fractal_",),
    "extrema": ("extrema_",),
    "basic_fractal": ("basic_", "fractal_"),
    "basic_extrema": ("basic_", "extrema_"),
    "fractal_extrema": ("fractal_", "extrema_"),
    "fused": ("basic_", "fractal_", "extrema_"),
}

COMPACT_EXTREMA_FEATURES = tuple(
    [
        f"extrema_s{scale}_{name}"
        for scale in (0, 1, 2, 4)
        for name in (
            "max_density",
            "min_density",
            "balance",
            "intensity_gap",
        )
    ]
    + [
        "extrema_scale_count_slope",
        "extrema_scale_count_r2",
        "extrema_density_auc",
    ]
)

FORMAL_FEATURE_GROUPS = (
    "basic",
    "fractal",
    "extrema_compact",
    "basic_fractal",
    "basic_extrema_compact",
    "fractal_extrema_compact",
    "fused_compact",
    "fused_full",
)

MULTIZONE_FEATURE_GROUPS = (
    "basic",
    "basic_fractal",
    "basic_extrema_compact",
    "multizone_extrema",
    "basic_margin_extrema",
    "basic_multizone_extrema",
    "fused_multizone",
)


def feature_columns_for_group(columns: Iterable[str], group: str) -> list[str]:
    """Return a fixed, label-independent feature subset for an ablation group."""
    ordered = list(columns)
    basic = [column for column in ordered if column.startswith("basic_")]
    fractal = [column for column in ordered if column.startswith("fractal_")]
    extrema_full = [column for column in ordered if column.startswith("extrema_")]
    extrema_compact = [
        column for column in ordered if column in COMPACT_EXTREMA_FEATURES
    ]
    multizone = [column for column in ordered if column.startswith("zone_")]
    margin = [
        column
        for column in ordered
        if column.startswith(("zone_inner_", "zone_outer_"))
    ]
    groups = {
        "basic": basic,
        "fractal": fractal,
        "extrema": extrema_full,
        "extrema_full": extrema_full,
        "extrema_compact": extrema_compact,
        "basic_fractal": basic + fractal,
        "basic_extrema": basic + extrema_full,
        "basic_extrema_compact": basic + extrema_compact,
        "fractal_extrema": fractal + extrema_full,
        "fractal_extrema_compact": fractal + extrema_compact,
        "fused": basic + fractal + extrema_full,
        "fused_full": basic + fractal + extrema_full,
        "fused_compact": basic + fractal + extrema_compact,
        "multizone_extrema": multizone,
        "basic_margin_extrema": basic + margin,
        "basic_multizone_extrema": basic + multizone,
        "fractal_multizone_extrema": fractal + multizone,
        "fused_multizone": basic + fractal + multizone,
    }
    if group not in groups:
        raise KeyError(f"unknown feature group: {group}")
    selected = groups[group]
    if not selected:
        raise ValueError(f"feature group {group!r} selected no columns")
    return selected


def _as_gray_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float64)
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    else:
        arr = arr.astype(np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("image contains no finite pixels")
    lo, hi = np.nanpercentile(arr[finite], [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _prepare_roi(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: int = 256,
    pad_fraction: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    gray = _as_gray_float(image)
    raw_mask = np.asarray(mask)
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[..., :3].max(axis=2)
    mask_max = float(np.nanmax(raw_mask))
    if mask_max <= 1.0:
        roi = raw_mask > 0
    else:
        roi = raw_mask > max(8.0, 0.10 * mask_max)
    if roi.shape != gray.shape:
        roi = resize(
            roi.astype(np.uint8),
            gray.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0
    ys, xs = np.where(roi)
    if ys.size == 0:
        raise ValueError("empty lesion mask")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad = int(round(max(y1 - y0, x1 - x0) * pad_fraction))
    y0, y1 = max(0, y0 - pad), min(gray.shape[0], y1 + pad)
    x0, x1 = max(0, x0 - pad), min(gray.shape[1], x1 + pad)
    gray_crop = gray[y0:y1, x0:x1]
    roi_crop = roi[y0:y1, x0:x1]
    gray_fixed = resize(
        gray_crop,
        (output_size, output_size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float64)
    roi_fixed = resize(
        roi_crop.astype(np.uint8),
        (output_size, output_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0
    if roi_fixed.sum() < 64:
        raise ValueError("lesion mask is too small after resizing")
    return gray_fixed, roi_fixed


def _shannon_entropy(values: np.ndarray, bins: int = 32) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins)
    prob = hist[hist > 0].astype(np.float64)
    prob /= prob.sum()
    return float(-(prob * np.log2(prob)).sum())


def _basic_features(image: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    values = image[roi]
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    result = {
        "basic_mean": float(values.mean()),
        "basic_std": float(values.std()),
        "basic_q10": float(q10),
        "basic_q25": float(q25),
        "basic_median": float(q50),
        "basic_q75": float(q75),
        "basic_q90": float(q90),
        "basic_iqr": float(q75 - q25),
        "basic_entropy": _shannon_entropy(values),
        "basic_roi_fraction": float(roi.mean()),
    }

    filled = image.copy()
    filled[~roi] = q50
    quant = np.clip(np.rint(filled * 31.0), 0, 31).astype(np.uint8)
    glcm = graycomatrix(
        quant,
        distances=[1, 2, 4],
        angles=[0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0],
        levels=32,
        symmetric=True,
        normed=True,
    )
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        result[f"basic_glcm_{prop}"] = float(np.nanmean(graycoprops(glcm, prop)))

    lbp = local_binary_pattern(quant, P=8, R=1, method="uniform")
    hist = np.bincount(lbp[roi].astype(np.int64), minlength=10).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    for idx, value in enumerate(hist[:10]):
        result[f"basic_lbp_{idx:02d}"] = float(value)
    return result


def _box_counts(binary: np.ndarray, box_sizes: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    counts: list[float] = []
    valid_sizes: list[float] = []
    arr = np.asarray(binary, dtype=bool)
    height, width = arr.shape
    for size in box_sizes:
        size = int(size)
        if size < 2 or size > min(height, width):
            continue
        pad_h = (-height) % size
        pad_w = (-width) % size
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")
        blocks = padded.reshape(
            padded.shape[0] // size,
            size,
            padded.shape[1] // size,
            size,
        )
        occupied = blocks.any(axis=(1, 3))
        count = float(occupied.sum())
        if count > 0:
            valid_sizes.append(float(size))
            counts.append(count)
    return np.asarray(valid_sizes), np.asarray(counts)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.square(y - fitted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / max(ss_tot, EPS)
    return float(slope), float(r2)


def _box_dimension(binary: np.ndarray) -> tuple[float, float]:
    sizes = (2, 4, 8, 16, 32, 64)
    valid, counts = _box_counts(binary, sizes)
    slope, r2 = _linear_fit(np.log(1.0 / valid), np.log(counts))
    return slope, r2


def _lacunarity(binary: np.ndarray, sizes: Iterable[int] = (4, 8, 16, 32)) -> dict[str, float]:
    arr = np.asarray(binary, dtype=np.float64)
    height, width = arr.shape
    result: dict[str, float] = {}
    values: list[float] = []
    for size in sizes:
        size = int(size)
        pad_h = (-height) % size
        pad_w = (-width) % size
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")
        blocks = padded.reshape(
            padded.shape[0] // size,
            size,
            padded.shape[1] // size,
            size,
        )
        mass = blocks.sum(axis=(1, 3)).ravel()
        mean = float(mass.mean())
        lac = float(mass.var() / max(mean * mean, EPS) + 1.0)
        result[f"fractal_lacunarity_s{size}"] = lac
        values.append(lac)
    result["fractal_lacunarity_mean"] = float(np.mean(values))
    return result


def _differential_box_slope(image: np.ndarray, roi: np.ndarray) -> tuple[float, float]:
    filled = image.copy()
    filled[~roi] = float(np.median(image[roi]))
    sizes = np.asarray([4, 8, 16, 32, 64], dtype=np.int64)
    measures: list[float] = []
    valid: list[float] = []
    height, width = filled.shape
    for size in sizes:
        if size > min(height, width):
            continue
        pad_h = (-height) % int(size)
        pad_w = (-width) % int(size)
        padded = np.pad(filled, ((0, pad_h), (0, pad_w)), mode="edge")
        blocks = padded.reshape(
            padded.shape[0] // size,
            size,
            padded.shape[1] // size,
            size,
        )
        local_range = blocks.max(axis=(1, 3)) - blocks.min(axis=(1, 3))
        measure = float(np.maximum(1.0, np.ceil(local_range * 255.0 / size)).sum())
        valid.append(float(size))
        measures.append(measure)
    slope, r2 = _linear_fit(
        np.log(1.0 / np.asarray(valid)),
        np.log(np.asarray(measures)),
    )
    return slope, r2


def _fractal_features(image: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    boundary = roi ^ binary_erosion(roi, disk(1))
    boundary_dim, boundary_r2 = _box_dimension(boundary)
    region_dim, region_r2 = _box_dimension(roi)
    gray_slope, gray_r2 = _differential_box_slope(image, roi)
    result = {
        "fractal_boundary_boxdim": boundary_dim,
        "fractal_boundary_boxdim_r2": boundary_r2,
        "fractal_region_boxdim": region_dim,
        "fractal_region_boxdim_r2": region_r2,
        "fractal_gray_dbc_slope": gray_slope,
        "fractal_gray_dbc_r2": gray_r2,
    }
    result.update(_lacunarity(roi))
    return result


def _component_stats(mask: np.ndarray, image: np.ndarray) -> dict[str, float]:
    labels = label(mask, connectivity=2)
    count = int(labels.max())
    if count == 0:
        return {
            "count": 0.0,
            "area_mean": 0.0,
            "area_median": 0.0,
            "area_q90": 0.0,
            "area_entropy": 0.0,
            "intensity_mean": 0.0,
        }
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.float64)
    sums = ndi.sum(image, labels, index=np.arange(1, count + 1))
    intensity = np.asarray(sums, dtype=np.float64) / np.maximum(areas, 1.0)
    return {
        "count": float(count),
        "area_mean": float(areas.mean()),
        "area_median": float(np.median(areas)),
        "area_q90": float(np.quantile(areas, 0.90)),
        "area_entropy": _shannon_entropy(areas, bins=min(16, max(2, count))),
        "intensity_mean": float(intensity.mean()),
    }


def _extrema_features(
    image: np.ndarray,
    roi: np.ndarray,
    scales: Iterable[float] = DEFAULT_SCALES,
) -> dict[str, float]:
    valid_roi = binary_erosion(roi, disk(2))
    if valid_roi.sum() < 32:
        valid_roi = roi
    roi_area = float(valid_roi.sum())
    result: dict[str, float] = {}
    counts: list[float] = []
    scale_values: list[float] = []

    for sigma in scales:
        sigma = float(sigma)
        smooth = image if sigma <= 0 else gaussian(image, sigma=sigma, preserve_range=True)
        maxima = local_maxima(smooth, connectivity=2) & valid_roi
        minima = local_minima(smooth, connectivity=2) & valid_roi
        max_stats = _component_stats(maxima, smooth)
        min_stats = _component_stats(minima, smooth)
        tag = str(int(sigma)) if sigma.is_integer() else str(sigma).replace(".", "p")
        for prefix, stats in (("max", max_stats), ("min", min_stats)):
            result[f"extrema_s{tag}_{prefix}_density"] = stats["count"] * 10000.0 / roi_area
            for name in ("area_mean", "area_median", "area_q90", "area_entropy", "intensity_mean"):
                result[f"extrema_s{tag}_{prefix}_{name}"] = stats[name]
        total = max_stats["count"] + min_stats["count"]
        result[f"extrema_s{tag}_balance"] = (
            (max_stats["count"] - min_stats["count"]) / max(total, 1.0)
        )
        result[f"extrema_s{tag}_intensity_gap"] = (
            max_stats["intensity_mean"] - min_stats["intensity_mean"]
        )
        counts.append(total * 10000.0 / roi_area)
        scale_values.append(sigma + 1.0)

    slope, r2 = _linear_fit(
        np.log(np.asarray(scale_values)),
        np.log1p(np.asarray(counts)),
    )
    result["extrema_scale_count_slope"] = slope
    result["extrema_scale_count_r2"] = r2
    result["extrema_density_auc"] = float(
        np.trapezoid(np.asarray(counts), np.log(np.asarray(scale_values)))
    )
    return result


def extract_feature_dict(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Extract the fixed whole-lesion feature signature from an image-mask pair."""
    prepared_image, prepared_roi = _prepare_roi(image, mask)
    result: dict[str, float] = {}
    result.update(_basic_features(prepared_image, prepared_roi))
    result.update(_fractal_features(prepared_image, prepared_roi))
    result.update(_extrema_features(prepared_image, prepared_roi))
    for key, value in result.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite feature {key}={value}")
    return result
