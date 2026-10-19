"""Batch convergence diagnostics for existing Monte Carlo score draws."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from metrology_paths import OUTPUT_ROOT

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
RESULTS = OUTPUT_ROOT
CASE_DIR = RESULTS / "v07_uncertainty_propagation" / "cases"
SELECTED = RESULTS / "v07_uncertainty_propagation" / "selected_cases.csv"
SCORE_UNCERTAINTY = RESULTS / "v07_uncertainty_propagation" / "per_image_score_uncertainty.csv"
VARIANCE_SUMMARY = RESULTS / "v07_uncertainty_propagation" / "variance_budget_summary.csv"
PAIR_FILE = RESULTS / "v07_interview_reproducibility" / "included_lesion_pairs.csv"
PAIRED_SCORES = RESULTS / "v07_interview_reproducibility" / "paired_locked_probabilities.csv"
OUTPUT = OUTPUT_ROOT / "metrology_statistics" / "mc_convergence"
SEED = 20261018
CONTOUR_LEVELS = (0.01, 0.025, 0.05)
PRIMARY_LEVEL = 0.025
SINGLE_SOURCES = ("speckle", "gamma", "resolution")
JOINT_BATCHES = 10
RATIO_BATCHES = 5
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_path(dataset: str, image: str) -> Path:
    token = hashlib.sha256(str(image).encode("utf-8")).hexdigest()[:20]
    return CASE_DIR / f"case_{dataset}_{token}.npz"


def split_batches(n_draws: int, n_batches: int) -> list[np.ndarray]:
    return [part for part in np.array_split(np.arange(n_draws), n_batches) if len(part)]


def batch_matrix(values: np.ndarray, n_batches: int) -> tuple[np.ndarray, np.ndarray]:
    chunks = split_batches(values.shape[1], n_batches)
    means = np.column_stack([values[:, index].mean(axis=1) for index in chunks])
    stds = np.column_stack([values[:, index].std(axis=1, ddof=1) for index in chunks])
    return means, stds


def one_batch_mcse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def two_significant_digit_tolerance(value: float) -> float:
    value = abs(float(value))
    if not math.isfinite(value) or value == 0:
        return float("nan")
    exponent = math.floor(math.log10(value))
    return float(0.5 * 10.0 ** (exponent - 1))


def convergence_row(quantity: str, level: float | None, point: float, batches: np.ndarray, tolerance: float) -> dict:
    batches = np.asarray(batches, dtype=float)
    mcse = one_batch_mcse(batches)
    return {
        "quantity": quantity,
        "contour_sigma": level,
        "point_estimate": float(point),
        "batch_count": int(len(batches)),
        "batch_mean": float(batches.mean()),
        "mc_standard_error": mcse,
        "two_times_mcse": float(2.0 * mcse),
        "numerical_tolerance": float(tolerance),
        "passes_2mcse": bool(2.0 * mcse <= tolerance),
    }


def ratio_batches(u2_batch: np.ndarray, first: np.ndarray, second: np.ndarray, mask: np.ndarray, observed: float) -> np.ndarray:
    result = []
    for batch_index in range(u2_batch.shape[1]):
        pair_variance = (u2_batch[first, batch_index] + u2_batch[second, batch_index]) / 2.0
        result.append(float(pair_variance[mask].mean() / observed))
    return np.asarray(result, dtype=float)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory already exists: {OUTPUT}")

    selected = pd.read_csv(SELECTED)
    paired = pd.read_csv(PAIRED_SCORES)
    pairs = pd.read_csv(PAIR_FILE)
    score_uncertainty = pd.read_csv(SCORE_UNCERTAINTY)
    pd.read_csv(VARIANCE_SUMMARY)

    if len(selected) != 2127 or selected[["dataset", "image"]].duplicated().any():
        raise RuntimeError("Unexpected selected-case inventory")
    if len(paired) != 810 or not paired["Case"].is_unique:
        raise RuntimeError("Expected 810 unique paired lesions")
    if len(pairs) != 810:
        raise RuntimeError("Expected 810 included lesion pairs")

    expected_paths = [case_path(str(row.dataset), str(row.image)) for row in selected.itertuples()]
    actual_paths = sorted(CASE_DIR.glob("case_*.npz"))
    if {p.name for p in expected_paths} != {p.name for p in actual_paths}:
        raise RuntimeError("Case cache inventory differs from selected cases")

    image_count = len(selected)
    joint = {level: np.full((image_count, 50), np.nan) for level in CONTOUR_LEVELS}
    contour = {level: np.full((image_count, 20), np.nan) for level in CONTOUR_LEVELS}
    singles = {source: np.full((image_count, 20), np.nan) for source in SINGLE_SOURCES}
    base_z = np.full(image_count, np.nan, dtype=float)
    case_hash_rows = []

    for row_index, (row, path) in enumerate(zip(selected.itertuples(index=False), expected_paths)):
        with np.load(path, allow_pickle=False) as archive:
            z = np.asarray(archive["z"], dtype=float)
            source = np.asarray(archive["source"]).astype(str)
            amplitude = np.asarray(archive["amplitude"], dtype=float)
            repetition = np.asarray(archive["rep"], dtype=int)
            base = float(np.asarray(archive["base_p"]).reshape(-1)[0])
        base_z[row_index] = np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
        for level in CONTOUR_LEVELS:
            selected_joint = (source == "joint") & np.isclose(amplitude, level, atol=1e-12, rtol=0)
            selected_contour = (source == "contour") & np.isclose(amplitude, level, atol=1e-12, rtol=0)
            joint_values = z[selected_joint][np.argsort(repetition[selected_joint])]
            contour_values = z[selected_contour][np.argsort(repetition[selected_contour])]
            if len(joint_values) != 50 or len(contour_values) != 20:
                raise RuntimeError(f"Unexpected draw count for {path.name}, level {level}")
            joint[level][row_index] = joint_values
            contour[level][row_index] = contour_values
        for source_name in SINGLE_SOURCES:
            selected_single = (source == source_name) & np.isclose(amplitude, 0.0, atol=1e-12, rtol=0)
            values = z[selected_single][np.argsort(repetition[selected_single])]
            if len(values) != 20:
                raise RuntimeError(f"Unexpected single-source draw count for {path.name}, source {source_name}")
            singles[source_name][row_index] = values
        case_hash_rows.append({"dataset": row.dataset, "image": row.image, "file": path.name, "sha256": sha256(path)})

    case_hashes = pd.DataFrame(case_hash_rows)
    case_hash_text = case_hashes.to_csv(index=False, lineterminator="\n")

    first_image = paired["first_image"].astype(str).to_numpy()
    second_image = paired["second_image"].astype(str).to_numpy()
    pair_lookup = {(str(row.dataset), str(row.image)): index for index, row in zip(range(image_count), selected.itertuples(index=False))}
    pair_first = np.asarray([pair_lookup[("busbra", image)] for image in first_image], dtype=int)
    pair_second = np.asarray([pair_lookup[("busbra", image)] for image in second_image], dtype=int)
    pair_target = paired["target"].to_numpy(dtype=int)
    pair_image_indices = np.unique(np.concatenate([pair_first, pair_second]))
    pair_image_set = set(first_image.tolist()) | set(second_image.tolist())
    pair_local_first = np.searchsorted(pair_image_indices, pair_first)
    pair_local_second = np.searchsorted(pair_image_indices, pair_second)

    probability_first = paired["advanced76_first"].to_numpy(dtype=float)
    probability_second = paired["advanced76_second"].to_numpy(dtype=float)
    logit_first = np.log(np.clip(probability_first, 1e-6, 1 - 1e-6) / (1 - np.clip(probability_first, 1e-6, 1 - 1e-6)))
    logit_second = np.log(np.clip(probability_second, 1e-6, 1 - 1e-6) / (1 - np.clip(probability_second, 1e-6, 1 - 1e-6)))
    paired_variance = (logit_first - logit_second) ** 2 / 2.0
    observed_variance = {
        "all": float(paired_variance.mean()),
        "0": float(paired_variance[pair_target == 0].mean()),
        "1": float(paired_variance[pair_target == 1].mean()),
    }

    joint_u2 = {}
    for level in CONTOUR_LEVELS:
        _, stds = batch_matrix(joint[level], JOINT_BATCHES)
        joint_u2[level] = stds ** 2

    contour_u2_ratio = {}
    single_u2_ratio = {}
    for level in CONTOUR_LEVELS:
        _, stds = batch_matrix(contour[level], RATIO_BATCHES)
        contour_u2_ratio[level] = stds ** 2
    for source_name in SINGLE_SOURCES:
        _, stds = batch_matrix(singles[source_name], RATIO_BATCHES)
        single_u2_ratio[source_name] = stds ** 2

    batch_rows = []
    convergence_rows = []

    for level in CONTOUR_LEVELS:
        batch_u2 = joint_u2[level][pair_image_indices]
        full_u2 = (joint[level][pair_image_indices].std(axis=1, ddof=1) ** 2)
        batch_means = batch_u2.mean(axis=0)
        point_u2 = float(full_u2.mean())
        tolerance = two_significant_digit_tolerance(math.sqrt(point_u2))
        convergence_rows.append(convergence_row("joint_mean_standard_uncertainty_squared", level, point_u2, batch_means, tolerance))
        for batch_index, value in enumerate(batch_means):
            batch_rows.append({"quantity": "joint_mean_standard_uncertainty_squared", "contour_sigma": level, "batch": batch_index + 1, "value": float(value)})

        for group_name, mask in (("all", np.ones(len(pair_target), dtype=bool)), ("benign", pair_target == 0), ("malignant", pair_target == 1)):
            batches = ratio_batches(batch_u2, pair_local_first, pair_local_second, mask, observed_variance["0" if group_name == "benign" else "1" if group_name == "malignant" else "all"])
            pair_values = (full_u2[pair_local_first] + full_u2[pair_local_second]) / 2.0
            point_ratio = float(pair_values[mask].mean() / observed_variance["0" if group_name == "benign" else "1" if group_name == "malignant" else "all"])
            tolerance = two_significant_digit_tolerance(point_ratio)
            convergence_rows.append(convergence_row(f"propagated_observed_variance_ratio:{group_name}", level, point_ratio, batches, tolerance))
            for batch_index, value in enumerate(batches):
                batch_rows.append({"quantity": f"propagated_observed_variance_ratio:{group_name}", "contour_sigma": level, "batch": batch_index + 1, "value": float(value)})

    share_groups = {"contour": contour_u2_ratio[PRIMARY_LEVEL], **single_u2_ratio}
    for source_name, source_u2 in share_groups.items():
        source_means = source_u2[pair_image_indices].mean(axis=0)
        batches = source_means / observed_variance["all"]
        source_draws_primary = contour[PRIMARY_LEVEL] if source_name == "contour" else singles[source_name]
        point_share = float(np.mean(source_draws_primary[pair_image_indices].std(axis=1, ddof=1) ** 2) / observed_variance["all"])
        tolerance = two_significant_digit_tolerance(point_share)
        convergence_rows.append(convergence_row(f"variance_share:{source_name}", PRIMARY_LEVEL, point_share, batches, tolerance))
        for batch_index, value in enumerate(batches):
            batch_rows.append({"quantity": f"variance_share:{source_name}", "contour_sigma": PRIMARY_LEVEL, "batch": batch_index + 1, "value": float(value)})

    source_draws = {}
    for level in CONTOUR_LEVELS:
        source_draws[("joint", level)] = joint[level]
        source_draws[("contour", level)] = contour[level]
    for source_name in SINGLE_SOURCES:
        source_draws[(source_name, 0.0)] = singles[source_name]

    datasets = selected["dataset"].astype(str).to_numpy()
    targets = selected["target"].to_numpy(dtype=int)
    for dataset in ("busbra", "breast"):
        dataset_mask = datasets == dataset
        for (source_name, level), draws in source_draws.items():
            for class_name, class_value in (("all", None), ("benign", 0), ("malignant", 1)):
                image_mask = dataset_mask.copy()
                if class_value is not None:
                    image_mask &= targets == class_value
                shifts = draws[image_mask].mean(axis=1) - base_z[image_mask]
                chunks = split_batches(draws.shape[1], 5)
                batch_values = np.asarray([np.median(shifts[index]) for index in chunks], dtype=float)
                point = float(np.median(shifts))
                tolerance = two_significant_digit_tolerance(point)
                quantity = f"median_logit_shift:{dataset}:{source_name}:{level:g}:{class_name}"
                convergence_rows.append(convergence_row(quantity, level, point, batch_values, tolerance))
                for batch_index, value in enumerate(batch_values):
                    batch_rows.append({"quantity": quantity, "contour_sigma": level, "batch": batch_index + 1, "value": float(value)})

    per_image_rows = []
    for level in CONTOUR_LEVELS:
        batch_u = joint_u2[level] ** 0.5
        overall_u = joint[level].std(axis=1, ddof=1)
        mcse_u = batch_u.std(axis=1, ddof=1) / math.sqrt(JOINT_BATCHES)
        relative = mcse_u / overall_u
        per_image_rows.append({"contour_sigma": level, "image_count": image_count, "median_relative_mcse_u": float(np.median(relative)), "q95_relative_mcse_u": float(np.quantile(relative, 0.95))})

    first25_rows = []
    for level in CONTOUR_LEVELS:
        subset = joint[level][pair_image_indices, :25]
        _, subset_stds = batch_matrix(subset, RATIO_BATCHES)
        subset_u2 = subset_stds ** 2
        subset_full_u2 = subset.std(axis=1, ddof=1) ** 2
        for group_name, mask in (("all", np.ones(len(pair_target), dtype=bool)), ("benign", pair_target == 0), ("malignant", pair_target == 1)):
            observed = observed_variance["0" if group_name == "benign" else "1" if group_name == "malignant" else "all"]
            batch_ratio = ratio_batches(subset_u2, pair_local_first, pair_local_second, mask, observed)
            pair_values = (subset_full_u2[pair_local_first] + subset_full_u2[pair_local_second]) / 2.0
            first25_rows.append({"contour_sigma": level, "quantity": f"propagated_observed_variance_ratio:{group_name}", "first25_ratio": float(pair_values[mask].mean() / observed), "first25_batch_mcse": one_batch_mcse(batch_ratio)})
        point_all = float(np.mean(joint[level][pair_image_indices].std(axis=1, ddof=1) ** 2))
        subset_all = float(np.mean(subset.std(axis=1, ddof=1) ** 2))
        first25_rows.append({"contour_sigma": level, "quantity": "joint_mean_standard_uncertainty_squared", "first25_ratio": subset_all, "first25_batch_mcse": one_batch_mcse(subset_u2.mean(axis=0)), "full_value": point_all})

    expected_rows = score_uncertainty[(score_uncertainty["output"] == "logit") & (score_uncertainty["source"] == "joint")].copy()
    expected_rows = expected_rows[(expected_rows["dataset"] == "busbra") & expected_rows["image"].astype(str).isin(pair_image_set)]
    reproduction_checks = []
    joint_points = {
        level: next(row["point_estimate"] for row in convergence_rows if row["quantity"] == "joint_mean_standard_uncertainty_squared" and row["contour_sigma"] == level)
        for level in CONTOUR_LEVELS
    }
    for level in CONTOUR_LEVELS:
        expected = float((expected_rows.loc[np.isclose(expected_rows["contour_sigma"], level), "u"].to_numpy() ** 2).mean())
        actual = joint_points[level]
        reproduction_checks.append({"quantity": "joint_mean_standard_uncertainty_squared", "contour_sigma": level, "reported_from_existing_csv": expected, "recomputed_from_cases": actual, "absolute_difference": abs(expected - actual)})

    batch_frame = pd.DataFrame(batch_rows)
    convergence_frame = pd.DataFrame(convergence_rows)
    per_image_frame = pd.DataFrame(per_image_rows)
    first25_frame = pd.DataFrame(first25_rows)
    reproduction_frame = pd.DataFrame(reproduction_checks)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory appeared during analysis: {OUTPUT}")
    OUTPUT.mkdir()
    case_hashes.to_csv(OUTPUT / "case_hashes.csv", index=False, lineterminator="\n")
    batch_frame.to_csv(OUTPUT / "batch_estimates.csv", index=False, lineterminator="\n")
    convergence_frame.to_csv(OUTPUT / "metric_convergence.csv", index=False, lineterminator="\n")
    per_image_frame.to_csv(OUTPUT / "per_image_mcse.csv", index=False, lineterminator="\n")
    first25_frame.to_csv(OUTPUT / "first25_vs_all.csv", index=False, lineterminator="\n")
    reproduction_frame.to_csv(OUTPUT / "reproduction_checks.csv", index=False, lineterminator="\n")

    output_hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    input_hashes = {
        "selected_cases.csv": sha256(SELECTED),
        "per_image_score_uncertainty.csv": sha256(SCORE_UNCERTAINTY),
        "variance_budget_summary.csv": sha256(VARIANCE_SUMMARY),
        "included_lesion_pairs.csv": sha256(PAIR_FILE),
        "paired_locked_probabilities.csv": sha256(PAIRED_SCORES),
        "case_hash_manifest_sha256": hashlib.sha256(case_hash_text.encode("utf-8")).hexdigest(),
    }
    summary = {
        "analysis": "monte_carlo_batch_convergence",
        "seed": SEED,
        "bootstrap_iterations": 0,
        "joint_batches": JOINT_BATCHES,
        "ratio_batches": RATIO_BATCHES,
        "input_hashes": input_hashes,
        "case_count": int(image_count),
        "paired_lesions": int(len(paired)),
        "observed_interview_variance": observed_variance,
        "key_metrics": convergence_frame.to_dict(orient="records"),
        "per_image_mcse": per_image_frame.to_dict(orient="records"),
        "reproduction_checks": reproduction_frame.to_dict(orient="records"),
        "output_hashes": output_hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "checks": len(reproduction_checks), "metrics": len(convergence_frame)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
