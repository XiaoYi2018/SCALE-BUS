"""Patient-level intervals for propagated-to-observed variance ratios."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from metrology_paths import OUTPUT_ROOT

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

RESULTS = OUTPUT_ROOT
CASE_DIR = RESULTS / "v07_uncertainty_propagation" / "cases"
SELECTED_CASES = RESULTS / "v07_uncertainty_propagation" / "selected_cases.csv"
PAIR_FILE = (
    RESULTS
    / "v07_interview_reproducibility"
    / "included_lesion_pairs.csv"
)
PAIRED_SCORES = (
    RESULTS
    / "v07_interview_reproducibility"
    / "paired_locked_probabilities.csv"
)
REFERENCE_SUMMARY = (
    RESULTS
    / "metrology_statistics"
    / "mc_convergence"
    / "summary.json"
)
OUTPUT = OUTPUT_ROOT / "metrology_statistics" / "ratio_interval"

SEED = 20261018
BOOTSTRAP_ITERATIONS = 2000
CONTOUR_LEVELS = (0.01, 0.025, 0.05)
JOINT_DRAWS = 50
JOINT_BATCHES = 10
GROUPS = ("all", "benign", "malignant")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_path(dataset: str, image: str) -> Path:
    token = hashlib.sha256(str(image).encode("utf-8")).hexdigest()[:20]
    return CASE_DIR / f"case_{dataset}_{token}.npz"


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), (0.025, 0.975))
    return float(low), float(high)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory already exists: {OUTPUT}")

    selected = pd.read_csv(SELECTED_CASES)
    pairs = pd.read_csv(PAIR_FILE)
    paired = pd.read_csv(PAIRED_SCORES)
    reference = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))

    if len(selected) != 2127 or selected[["dataset", "image"]].duplicated().any():
        raise RuntimeError("Unexpected selected-case inventory")
    if len(pairs) != 810 or not pairs["Case"].is_unique:
        raise RuntimeError("Expected 810 included lesion pairs")
    if len(paired) != 810 or not paired["Case"].is_unique:
        raise RuntimeError("Expected 810 paired score rows")

    expected_paths = [
        case_path(str(row.dataset), str(row.image))
        for row in selected.itertuples(index=False)
    ]
    actual_paths = sorted(CASE_DIR.glob("case_*.npz"))
    if {path.name for path in expected_paths} != {path.name for path in actual_paths}:
        raise RuntimeError("Case cache inventory differs from selected cases")

    image_lookup = {
        (str(row.dataset), str(row.image)): index
        for index, row in enumerate(selected.itertuples(index=False))
    }
    first_image = pairs["first_image"].astype(str).to_numpy()
    second_image = pairs["second_image"].astype(str).to_numpy()
    first_index = np.asarray(
        [image_lookup[("busbra", image)] for image in first_image], dtype=int
    )
    second_index = np.asarray(
        [image_lookup[("busbra", image)] for image in second_image], dtype=int
    )
    pair_images = np.unique(np.concatenate([first_index, second_index]))
    pair_local_first = np.searchsorted(pair_images, first_index)
    pair_local_second = np.searchsorted(pair_images, second_index)
    pair_target = paired["target"].to_numpy(dtype=int)

    probability_first = paired["advanced76_first"].to_numpy(dtype=float)
    probability_second = paired["advanced76_second"].to_numpy(dtype=float)
    logit_first = np.log(
        np.clip(probability_first, 1e-6, 1 - 1e-6)
        / (1 - np.clip(probability_first, 1e-6, 1 - 1e-6))
    )
    logit_second = np.log(
        np.clip(probability_second, 1e-6, 1 - 1e-6)
        / (1 - np.clip(probability_second, 1e-6, 1 - 1e-6))
    )
    paired_variance = (logit_first - logit_second) ** 2 / 2.0
    denominator_by_group = {
        "all": float(paired_variance.mean()),
        "benign": float(paired_variance[pair_target == 0].mean()),
        "malignant": float(paired_variance[pair_target == 1].mean()),
    }

    full_u2: dict[float, np.ndarray] = {}
    batch_u2: dict[float, np.ndarray] = {}
    case_manifest_rows: list[dict[str, str]] = []

    for level in CONTOUR_LEVELS:
        level_full = np.full(len(pair_images), np.nan, dtype=float)
        level_batches = np.full((len(pair_images), JOINT_BATCHES), np.nan, dtype=float)
        for local_index, image_index in enumerate(pair_images):
            row = selected.iloc[image_index]
            path = case_path(str(row["dataset"]), str(row["image"]))
            with np.load(path, allow_pickle=False) as archive:
                scores = np.asarray(archive["z"], dtype=float)
                source = np.asarray(archive["source"]).astype(str)
                amplitude = np.asarray(archive["amplitude"], dtype=float)
                repetition = np.asarray(archive["rep"], dtype=int)
            selected_joint = (source == "joint") & np.isclose(
                amplitude, level, atol=1e-12, rtol=0
            )
            joint_values = scores[selected_joint][
                np.argsort(repetition[selected_joint])
            ]
            if len(joint_values) != JOINT_DRAWS:
                raise RuntimeError(
                    f"Unexpected joint draw count for {path.name}, level {level}"
                )
            level_full[local_index] = float(joint_values.std(ddof=1) ** 2)
            for batch_index, draw_index in enumerate(
                np.array_split(np.arange(JOINT_DRAWS), JOINT_BATCHES)
            ):
                level_batches[local_index, batch_index] = float(
                    joint_values[draw_index].std(ddof=1) ** 2
                )
            case_manifest_rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "image": str(row["image"]),
                    "file": path.name,
                    "sha256": sha256(path),
                }
            )
        full_u2[level] = level_full
        batch_u2[level] = level_batches

    case_manifest = (
        pd.DataFrame(case_manifest_rows)
        .drop_duplicates(subset=["dataset", "image"])
        .sort_values(["dataset", "image"])
        .reset_index(drop=True)
    )
    case_manifest_text = case_manifest.to_csv(index=False, lineterminator="\n")

    reference_points = {
        (float(row["contour_sigma"]), row["quantity"].split(":")[-1]): float(
            row["point_estimate"]
        )
        for row in reference["key_metrics"]
        if row["quantity"].startswith("propagated_observed_variance_ratio:")
    }

    point_rows: list[dict[str, float | int | str]] = []
    pair_full_u2: dict[float, np.ndarray] = {}
    pair_batch_u2: dict[float, np.ndarray] = {}
    for level in CONTOUR_LEVELS:
        pair_full_u2[level] = (
            full_u2[level][pair_local_first] + full_u2[level][pair_local_second]
        ) / 2.0
        pair_batch_u2[level] = (
            batch_u2[level][pair_local_first] + batch_u2[level][pair_local_second]
        ) / 2.0
        for group in GROUPS:
            mask = (
                np.ones(len(pair_target), dtype=bool)
                if group == "all"
                else pair_target == (0 if group == "benign" else 1)
            )
            numerator = float(pair_full_u2[level][mask].mean())
            denominator = denominator_by_group[group]
            ratio = numerator / denominator
            expected = reference_points[(level, group)]
            point_rows.append(
                {
                    "contour_sigma": level,
                    "lesion_group": group,
                    "paired_lesions": int(mask.sum()),
                    "propagated_variance": numerator,
                    "observed_variance": denominator,
                    "ratio": ratio,
                    "reference_ratio": expected,
                    "absolute_difference": abs(ratio - expected),
                }
            )

    rng = np.random.default_rng(SEED)
    distribution_rows: list[dict[str, float | int | str]] = []
    interval_rows: list[dict[str, float | int | str]] = []

    for level in CONTOUR_LEVELS:
        for group in GROUPS:
            mask = (
                np.ones(len(pair_target), dtype=bool)
                if group == "all"
                else pair_target == (0 if group == "benign" else 1)
            )
            group_indices = np.flatnonzero(mask)
            s_values: list[float] = []
            c_values: list[float] = []
            for replicate in range(BOOTSTRAP_ITERATIONS):
                sampled = group_indices[
                    rng.integers(0, len(group_indices), size=len(group_indices))
                ]
                denominator = float(paired_variance[sampled].mean())
                sampling_numerator = float(pair_full_u2[level][sampled].mean())
                sampling_ratio = sampling_numerator / denominator
                s_values.append(sampling_ratio)
                distribution_rows.append(
                    {
                        "contour_sigma": level,
                        "lesion_group": group,
                        "interval_type": "sampling_only",
                        "replicate": replicate + 1,
                        "propagated_variance": sampling_numerator,
                        "observed_variance": denominator,
                        "ratio": sampling_ratio,
                    }
                )

                batch_indices = rng.integers(
                    0, JOINT_BATCHES, size=len(group_indices)
                )
                combined_numerator = float(
                    pair_batch_u2[level][sampled, batch_indices].mean()
                )
                combined_ratio = combined_numerator / denominator
                c_values.append(combined_ratio)
                distribution_rows.append(
                    {
                        "contour_sigma": level,
                        "lesion_group": group,
                        "interval_type": "mc_plus_sampling",
                        "replicate": replicate + 1,
                        "propagated_variance": combined_numerator,
                        "observed_variance": denominator,
                        "ratio": combined_ratio,
                    }
                )

            point = next(
                float(row["ratio"])
                for row in point_rows
                if float(row["contour_sigma"]) == level
                and row["lesion_group"] == group
            )
            for interval_type, values in (
                ("sampling_only", np.asarray(s_values, dtype=float)),
                ("mc_plus_sampling", np.asarray(c_values, dtype=float)),
            ):
                low, high = percentile_interval(values)
                interval_rows.append(
                    {
                        "contour_sigma": level,
                        "lesion_group": group,
                        "interval_type": interval_type,
                        "point_estimate": point,
                        "bootstrap_mean": float(values.mean()),
                        "bootstrap_sd": float(values.std(ddof=1)),
                        "low_95": low,
                        "high_95": high,
                        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    }
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory appeared during analysis: {OUTPUT}")
    OUTPUT.mkdir()

    point_frame = pd.DataFrame(point_rows)
    interval_frame = pd.DataFrame(interval_rows)
    distribution_frame = pd.DataFrame(distribution_rows)
    point_frame.to_csv(OUTPUT / "point_estimates.csv", index=False, lineterminator="\n")
    interval_frame.to_csv(
        OUTPUT / "bootstrap_intervals.csv", index=False, lineterminator="\n"
    )
    distribution_frame.to_csv(
        OUTPUT / "bootstrap_distribution.csv", index=False, lineterminator="\n"
    )

    output_hashes = {
        path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()
    }
    summary = {
        "analysis": "patient_bootstrap_of_propagated_observed_variance_ratio",
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "contour_levels": list(CONTOUR_LEVELS),
        "paired_lesions": len(pairs),
        "paired_images": int(len(pair_images)),
        "formula": {
            "numerator": "mean over resampled lesions of the mean image-level variance across 50 joint draws",
            "denominator": "mean paired logit variance in the same resampled lesion set",
            "sampling_only": "full 50-draw image variances are held fixed while lesions are resampled",
            "mc_plus_sampling": "one of ten five-draw batch variances is sampled with replacement for each resampled lesion; both images of a lesion use the same batch index",
            "interval": "2.5th and 97.5th percentiles of the bootstrap distribution",
        },
        "input_hashes": {
            "selected_cases.csv": sha256(SELECTED_CASES),
            "included_lesion_pairs.csv": sha256(PAIR_FILE),
            "paired_locked_probabilities.csv": sha256(PAIRED_SCORES),
            "mc_convergence_summary.json": sha256(REFERENCE_SUMMARY),
            "case_hash_manifest_sha256": hashlib.sha256(
                case_manifest_text.encode("utf-8")
            ).hexdigest(),
        },
        "reproduction_checks": point_frame.to_dict(orient="records"),
        "output_hashes": output_hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
