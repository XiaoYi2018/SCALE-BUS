"""Compare patients represented by two views with the remaining BUS-BRA cohort."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from metrology_paths import DATA_ROOT, OUTPUT_ROOT

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

RESULTS = OUTPUT_ROOT
METADATA = Path(os.environ.get(
    "SCALE_BUS_BUSBRA_METADATA",
    str(DATA_ROOT / "data/raw/BUS-BRA_zenodo_8231412/BUSBRA/BUSBRA/bus_data.csv"),
))
PAIR_FILE = (
    RESULTS
    / "v07_interview_reproducibility"
    / "included_lesion_pairs.csv"
)
GEOMETRY = RESULTS / "v07_reference_calibration" / "real_lesion_geometry.csv"
OUTPUT = OUTPUT_ROOT / "metrology_statistics" / "selection_audit"

GROUPS = ("included_two_view", "not_included", "all")
COMPARISONS = (
    ("included_two_view", "not_included"),
    ("included_two_view", "all"),
    ("not_included", "all"),
)
CATEGORICAL = ("pathology", "BIRADS", "device")
CONTINUOUS = ("mask_equivalent_radius",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def continuous_smd(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    pooled = np.sqrt(
        (np.var(first, ddof=1) + np.var(second, ddof=1)) / 2.0
    )
    if pooled == 0:
        return float("nan")
    return float(abs(first.mean() - second.mean()) / pooled)


def categorical_smd(
    first: pd.Series, second: pd.Series, category: str
) -> float:
    first_proportion = float((first.astype(str) == str(category)).mean())
    second_proportion = float((second.astype(str) == str(category)).mean())
    denominator = np.sqrt(
        (
            first_proportion * (1.0 - first_proportion)
            + second_proportion * (1.0 - second_proportion)
        )
        / 2.0
    )
    if denominator == 0:
        return 0.0
    return float(abs(first_proportion - second_proportion) / denominator)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory already exists: {OUTPUT}")

    metadata = pd.read_csv(METADATA)
    pairs = pd.read_csv(PAIR_FILE)
    geometry = pd.read_csv(GEOMETRY)
    geometry = geometry[geometry["dataset"] == "busbra"][
        ["image", "mask_equivalent_radius"]
    ].copy()

    required_metadata = {
        "ID",
        "Case",
        "Pathology",
        "BIRADS",
        "Device",
    }
    if not required_metadata.issubset(metadata.columns):
        raise RuntimeError("BUS-BRA metadata is missing required columns")
    if len(metadata) != 1875 or not metadata["ID"].is_unique:
        raise RuntimeError("Unexpected BUS-BRA image inventory")
    if len(pairs) != 810 or not pairs["Case"].is_unique:
        raise RuntimeError("Expected 810 included lesion pairs")
    if len(geometry) != 1875 or not geometry["image"].is_unique:
        raise RuntimeError("Unexpected BUS-BRA geometry inventory")

    included_cases = set(pairs["Case"].astype(int))
    if len(included_cases) != 810:
        raise RuntimeError("Included case identifiers are not unique")

    metadata["geometry_image"] = metadata["ID"].astype(str) + ".png"
    metadata = metadata.merge(
        geometry,
        left_on="geometry_image",
        right_on="image",
        validate="one_to_one",
        how="left",
    )
    if metadata["mask_equivalent_radius"].isna().any():
        raise RuntimeError("Missing normalized geometry for a BUS-BRA image")

    consistent_columns = ("Pathology", "Device")
    for column in consistent_columns:
        if (metadata.groupby("Case")[column].nunique(dropna=False) != 1).any():
            raise RuntimeError(f"Patient-level metadata disagree in {column}")

    image_counts = metadata.groupby("Case").size()
    if not set(image_counts.unique()).issubset({1, 2}):
        raise RuntimeError("Unexpected number of views per BUS-BRA patient")
    if int((image_counts == 2).sum()) != 811:
        raise RuntimeError("Expected 811 patients with two-view candidates")

    metadata = metadata.sort_values(["Case", "ID"])
    patient_frame = (
        metadata.groupby("Case", as_index=False)
        .agg(
            pathology=("Pathology", "first"),
            BIRADS=("BIRADS", "first"),
            device=("Device", "first"),
            mask_equivalent_radius=("mask_equivalent_radius", "mean"),
            view_count=("ID", "size"),
        )
        .reset_index(drop=True)
    )
    patient_frame["pathology"] = patient_frame["pathology"].astype(str).str.lower()
    patient_frame["BIRADS"] = patient_frame["BIRADS"].astype(int).astype(str)
    patient_frame["group"] = np.where(
        patient_frame["Case"].isin(included_cases),
        "included_two_view",
        "not_included",
    )
    if len(patient_frame) != 1064:
        raise RuntimeError("Expected 1064 BUS-BRA patients")

    group_frames = {
        "included_two_view": patient_frame[
            patient_frame["group"] == "included_two_view"
        ],
        "not_included": patient_frame[patient_frame["group"] == "not_included"],
        "all": patient_frame,
    }
    if {name: len(frame) for name, frame in group_frames.items()} != {
        "included_two_view": 810,
        "not_included": 254,
        "all": 1064,
    }:
        raise RuntimeError("Unexpected group sizes")

    summary_rows: list[dict[str, float | int | str]] = []
    for group in GROUPS:
        frame = group_frames[group]
        for variable in CONTINUOUS:
            values = frame[variable].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "group": group,
                    "variable": variable,
                    "level": "continuous",
                    "patients": len(frame),
                    "count": len(values),
                    "proportion": np.nan,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
        for variable in CATEGORICAL:
            counts = frame[variable].astype(str).value_counts()
            for level in sorted(counts.index):
                summary_rows.append(
                    {
                        "group": group,
                        "variable": variable,
                        "level": level,
                        "patients": len(frame),
                        "count": int(counts[level]),
                        "proportion": float(counts[level] / len(frame)),
                        "mean": np.nan,
                        "sd": np.nan,
                        "median": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                    }
                )

    comparison_rows: list[dict[str, float | int | str | bool]] = []
    for first_group, second_group in COMPARISONS:
        first = group_frames[first_group]
        second = group_frames[second_group]
        comparison = f"{first_group}_vs_{second_group}"
        for variable in CONTINUOUS:
            smd = continuous_smd(first[variable], second[variable])
            comparison_rows.append(
                {
                    "comparison": comparison,
                    "first_group": first_group,
                    "second_group": second_group,
                    "variable": variable,
                    "level": "continuous",
                    "smd": smd,
                    "variable_max_smd": smd,
                    "level_abs_smd_gt_0_1": bool(abs(smd) > 0.1),
                    "variable_abs_smd_gt_0_1": bool(abs(smd) > 0.1),
                }
            )
        for variable in CATEGORICAL:
            levels = sorted(
                set(first[variable].astype(str)) | set(second[variable].astype(str))
            )
            level_smds = {
                level: categorical_smd(first[variable], second[variable], level)
                for level in levels
            }
            maximum = max(level_smds.values())
            for level, smd in level_smds.items():
                comparison_rows.append(
                    {
                        "comparison": comparison,
                        "first_group": first_group,
                        "second_group": second_group,
                        "variable": variable,
                        "level": level,
                        "smd": smd,
                        "variable_max_smd": maximum,
                        "level_abs_smd_gt_0_1": bool(abs(smd) > 0.1),
                        "variable_abs_smd_gt_0_1": bool(abs(maximum) > 0.1),
                    }
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory appeared during analysis: {OUTPUT}")
    OUTPUT.mkdir()

    summary_frame = pd.DataFrame(summary_rows)
    smd_frame = pd.DataFrame(comparison_rows)
    summary_frame.to_csv(
        OUTPUT / "group_summary.csv", index=False, lineterminator="\n"
    )
    smd_frame.to_csv(OUTPUT / "smd.csv", index=False, lineterminator="\n")

    output_hashes = {
        path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()
    }
    flagged = smd_frame.loc[
        smd_frame["level_abs_smd_gt_0_1"],
        ["comparison", "variable", "level", "smd"],
    ].to_dict(orient="records")
    flagged_variables = (
        smd_frame.loc[
            smd_frame["variable_abs_smd_gt_0_1"],
            ["comparison", "variable", "variable_max_smd"],
        ]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    summary = {
        "analysis": "two_view_selection_audit",
        "unit": "patient",
        "continuous_geometry": "mean normalized-grid mask equivalent radius across available views",
        "categorical_metadata": "first lexicographic image identifier per patient; all patients except one excluded candidate have consistent BI-RADS values across views",
        "group_sizes": {group: len(frame) for group, frame in group_frames.items()},
        "included_cases": len(included_cases),
        "two_view_candidates": int((image_counts == 2).sum()),
        "input_hashes": {
            "bus_data.csv": sha256(METADATA),
            "included_lesion_pairs.csv": sha256(PAIR_FILE),
            "real_lesion_geometry.csv": sha256(GEOMETRY),
        },
        "maximum_smd": smd_frame.groupby(["comparison", "variable"])[
            "variable_max_smd"
        ]
        .max()
        .reset_index()
        .to_dict(orient="records"),
        "flagged_abs_smd_gt_0_1": flagged,
        "flagged_variables_abs_max_smd_gt_0_1": flagged_variables,
        "output_hashes": output_hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
