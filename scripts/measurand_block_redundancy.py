"""Redundancy analysis of the predefined generalized-dimension, wavelet, and boundary blocks."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from metrology_paths import DATA_ROOT, OUTPUT_ROOT, MODEL_CACHE

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
ROOT = DATA_ROOT
RESULTS = ROOT / "results"
OUTPUT = OUTPUT_ROOT / "metrology_statistics" / "block_redundancy"
SEED = 20261018
ALPHAS = (0.1, 1.0, 10.0, 100.0)
OUTER_SPLITS = 5
INNER_SPLITS = 4

FEATURE_INPUTS = {
    "BUS-UCLM": {
        "path": RESULTS / "v04_advanced_features" / "features_busuclm_advanced.csv",
        "group_column": "patient_id",
    },
    "BUSI": {
        "path": RESULTS / "v04_advanced_features" / "features_busi_advanced.csv",
        "group_column": "cv_group_id",
    },
    "BUS-BRA": {
        "path": RESULTS / "v04_external_features" / "features_busbra_advanced.csv",
        "group_column": "patient_id",
    },
    "BrEaST": {
        "path": RESULTS / "v04_external_features" / "features_breast_advanced.csv",
        "group_column": "patient_id",
    },
}



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_columns(columns: list[str]) -> dict[str, list[str]]:
    return {
        "GF": [column for column in columns if column.startswith("advanced_mf_")],
        "W": [column for column in columns if column.startswith("advanced_wavelet_")],
        "S": [column for column in columns if column.startswith("advanced_shape_")],
    }


def support_zone(column: str, block: str) -> str:
    if block == "S":
        return "lesion"
    for zone in ("lesion", "inner", "outer"):
        if f"_{zone}_" in column:
            return zone
    raise RuntimeError(f"Cannot assign support zone to {column}")


def validate_feature_reference(blocks: dict[str, list[str]]) -> tuple[list[str], dict[str, object]]:
    reference_columns = joblib.load(MODEL_CACHE)["columns"]["advanced76"]
    reference_blocks = block_columns(reference_columns)
    expected_counts = {"GF": 30, "W": 36, "S": 10}
    reference_names = {
        "GF": reference_blocks["GF"],
        "W": reference_blocks["W"],
        "S": reference_blocks["S"],
    }
    for block, count in expected_counts.items():
        if len(blocks[block]) != count or len(reference_names[block]) != count:
            raise RuntimeError(f"Unexpected {block} feature count")
        if blocks[block] != reference_names[block]:
            raise RuntimeError(f"{block} feature names or order differ from the locked model reference")
    columns = blocks["GF"] + blocks["W"] + blocks["S"]
    if len(columns) != 76 or len(set(columns)) != 76:
        raise RuntimeError("Expected 76 unique SCALE-BUS feature columns")
    details = {
        "reference_path": str(MODEL_CACHE.relative_to(OUTPUT_ROOT)).replace("\\", "/") if MODEL_CACHE.is_relative_to(OUTPUT_ROOT) else MODEL_CACHE.name,
        "reference_sha256": sha256(MODEL_CACHE),
        "column_order_matches_reference_exactly": True,
        "block_counts": expected_counts,
        "total_unique_columns": 76,
    }
    return columns, details


def load_feature_frame(path: Path, columns: list[str], group_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    missing = sorted(set(columns + [group_column]) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path.name} is missing columns: {missing}")
    groups = frame[group_column].astype(str).to_numpy()
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path.name} contains non-finite model features")
    if np.any(values.std(axis=0, ddof=0) == 0):
        constant = [column for column, std in zip(columns, values.std(axis=0, ddof=0)) if std == 0]
        raise RuntimeError(f"{path.name} contains constant model features: {constant}")
    return frame[columns].copy(), groups


def correlation_summary(matrix: pd.DataFrame, first: list[str], second: list[str], include_diagonal: bool) -> dict[str, float | int]:
    values = np.abs(matrix.loc[first, second].to_numpy(dtype=float))
    if not include_diagonal:
        values = values[~np.eye(len(first), dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("Empty correlation pair set")
    return {
        "n_pairs": int(values.size),
        "median_abs_spearman": float(np.median(values)),
        "q90_abs_spearman": float(np.quantile(values, 0.90)),
        "max_abs_spearman": float(values.max()),
    }


def rank_summaries(
    cohort_matrices: dict[str, pd.DataFrame],
    blocks: dict[str, list[str]],
    zones: dict[str, dict[str, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    relations = [
        ("GF", "W", "between"),
        ("GF", "S", "between"),
        ("W", "S", "between"),
        ("GF", "GF", "within"),
        ("W", "W", "within"),
        ("S", "S", "within"),
    ]
    for cohort, matrix in cohort_matrices.items():
        for first, second, relation_type in relations:
            summary = correlation_summary(
                matrix,
                blocks[first],
                blocks[second],
                include_diagonal=(first != second),
            )
            rows.append(
                {
                    "cohort": cohort,
                    "zone": "all",
                    "relation": f"{first}-{second}",
                    "relation_type": relation_type,
                    **summary,
                }
            )
    full = pd.DataFrame(rows)
    cohort_medians = []
    for (relation, relation_type), group in full.groupby(["relation", "relation_type"], sort=False):
        cohort_medians.append(
            {
                "cohort": "four-cohort median",
                "zone": "all",
                "relation": relation,
                "relation_type": relation_type,
                "n_pairs": int(round(group["n_pairs"].median())),
                "median_abs_spearman": float(group["median_abs_spearman"].median()),
                "q90_abs_spearman": float(group["q90_abs_spearman"].median()),
                "max_abs_spearman": float(group["max_abs_spearman"].median()),
            }
        )
    full = pd.concat([full, pd.DataFrame(cohort_medians)], ignore_index=True)

    zone_rows = []
    for cohort, matrix in cohort_matrices.items():
        for zone in ("lesion", "inner", "outer"):
            available = [block for block in ("GF", "W", "S") if zones[zone][block]]
            for first_index, first in enumerate(available):
                for second in available[first_index + 1 :]:
                    summary = correlation_summary(
                        matrix,
                        zones[zone][first],
                        zones[zone][second],
                        include_diagonal=True,
                    )
                    zone_rows.append(
                        {
                            "cohort": cohort,
                            "zone": zone,
                            "relation": f"{first}-{second}",
                            "n_pairs": summary["n_pairs"],
                            "median_abs_spearman": summary["median_abs_spearman"],
                            "q90_abs_spearman": summary["q90_abs_spearman"],
                            "max_abs_spearman": summary["max_abs_spearman"],
                        }
                    )
    zone_frame = pd.DataFrame(zone_rows)
    zone_medians = []
    for (zone, relation), group in zone_frame.groupby(["zone", "relation"], sort=False):
        zone_medians.append(
            {
                "cohort": "four-cohort median",
                "zone": zone,
                "relation": relation,
                "n_pairs": int(round(group["n_pairs"].median())),
                "median_abs_spearman": float(group["median_abs_spearman"].median()),
                "q90_abs_spearman": float(group["q90_abs_spearman"].median()),
                "max_abs_spearman": float(group["max_abs_spearman"].median()),
            }
        )
    zone_frame = pd.concat([zone_frame, pd.DataFrame(zone_medians)], ignore_index=True)
    return full, zone_frame


def fit_standardized_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_train)
    y_scaled = y_scaler.fit_transform(y_train)
    model = Ridge(alpha=alpha)
    model.fit(x_scaled, y_scaled)
    predicted_scaled = model.predict(x_scaler.transform(x_test))
    return y_scaler.inverse_transform(predicted_scaled)


def select_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    splitter = GroupKFold(n_splits=INNER_SPLITS)
    splits = list(splitter.split(x_train, y_train, groups_train))
    alpha_rows = []
    alpha_scores = {}
    for alpha in ALPHAS:
        predicted = np.full_like(y_train, np.nan, dtype=float)
        for inner_train, inner_validation in splits:
            predicted[inner_validation] = fit_standardized_ridge(
                x_train[inner_train],
                y_train[inner_train],
                x_train[inner_validation],
                alpha,
            )
        target_scores = [
            r2_score(y_train[:, index], predicted[:, index])
            for index in range(y_train.shape[1])
        ]
        mean_score = float(np.mean(target_scores))
        alpha_scores[alpha] = mean_score
        alpha_rows.append(
            {
                "alpha": alpha,
                "mean_inner_r2": mean_score,
                "median_inner_r2": float(np.median(target_scores)),
            }
        )
    selected = max(ALPHAS, key=lambda alpha: alpha_scores[alpha])
    return selected, alpha_rows


def ridge_block(
    cohort: str,
    block: str,
    frame: pd.DataFrame,
    groups: np.ndarray,
    blocks: dict[str, list[str]],
) -> tuple[dict[str, float | int | str], list[dict], list[dict]]:
    predictor_columns = []
    for other in blocks:
        if other != block:
            predictor_columns.extend(blocks[other])
    target_columns = blocks[block]
    x = frame[predictor_columns].to_numpy(dtype=float)
    y = frame[target_columns].to_numpy(dtype=float)
    predicted = np.full_like(y, np.nan, dtype=float)
    outer_splitter = GroupKFold(n_splits=OUTER_SPLITS)
    outer_splits = list(outer_splitter.split(x, y, groups))
    selected_alpha_rows = []
    for fold, (train, test) in enumerate(outer_splits, start=1):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError(f"{cohort}/{block}: outer fold group overlap")
        alpha, inner_rows = select_alpha(x[train], y[train], groups[train])
        predicted[test] = fit_standardized_ridge(x[train], y[train], x[test], alpha)
        for row in inner_rows:
            selected_alpha_rows.append(
                {
                    "cohort": cohort,
                    "target_block": block,
                    "outer_fold": fold,
                    **row,
                    "selected": row["alpha"] == alpha,
                }
            )
    if not np.isfinite(predicted).all():
        raise RuntimeError(f"{cohort}/{block}: incomplete cross-validated predictions")
    target_rows = []
    for index, target in enumerate(target_columns):
        target_rows.append(
            {
                "cohort": cohort,
                "target_block": block,
                "target_feature": target,
                "r2": float(r2_score(y[:, index], predicted[:, index])),
            }
        )
    values = np.asarray([row["r2"] for row in target_rows], dtype=float)
    summary = {
        "cohort": cohort,
        "target_block": block,
        "n_images": int(len(frame)),
        "n_groups": int(len(np.unique(groups))),
        "n_predictors": len(predictor_columns),
        "n_targets": len(target_columns),
        "median_r2": float(np.median(values)),
        "q90_r2": float(np.quantile(values, 0.90)),
        "minimum_r2": float(values.min()),
        "maximum_r2": float(values.max()),
        "selected_alphas": ",".join(
            f"{alpha:g}"
            for alpha in sorted({
                row["alpha"]
                for row in selected_alpha_rows
                if row["selected"]
            })
        ),
    }
    return summary, target_rows, selected_alpha_rows


def effective_dimensions(cohort: str, frame: pd.DataFrame, groups: np.ndarray, columns: list[str], label: str) -> list[dict]:
    scaled = StandardScaler().fit_transform(frame[columns].to_numpy(dtype=float))
    pca = PCA(svd_solver="full")
    pca.fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return [
        {
            "cohort": cohort,
            "feature_set": label,
            "n_images": int(len(frame)),
            "n_groups": int(len(np.unique(groups))),
            "n_features": len(columns),
            "components_90_percent": int(np.searchsorted(cumulative, 0.90) + 1),
            "components_95_percent": int(np.searchsorted(cumulative, 0.95) + 1),
        }
    ]




def main() -> int:
    started = time.perf_counter()
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory already exists: {OUTPUT}")

    first_path = FEATURE_INPUTS["BUS-UCLM"]["path"]
    header = pd.read_csv(first_path, nrows=0)
    blocks = block_columns(list(header.columns))
    columns, reference_details = validate_feature_reference(blocks)
    zones = {
        zone: {
            block: [column for column in blocks[block] if support_zone(column, block) == zone]
            for block in ("GF", "W", "S")
        }
        for zone in ("lesion", "inner", "outer")
    }
    if zones["inner"]["S"] or zones["outer"]["S"]:
        raise RuntimeError("Shape features unexpectedly carry an inner/outer support tag")

    frames = {}
    group_arrays = {}
    input_rows = []
    for cohort, specification in FEATURE_INPUTS.items():
        path = specification["path"]
        frame, groups = load_feature_frame(path, columns, specification["group_column"])
        frames[cohort] = frame
        group_arrays[cohort] = groups
        input_rows.append(
            {
                "input": cohort,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "n_images": len(frame),
                "n_groups": len(np.unique(groups)),
                "n_features": len(columns),
                "sha256": sha256(path),
            }
        )
    input_rows.append(
        {
            "input": "locked feature reference",
            "path": reference_details["reference_path"],
            "n_images": "",
            "n_groups": "",
            "n_features": 76,
            "sha256": reference_details["reference_sha256"],
        }
    )

    matrices = {
        cohort: frame[columns].corr(method="spearman")
        for cohort, frame in frames.items()
    }
    for cohort, matrix in matrices.items():
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{cohort}: non-finite Spearman correlations")
    correlation_frame, zone_frame = rank_summaries(matrices, blocks, zones)

    combined_frame = pd.concat(
        [frames["BUS-UCLM"], frames["BUSI"]],
        ignore_index=True,
    )
    combined_groups = np.concatenate(
        [
            np.asarray([f"busuclm:{group}" for group in group_arrays["BUS-UCLM"]]),
            np.asarray([f"busi:{group}" for group in group_arrays["BUSI"]]),
        ]
    )
    combined = {"BUS-UCLM + BUSI": (combined_frame, combined_groups)}
    for cohort in frames:
        combined[cohort] = (frames[cohort], group_arrays[cohort])

    ridge_rows = []
    ridge_target_rows = []
    alpha_rows = []
    for cohort, (frame, groups) in combined.items():
        for block in ("GF", "W", "S"):
            summary, target_rows, selected_rows = ridge_block(
                cohort,
                block,
                frame,
                groups,
                blocks,
            )
            ridge_rows.append(summary)
            ridge_target_rows.extend(target_rows)
            alpha_rows.extend(selected_rows)
            print(f"completed ridge {cohort}/{block}", flush=True)

    dimension_rows = []
    for cohort, frame in frames.items():
        groups = group_arrays[cohort]
        dimension_rows.extend(effective_dimensions(cohort, frame, groups, blocks["GF"], "GF30"))
        dimension_rows.extend(effective_dimensions(cohort, frame, groups, blocks["W"], "W36"))
        dimension_rows.extend(effective_dimensions(cohort, frame, groups, blocks["S"], "S10"))
        dimension_rows.extend(effective_dimensions(cohort, frame, groups, columns, "all76"))

    input_frame = pd.DataFrame(input_rows)
    correlation_frame = pd.DataFrame(correlation_frame)
    zone_frame = pd.DataFrame(zone_frame)
    ridge_frame = pd.DataFrame(ridge_rows)
    ridge_target_frame = pd.DataFrame(ridge_target_rows)
    alpha_frame = pd.DataFrame(alpha_rows)
    dimension_frame = pd.DataFrame(dimension_rows)
    feature_frame = pd.DataFrame(
        [
            {
                "position": index + 1,
                "block": next(block for block in ("GF", "W", "S") if column in blocks[block]),
                "support_zone": next(block for block in ("GF", "W", "S") if column in blocks[block]) and support_zone(
                    column,
                    next(block for block in ("GF", "W", "S") if column in blocks[block]),
                ),
                "feature": column,
            }
            for index, column in enumerate(columns)
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory appeared during analysis: {OUTPUT}")
    OUTPUT.mkdir()

    input_frame.to_csv(OUTPUT / "input_manifest.csv", index=False, lineterminator="\n")
    feature_frame.to_csv(OUTPUT / "feature_columns.csv", index=False, lineterminator="\n")
    correlation_frame.to_csv(OUTPUT / "rank_correlation_summary.csv", index=False, lineterminator="\n")
    zone_frame.to_csv(OUTPUT / "support_zone_correlation_summary.csv", index=False, lineterminator="\n")
    ridge_frame.to_csv(OUTPUT / "ridge_predictability.csv", index=False, lineterminator="\n")
    ridge_target_frame.to_csv(OUTPUT / "ridge_target_r2.csv", index=False, lineterminator="\n")
    alpha_frame.to_csv(OUTPUT / "ridge_alpha_selection.csv", index=False, lineterminator="\n")
    dimension_frame.to_csv(OUTPUT / "effective_dimensions.csv", index=False, lineterminator="\n")
    matrices["BUS-BRA"].to_csv(OUTPUT / "spearman_busbra.csv", lineterminator="\n")
    np.savez_compressed(
        OUTPUT / "spearman_matrices.npz",
        **{cohort.replace(" ", "_"): matrix.to_numpy(dtype=float) for cohort, matrix in matrices.items()},
        columns=np.asarray(columns),
    )

    output_hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    summary = {
        "analysis": "measurand_block_redundancy",
        "random_seed": SEED,
        "seed_note": "All folds are deterministic grouped folds; the seed is recorded for provenance.",
        "feature_validation": reference_details,
        "feature_columns": columns,
        "block_sizes": {block: len(values) for block, values in blocks.items()},
        "support_zone_policy": "GF and W use explicit lesion/inner/outer tags; the 10 boundary-shape variables are lesion-support measures.",
        "support_zone_counts": {
            zone: {block: len(values) for block, values in block_values.items()}
            for zone, block_values in zones.items()
        },
        "cohorts": {
            cohort: {
                "n_images": int(len(frame)),
                "n_groups": int(len(np.unique(group_arrays[cohort]))),
                "group_column": FEATURE_INPUTS[cohort]["group_column"],
            }
            for cohort, frame in frames.items()
        },
        "rank_correlation": correlation_frame.to_dict(orient="records"),
        "support_zone_correlation": zone_frame.to_dict(orient="records"),
        "ridge": {
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "alphas": list(ALPHAS),
            "standardization": "predictors and target variables standardized within each training fold",
            "alpha_selection": "maximum mean inner grouped-fold R2 across target variables",
            "point_estimates": ridge_frame.to_dict(orient="records"),
        },
        "effective_dimensions": dimension_frame.to_dict(orient="records"),
        "input_manifest": input_frame.to_dict(orient="records"),
        "output_hashes": output_hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "rank_rows": len(correlation_frame),
                "zone_rows": len(zone_frame),
                "ridge_rows": len(ridge_frame),
                "effective_rows": len(dimension_frame),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
