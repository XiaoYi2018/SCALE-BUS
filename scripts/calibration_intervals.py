"""Patient-bootstrap calibration intervals and local recalibration sensitivity."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from metrology_paths import OUTPUT_ROOT

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

sys.dont_write_bytecode = True
RESULTS = OUTPUT_ROOT
CALIBRATION_DIR = RESULTS / "v07_label_free_calibration"
METRICS_FILE = CALIBRATION_DIR / "calibration_metrics.csv"
LOCKED_METRICS_FILE = CALIBRATION_DIR / "locked_calibration_metrics.csv"
OUTPUT = OUTPUT_ROOT / "metrology_statistics" / "calibration_intervals"
SEED = 20261018
BOOTSTRAP_ITERATIONS = 2000
LOCAL_REPEATS = 200
LOCAL_PATIENT_COUNTS = (50, 100, 200)
VARIANTS = ("a", "b", "c", "d")
SCOPES = ("locked", "lodo")
DATASETS = ("busbra", "breast", "busi", "busuclm")
LOCKED_DATASETS = ("busbra", "breast")
GROUPS = ("advanced76", "frozen98")
METRIC_COLUMNS = ("brier", "log_loss", "ece_10", "calibration_intercept", "calibration_slope")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_path(scope: str, dataset: str, group: str) -> Path:
    return CALIBRATION_DIR / f"{scope}_{dataset}_{group}_predictions.csv"


def patient_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("patient_id", sort=False)["patient_id"].transform("size")
    return (1.0 / counts.to_numpy(dtype=float))


def clipped_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def probability_logit(values: np.ndarray) -> np.ndarray:
    values = clipped_probability(values)
    return np.log(values / (1.0 - values))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not selected.any():
            continue
        current_weight = weights[selected]
        fraction = float(current_weight.sum()) / total
        observed = float(np.average(labels[selected], weights=current_weight))
        predicted = float(np.average(probabilities[selected], weights=current_weight))
        value += fraction * abs(observed - predicted)
    return float(value)


def weighted_quantile_edges(probabilities: np.ndarray, weights: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(probabilities, kind="stable")
    sorted_probability = probabilities[order]
    sorted_weight = weights[order]
    cumulative = np.cumsum(sorted_weight)
    targets = np.arange(1, bins, dtype=float) / bins * cumulative[-1]
    positions = np.searchsorted(cumulative, targets, side="left")
    positions = np.clip(positions, 0, len(sorted_probability) - 1)
    return np.asarray([0.0, *sorted_probability[positions], 1.0], dtype=float)


def equal_frequency_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    bins: int,
) -> float:
    edges = weighted_quantile_edges(probabilities, weights, bins)
    assignments = np.searchsorted(edges[1:-1], probabilities, side="right")
    total = float(weights.sum())
    value = 0.0
    for index in range(bins):
        selected = assignments == index
        if not selected.any():
            continue
        current_weight = weights[selected]
        fraction = float(current_weight.sum()) / total
        observed = float(np.average(labels[selected], weights=current_weight))
        predicted = float(np.average(probabilities[selected], weights=current_weight))
        value += fraction * abs(observed - predicted)
    return float(value)


def fit_platt(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=1e6,
        max_iter=5000,
        random_state=SEED,
        solver="lbfgs",
    )
    model.fit(
        probability_logit(probabilities).reshape(-1, 1),
        labels,
        sample_weight=weights,
    )
    return model


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    clipped = clipped_probability(probabilities)
    model = fit_platt(labels, probabilities, weights)
    return {
        "brier": float(np.average(np.square(probabilities - labels), weights=weights)),
        "log_loss": float(np.average(-labels * np.log(clipped) - (1 - labels) * np.log1p(-clipped), weights=weights)),
        "ece_10": expected_calibration_error(labels, probabilities, weights, 10),
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def patient_index_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    patients = list(pd.unique(frame["patient_id"].astype(str)))
    lookup = []
    for patient in patients:
        lookup.append(frame.index[frame["patient_id"].astype(str) == patient].to_numpy(dtype=int))
    return lookup


def bootstrap_distribution(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_groups: list[np.ndarray],
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for iteration in range(BOOTSTRAP_ITERATIONS):
        positions = rng.integers(0, len(patient_groups), size=len(patient_groups))
        indices = np.concatenate([patient_groups[position] for position in positions])
        weights = np.concatenate([np.full(len(patient_groups[position]), 1.0 / len(patient_groups[position])) for position in positions])
        values = calculate_metrics(labels[indices], probabilities[indices], weights)
        rows.append({"iteration": iteration + 1, **values})
    return pd.DataFrame(rows)


def interval_record(values: pd.Series, valid: int) -> dict[str, float | int]:
    finite = values[np.isfinite(values)].to_numpy(dtype=float)
    return {
        "valid_bootstraps": int(valid),
        "mean": float(finite.mean()),
        "low_95": float(np.quantile(finite, 0.025)),
        "high_95": float(np.quantile(finite, 0.975)),
    }


def cross_fitted_platt(frame: pd.DataFrame, probability_column: str) -> np.ndarray:
    labels = frame["target"].to_numpy(dtype=int)
    probabilities = frame[probability_column].to_numpy(dtype=float)
    groups = frame["patient_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = np.full(len(frame), np.nan, dtype=float)
    split_iterator = splitter.split(np.zeros(len(frame)), labels, groups)
    for train, test in split_iterator:
        weights = patient_equal_weights(frame.iloc[train])
        model = fit_platt(labels[train], probabilities[train], weights)
        predictions[test] = model.predict_proba(probability_logit(probabilities[test]).reshape(-1, 1))[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Cross-fitted Platt predictions are incomplete")
    return predictions


def fixed_probability_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_groups: list[np.ndarray],
) -> tuple[dict[str, dict[str, float | int]], pd.DataFrame]:
    distribution = bootstrap_distribution(labels, probabilities, patient_groups)
    intervals = {
        column: interval_record(distribution[column], len(distribution))
        for column in METRIC_COLUMNS
    }
    return intervals, distribution


def local_label_sensitivity(frame: pd.DataFrame, dataset: str) -> tuple[list[dict], list[dict]]:
    labels = frame["target"].to_numpy(dtype=int)
    probabilities = frame["p_a"].to_numpy(dtype=float)
    patient_groups = patient_index_groups(frame)
    rows = []
    summaries = []
    for patient_count in LOCAL_PATIENT_COUNTS:
        rng = np.random.default_rng(SEED + patient_count)
        for repeat in range(LOCAL_REPEATS):
            order = rng.permutation(len(patient_groups))
            train_positions = order[:patient_count]
            test_positions = order[patient_count:]
            if len(test_positions) == 0:
                raise RuntimeError(f"{dataset}: no patients remain for local-label evaluation")
            train = np.concatenate([patient_groups[position] for position in train_positions])
            test = np.concatenate([patient_groups[position] for position in test_positions])
            train_weights = np.concatenate([np.full(len(patient_groups[position]), 1.0 / len(patient_groups[position])) for position in train_positions])
            test_weights = np.concatenate([np.full(len(patient_groups[position]), 1.0 / len(patient_groups[position])) for position in test_positions])
            try:
                model = fit_platt(labels[train], probabilities[train], train_weights)
                recalibrated = model.predict_proba(probability_logit(probabilities[test]).reshape(-1, 1))[:, 1]
                metrics = calculate_metrics(labels[test], recalibrated, test_weights)
            except ValueError as error:
                rows.append({
                    "dataset": dataset,
                    "training_patients": patient_count,
                    "repeat": repeat + 1,
                    "valid": False,
                    "reason": str(error),
                    **{column: float("nan") for column in METRIC_COLUMNS},
                })
                continue
            rows.append({
                "dataset": dataset,
                "training_patients": patient_count,
                "repeat": repeat + 1,
                "valid": True,
                "reason": "",
                **metrics,
            })
        current = pd.DataFrame([row for row in rows if row["training_patients"] == patient_count and row["valid"]])
        for metric in ("ece_10", "calibration_slope"):
            values = current[metric].to_numpy(dtype=float)
            summaries.append({
                "dataset": dataset,
                "training_patients": patient_count,
                "metric": metric,
                "attempted_repeats": LOCAL_REPEATS,
                "valid_repeats": int(len(values)),
                "median": float(np.median(values)),
                "low_95": float(np.quantile(values, 0.025)),
                "high_95": float(np.quantile(values, 0.975)),
            })
    return rows, summaries


def validate_expected_metrics(expected: pd.DataFrame) -> pd.DataFrame:
    required = {"scope", "dataset", "group", "variant", *METRIC_COLUMNS}
    missing = sorted(required - set(expected.columns))
    if missing:
        raise RuntimeError(f"Calibration metrics are missing columns: {missing}")
    if len(expected) != 48:
        raise RuntimeError(f"Unexpected calibration metric row count: {len(expected)}")
    keys = expected[["scope", "dataset", "group", "variant"]]
    if keys.duplicated().any():
        raise RuntimeError("Duplicate calibration metric keys")
    return expected


def validate_locked_duplicate(main: pd.DataFrame, locked: pd.DataFrame) -> float:
    columns = ["scope", "dataset", "group", "variant", *METRIC_COLUMNS]
    merged = locked[columns].merge(
        main.loc[main["scope"] == "locked", columns],
        on=["scope", "dataset", "group", "variant"],
        how="outer",
        suffixes=("_locked", "_main"),
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any() or len(merged) != 16:
        raise RuntimeError("Locked calibration metrics do not match the main inventory")
    differences = []
    for column in METRIC_COLUMNS:
        differences.append(np.abs(merged[f"{column}_locked"] - merged[f"{column}_main"]).to_numpy(dtype=float))
    return float(np.max(np.concatenate(differences)))


def main() -> int:
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory already exists: {OUTPUT}")
    expected = validate_expected_metrics(pd.read_csv(METRICS_FILE))
    locked_expected = pd.read_csv(LOCKED_METRICS_FILE)
    locked_duplicate_max_difference = validate_locked_duplicate(expected, locked_expected)

    input_paths = [METRICS_FILE, LOCKED_METRICS_FILE]
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    for scope in SCOPES:
        datasets = LOCKED_DATASETS if scope == "locked" else DATASETS
        for dataset in datasets:
            for group in GROUPS:
                path = probability_path(scope, dataset, group)
                input_paths.append(path)
                frame = pd.read_csv(path)
                required = {"image", "patient_id", "target", *[f"p_{variant}" for variant in VARIANTS]}
                missing = sorted(required - set(frame.columns))
                if missing:
                    raise RuntimeError(f"{path.name} is missing columns: {missing}")
                if frame["image"].duplicated().any():
                    raise RuntimeError(f"{path.name} has duplicate images")
                frames[(scope, dataset, group)] = frame.reset_index(drop=True)
    if len(frames) != 12:
        raise RuntimeError(f"Unexpected prediction-file count: {len(frames)}")

    point_rows = []
    sensitivity_rows = []
    reproduction_rows = []
    bootstrap_frames = []
    interval_rows = []
    raw_intervals: dict[tuple[str, str, str], dict[str, dict[str, float | int]]] = {}

    for scope, dataset, group in frames:
        frame = frames[(scope, dataset, group)]
        labels = frame["target"].to_numpy(dtype=int)
        patient_groups = patient_index_groups(frame)
        weights = patient_equal_weights(frame)
        for variant in VARIANTS:
            probability_column = f"p_{variant}"
            probabilities = frame[probability_column].to_numpy(dtype=float)
            metrics = calculate_metrics(labels, probabilities, weights)
            point_rows.append({
                "scope": scope,
                "dataset": dataset,
                "group": group,
                "variant": variant,
                "n_images": len(frame),
                "n_patients": len(patient_groups),
                **metrics,
            })
            sensitivity_rows.append({
                "scope": scope,
                "dataset": dataset,
                "group": group,
                "variant": variant,
                "ece_10_equal_width": expected_calibration_error(labels, probabilities, weights, 10),
                "ece_15_equal_width": expected_calibration_error(labels, probabilities, weights, 15),
                "ece_10_equal_frequency": equal_frequency_calibration_error(labels, probabilities, weights, 10),
            })
            expected_row = expected[
                (expected["scope"] == scope)
                & (expected["dataset"] == dataset)
                & (expected["group"] == group)
                & (expected["variant"] == variant)
            ]
            if len(expected_row) != 1:
                raise RuntimeError(f"Expected one metric row for {scope}/{dataset}/{group}/{variant}")
            for metric in METRIC_COLUMNS:
                reported = float(expected_row.iloc[0][metric])
                reproduction_rows.append({
                    "scope": scope,
                    "dataset": dataset,
                    "group": group,
                    "variant": variant,
                    "metric": metric,
                    "reported": reported,
                    "recomputed": metrics[metric],
                    "absolute_difference": abs(reported - metrics[metric]),
                })

            distribution = bootstrap_distribution(labels, probabilities, patient_groups)
            distribution.insert(0, "variant", variant)
            distribution.insert(0, "group", group)
            distribution.insert(0, "dataset", dataset)
            distribution.insert(0, "scope", scope)
            bootstrap_frames.append(distribution)
            intervals = {
                metric: interval_record(distribution[metric], len(distribution))
                for metric in METRIC_COLUMNS
            }
            for metric, interval in intervals.items():
                interval_rows.append({
                    "scope": scope,
                    "dataset": dataset,
                    "group": group,
                    "variant": variant,
                    "metric": metric,
                    "point_estimate": metrics[metric],
                    **interval,
                })
            if scope == "locked":
                raw_intervals[(dataset, group, variant)] = intervals
        print(f"completed {scope}/{dataset}/{group}", flush=True)

    recalibration_rows = []
    recalibration_distribution_frames = []
    for dataset in LOCKED_DATASETS:
        frame = frames[("locked", dataset, "advanced76")]
        labels = frame["target"].to_numpy(dtype=int)
        patient_groups = patient_index_groups(frame)
        weights = patient_equal_weights(frame)
        cross_fitted = cross_fitted_platt(frame, "p_a")
        cross_metrics = calculate_metrics(labels, cross_fitted, weights)
        cross_intervals, cross_distribution = fixed_probability_bootstrap(labels, cross_fitted, patient_groups)
        cross_distribution.insert(0, "variant", "cross_fitted_platt")
        cross_distribution.insert(0, "dataset", dataset)
        recalibration_distribution_frames.append(cross_distribution)

        for variant, label in (("raw", "raw"), ("cross_fitted_platt", "cross_fitted_platt"), ("em", "em")):
            if variant == "raw":
                point_row = next(row for row in point_rows if row["scope"] == "locked" and row["dataset"] == dataset and row["group"] == "advanced76" and row["variant"] == "a")
                metrics = {metric: point_row[metric] for metric in METRIC_COLUMNS}
                intervals = raw_intervals[(dataset, "advanced76", "a")]
            elif variant == "em":
                point_row = next(row for row in point_rows if row["scope"] == "locked" and row["dataset"] == dataset and row["group"] == "advanced76" and row["variant"] == "d")
                metrics = {metric: point_row[metric] for metric in METRIC_COLUMNS}
                intervals = raw_intervals[(dataset, "advanced76", "d")]
            else:
                metrics = cross_metrics
                intervals = cross_intervals
            for metric in METRIC_COLUMNS:
                recalibration_rows.append({
                    "dataset": dataset,
                    "score": label,
                    "metric": metric,
                    "point_estimate": metrics[metric],
                    **intervals[metric],
                })
        print(f"completed locked recalibration/{dataset}", flush=True)

    local_distribution_rows = []
    local_summary_rows = []
    for dataset in LOCKED_DATASETS:
        rows, summaries = local_label_sensitivity(frames[("locked", dataset, "advanced76")], dataset)
        local_distribution_rows.extend(rows)
        local_summary_rows.extend(summaries)
        print(f"completed local-label sensitivity/{dataset}", flush=True)

    output_hashes = {}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory appeared during analysis: {OUTPUT}")
    OUTPUT.mkdir()

    point_frame = pd.DataFrame(point_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    reproduction_frame = pd.DataFrame(reproduction_rows)
    bootstrap_frame = pd.concat(bootstrap_frames, ignore_index=True)
    interval_frame = pd.DataFrame(interval_rows)
    recalibration_frame = pd.DataFrame(recalibration_rows)
    local_distribution_frame = pd.DataFrame(local_distribution_rows)
    local_summary_frame = pd.DataFrame(local_summary_rows)

    outputs = {
        "calibration_estimates.csv": point_frame,
        "ece_sensitivity.csv": sensitivity_frame,
        "bootstrap_intervals.csv": interval_frame,
        "bootstrap_distribution.csv": bootstrap_frame,
        "locked_recalibration.csv": recalibration_frame,
        "local_label_requirements.csv": local_summary_frame,
        "local_label_distribution.csv": local_distribution_frame,
        "reproduction_checks.csv": reproduction_frame,
    }
    for name, frame in outputs.items():
        path = OUTPUT / name
        frame.to_csv(path, index=False, lineterminator="\n")
        output_hashes[name] = sha256(path)

    input_hashes = {str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/"): sha256(path) for path in input_paths}
    summary = {
        "analysis": "calibration_intervals_and_local_recalibration",
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_unit": "patient; duplicate each drawn patient with unit multiplicity and re-normalize image weights within each copy",
        "patient_equal_weight": "1 / images_in_patient",
        "ece_equal_width": "10 bins by probability width, right edge included in the final bin",
        "ece_equal_frequency": "10 weighted-quantile bins using patient-equal image weights",
        "calibration_model": {"penalty": "l2", "C": 1000000.0, "solver": "lbfgs", "max_iter": 5000},
        "locked_recalibration_method": "5-fold StratifiedGroupKFold by patient; Platt intercept and slope fitted on logit(raw score)",
        "locked_recalibration_bootstrap": "fixed out-of-fold probabilities, resampling patients",
        "local_label_sampling": "without replacement within each repeat; Platt fitted on sampled patients and evaluated on all remaining patients",
        "local_repeat_count": LOCAL_REPEATS,
        "local_patient_counts": list(LOCAL_PATIENT_COUNTS),
        "input_hashes": input_hashes,
        "locked_duplicate_max_absolute_difference": locked_duplicate_max_difference,
        "point_estimates": point_frame.to_dict(orient="records"),
        "ece_sensitivity": sensitivity_frame.to_dict(orient="records"),
        "bootstrap_intervals": interval_frame.to_dict(orient="records"),
        "locked_recalibration": recalibration_frame.to_dict(orient="records"),
        "local_label_requirements": local_summary_frame.to_dict(orient="records"),
        "reproduction_max_absolute_difference": float(reproduction_frame["absolute_difference"].max()),
        "output_hashes": output_hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": float(time.time() - started),
    }
    summary_path = OUTPUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "point_rows": len(point_frame), "intervals": len(interval_frame)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
