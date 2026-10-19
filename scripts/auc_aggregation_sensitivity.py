"""Sensitivity of locked external AUC estimates to patient/group aggregation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from metrology_paths import DATA_ROOT, OUTPUT_ROOT

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
RESULTS = DATA_ROOT / "results"
INPUTS = {
    "busbra": RESULTS / "v04_locked_external" / "busbra_locked_predictions.csv",
    "breast": RESULTS / "v04_locked_external" / "breast_locked_predictions.csv",
}
OUTPUT = OUTPUT_ROOT / "metrology_statistics" / "auc_aggregation"
SEED = 20261018
BOOTSTRAP_ITERATIONS = 2000
METHODS = {
    "SCALE-BUS": "probability_advanced76_logistic_ensemble",
    "FRE": "probability_frozen98_logistic_ensemble",
    "SCALE-BUS-X": "probability_frozen98_shape_logistic_ensemble",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    required = {"image", "patient_id", "target", *METHODS.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    if frame[["image", "patient_id"]].duplicated().any():
        raise RuntimeError("Duplicate image/patient rows")
    patients = list(pd.unique(frame["patient_id"].astype(str)))
    patient_index = {patient: index for index, patient in enumerate(patients)}
    frame = frame.copy()
    frame["_patient_index"] = frame["patient_id"].astype(str).map(patient_index)
    return frame, len(patients)


def patient_observations(frame: pd.DataFrame, patient_indices: list[np.ndarray], draw_positions: np.ndarray) -> pd.DataFrame:
    rows = []
    for position in draw_positions:
        indices = patient_indices[int(position)]
        patient_slice = frame.iloc[indices]
        rows.append(
            {
                "target": int(patient_slice["target"].iloc[0]),
                **{name: float(patient_slice[column].mean()) for name, column in METHODS.items()},
            }
        )
    return pd.DataFrame(rows)


def metric_values(
    frame: pd.DataFrame,
    indices: np.ndarray,
    weights: np.ndarray | None = None,
    patient_rows: pd.DataFrame | None = None,
) -> dict[str, float]:
    y = frame["target"].to_numpy(dtype=int)[indices]
    if np.unique(y).size < 2:
        return {}
    values = {}
    for name, column in METHODS.items():
        probability = frame[column].to_numpy(dtype=float)[indices]
        values[f"{name}:group_equal"] = float(roc_auc_score(y, probability, sample_weight=weights))
        values[f"{name}:image"] = float(roc_auc_score(y, probability))
    if patient_rows is not None and patient_rows["target"].nunique() >= 2:
        for name in METHODS:
            values[f"{name}:patient"] = float(roc_auc_score(patient_rows["target"], patient_rows[name]))
    for first, second in (("SCALE-BUS", "FRE"), ("SCALE-BUS-X", "SCALE-BUS")):
        for aggregation in ("group_equal", "image", "patient"):
            key_first = f"{first}:{aggregation}"
            key_second = f"{second}:{aggregation}"
            if key_first in values and key_second in values:
                values[f"{first}-{second}:{aggregation}"] = values[key_first] - values[key_second]
    return values


def bootstrap(frame: pd.DataFrame, patient_indices: list[np.ndarray]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        draw_positions = rng.integers(0, len(patient_indices), size=len(patient_indices))
        pieces = [patient_indices[position] for position in draw_positions]
        indices = np.concatenate(pieces)
        weights = np.concatenate([np.full(len(piece), 1.0 / len(piece)) for piece in pieces])
        patient_rows = patient_observations(frame, patient_indices, draw_positions)
        values = metric_values(frame, indices, weights, patient_rows)
        if values:
            samples.append(values)
    samples_frame = pd.DataFrame(samples)
    intervals = {}
    for column in samples_frame.columns:
        values = samples_frame[column].to_numpy(dtype=float)
        intervals[column] = {
            "mean": float(values.mean()),
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
    return samples_frame, intervals


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory already exists: {OUTPUT}")
    point_rows = []
    interval_rows = []
    distribution_frames = []
    input_hashes = {}
    for dataset in ("busbra", "breast"):
        path = INPUTS[dataset]
        input_hashes[path.name] = sha256(path)
        raw = pd.read_csv(path)
        frame, patient_count = prepare(raw)
        patient_indices = [
            frame.index[frame["_patient_index"] == patient].to_numpy()
            for patient in range(patient_count)
        ]
        all_indices = frame.index.to_numpy()
        group_counts = frame.groupby("patient_id")["patient_id"].transform("size").to_numpy(dtype=float)
        group_weights = 1.0 / group_counts
        patient_rows = patient_observations(frame, patient_indices, np.arange(patient_count))
        point = metric_values(frame, all_indices, group_weights, patient_rows)
        distribution, intervals = bootstrap(frame, patient_indices)
        for metric, value in point.items():
            point_rows.append({"dataset": dataset, "metric": metric, "estimate": value})
        for metric, interval in intervals.items():
            interval_rows.append({"dataset": dataset, "metric": metric, **interval})
        distribution.insert(0, "dataset", dataset)
        distribution_frames.append(distribution)
        print(dataset, json.dumps(point, indent=2), flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory appeared during analysis: {OUTPUT}")
    OUTPUT.mkdir()
    pd.DataFrame(point_rows).to_csv(OUTPUT / "auc_estimates.csv", index=False, lineterminator="\n")
    pd.DataFrame(interval_rows).to_csv(OUTPUT / "bootstrap_intervals.csv", index=False, lineterminator="\n")
    pd.concat(distribution_frames, ignore_index=True).to_csv(OUTPUT / "bootstrap_distribution.csv", index=False, lineterminator="\n")
    formula = "group_equal AUC: roc_auc_score(y, score, sample_weight=1/n_images_in_patient); image AUC: roc_auc_score(y, score); patient AUC: roc_auc_score(patient target, mean score within patient)"
    output_hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    summary = {
        "analysis": "auc_aggregation_sensitivity",
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_unit": "patient/visual group",
        "group_equal_formula": formula,
        "implementation_reference": {
            "existing_weight_function": "scripts/run_busuclm_grouped_cv.py:72-74",
            "existing_metric_call": "scripts/evaluate_v04_locked_external.py:395-398",
        },
        "input_hashes": input_hashes,
        "point_estimates": point_rows,
        "bootstrap_intervals": interval_rows,
        "output_hashes": output_hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "rows": len(point_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
