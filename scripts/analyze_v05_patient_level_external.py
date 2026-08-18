from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_busuclm_grouped_cv import SEED  # noqa: E402


INPUT_DIR = ROOT / "results" / "v04_locked_external"
OUTPUT_DIR = ROOT / "results" / "v05_patient_level_external"
BOOTSTRAP_ITERATIONS = 3000
TARGET_FILES = {
    "busbra": INPUT_DIR / "busbra_locked_predictions.csv",
    "breast": INPUT_DIR / "breast_locked_predictions.csv",
}
METHODS = {
    "fre98_logistic": "probability_frozen98_logistic_ensemble",
    "gfwb76_logistic": "probability_advanced76_logistic_ensemble",
    "freb108_logistic": "probability_frozen98_shape_logistic_ensemble",
    "gfwb76_linear_svm": "probability_advanced76_linear_svm_ensemble",
    "resnet18_frozen": "probability_resnet18_inner_only_ensemble",
    "gfwb76_resnet18_early": "probability_advanced76_resnet18_early_ensemble",
}
REFERENCE = "fre98_logistic"
PRIMARY = "gfwb76_logistic"


def aggregate_patients(
    frame: pd.DataFrame,
    probability_column: str,
    aggregation: str,
) -> pd.DataFrame:
    label_counts = frame.groupby("patient_id")["target"].nunique()
    if int(label_counts.max()) != 1:
        inconsistent = label_counts.loc[label_counts > 1].index.tolist()
        raise RuntimeError(
            f"inconsistent labels within patient IDs: {inconsistent[:10]}"
        )
    reducer = "mean" if aggregation == "mean" else "median"
    return (
        frame.groupby("patient_id", as_index=False)
        .agg(
            target=("target", "first"),
            label=("label", "first"),
            images=("image", "size"),
            probability=(probability_column, reducer),
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def bootstrap_metrics(
    wide: pd.DataFrame,
    methods: tuple[str, ...],
    iterations: int,
    seed_offset: int,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, tuple[float, float]],
    dict[str, dict[str, float]],
    int,
]:
    rng = np.random.default_rng(SEED + seed_offset)
    size = len(wide)
    auc_samples = {method: [] for method in methods}
    ap_samples = {method: [] for method in methods}
    delta_samples = {
        method: [] for method in methods if method != REFERENCE
    }
    valid = 0
    for _ in range(iterations):
        indices = rng.integers(0, size, size=size)
        boot = wide.iloc[indices]
        labels = boot["target"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        current_auc: dict[str, float] = {}
        for method in methods:
            probability = boot[f"probability_{method}"].to_numpy()
            auc = float(roc_auc_score(labels, probability))
            ap = float(average_precision_score(labels, probability))
            auc_samples[method].append(auc)
            ap_samples[method].append(ap)
            current_auc[method] = auc
        for method in delta_samples:
            delta_samples[method].append(
                current_auc[method] - current_auc[REFERENCE]
            )
        valid += 1
    if valid == 0:
        raise RuntimeError("no class-valid patient bootstrap was generated")

    def intervals(
        values: dict[str, list[float]],
    ) -> dict[str, tuple[float, float]]:
        return {
            method: (
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            )
            for method, samples in values.items()
        }

    deltas = {
        method: {
            "delta_mean": float(np.mean(samples)),
            "delta_ci_low": float(np.quantile(samples, 0.025)),
            "delta_ci_high": float(np.quantile(samples, 0.975)),
            "probability_positive": float(
                np.mean(np.asarray(samples) > 0)
            ),
        }
        for method, samples in delta_samples.items()
    }
    return intervals(auc_samples), intervals(ap_samples), deltas, valid


def evaluate_target(
    target: str,
    frame: pd.DataFrame,
    aggregation: str,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    patient_tables: dict[str, pd.DataFrame] = {}
    for method, probability_column in METHODS.items():
        patient_tables[method] = aggregate_patients(
            frame,
            probability_column,
            aggregation,
        )
    identity_columns = ["patient_id", "target", "label", "images"]
    wide = patient_tables[next(iter(METHODS))][identity_columns].copy()
    for method, table in patient_tables.items():
        if not wide[identity_columns].equals(table[identity_columns]):
            raise RuntimeError(
                f"{target}/{aggregation}: patient rows differ across methods"
            )
        wide[f"probability_{method}"] = table["probability"]

    auc_intervals, ap_intervals, deltas, valid = bootstrap_metrics(
        wide,
        tuple(METHODS),
        BOOTSTRAP_ITERATIONS,
        seed_offset,
    )
    metric_rows: list[dict[str, object]] = []
    for method in METHODS:
        labels = wide["target"].to_numpy()
        probability = wide[f"probability_{method}"].to_numpy()
        metric_rows.append(
            {
                "target": target,
                "aggregation": aggregation,
                "method": method,
                "patients": len(wide),
                "images": int(wide["images"].sum()),
                "patients_with_multiple_images": int(
                    (wide["images"] > 1).sum()
                ),
                "auc": float(roc_auc_score(labels, probability)),
                "auc_ci_low": auc_intervals[method][0],
                "auc_ci_high": auc_intervals[method][1],
                "average_precision": float(
                    average_precision_score(labels, probability)
                ),
                "ap_ci_low": ap_intervals[method][0],
                "ap_ci_high": ap_intervals[method][1],
            }
        )
    point_aucs = {
        row["method"]: float(row["auc"]) for row in metric_rows
    }
    delta_rows = [
        {
            "target": target,
            "aggregation": aggregation,
            "reference": REFERENCE,
            "candidate": method,
            "point_delta": point_aucs[method] - point_aucs[REFERENCE],
            **interval,
            "bootstrap_iterations_requested": BOOTSTRAP_ITERATIONS,
            "bootstrap_iterations_valid": valid,
        }
        for method, interval in deltas.items()
    ]
    audit = {
        "target": target,
        "aggregation": aggregation,
        "patients": len(wide),
        "images": int(wide["images"].sum()),
        "patients_with_multiple_images": int((wide["images"] > 1).sum()),
        "maximum_images_per_patient": int(wide["images"].max()),
        "bootstrap_iterations_requested": BOOTSTRAP_ITERATIONS,
        "bootstrap_iterations_valid": valid,
        "primary_delta": next(
            row for row in delta_rows if row["candidate"] == PRIMARY
        ),
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows), wide, audit


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metric_frames: list[pd.DataFrame] = []
    delta_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for target_index, (target, path) in enumerate(TARGET_FILES.items()):
        frame = pd.read_csv(path)
        frame["patient_id"] = frame["patient_id"].astype(str)
        for aggregation_index, aggregation in enumerate(("mean", "median")):
            metrics, deltas, wide, audit = evaluate_target(
                target,
                frame,
                aggregation,
                100 * target_index + aggregation_index,
            )
            metric_frames.append(metrics)
            delta_frames.append(deltas)
            audits.append(audit)
            wide.to_csv(
                OUTPUT_DIR
                / f"{target}_{aggregation}_patient_predictions.csv",
                index=False,
            )

    metric_table = pd.concat(metric_frames, ignore_index=True)
    delta_table = pd.concat(delta_frames, ignore_index=True)
    metric_table.to_csv(
        OUTPUT_DIR / "patient_level_external_metrics.csv",
        index=False,
    )
    delta_table.to_csv(
        OUTPUT_DIR / "patient_level_external_differences.csv",
        index=False,
    )
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "aggregation_rules": ["mean", "median"],
        "methods": METHODS,
        "reference": REFERENCE,
        "primary": PRIMARY,
        "audits": audits,
    }
    (OUTPUT_DIR / "patient_level_external_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
