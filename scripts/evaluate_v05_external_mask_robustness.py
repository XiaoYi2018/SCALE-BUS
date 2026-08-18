from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_v04_feature_domain_ablation import (  # noqa: E402
    build_model,
    feature_blocks,
)
from evaluate_v04_locked_external import external_frame  # noqa: E402
from run_busuclm_grouped_cv import patient_equal_weights  # noqa: E402
from screen_v04_embedding import prepare_metadata  # noqa: E402


INPUT_DIR = ROOT / "results" / "v04_external_mask_perturbations"
OUTPUT_DIR = ROOT / "results" / "v05_external_mask_robustness"
DATASETS = ("busbra", "breast")
CONDITIONS = ("clean", "mask_erode_3", "mask_dilate_3")
METHODS = {
    "shape10": "Boundary shape (10)",
    "advanced76": "GFWB-76",
}
ITERATIONS = 5000
SEED = 20260718


def auc(frame: pd.DataFrame, probability: np.ndarray) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            probability,
            sample_weight=patient_equal_weights(frame),
        )
    )


def paired_bootstrap_delta(
    metadata: pd.DataFrame,
    clean: np.ndarray,
    perturbed: np.ndarray,
) -> tuple[float, float, float]:
    working = metadata[["patient_id", "target"]].copy()
    working["unit"] = (
        working["patient_id"].astype(str)
        + "|"
        + working["target"].astype(str)
    )
    units = {
        unit: part.index.to_numpy(dtype=np.int64)
        for unit, part in working.groupby("unit", sort=True)
    }
    names = np.asarray(list(units), dtype=object)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(ITERATIONS):
        sampled = rng.choice(names, size=len(names), replace=True)
        selected_groups = [units[name] for name in sampled]
        indices = np.concatenate(selected_groups)
        weights = np.concatenate(
            [
                np.repeat(1.0 / len(selected), len(selected))
                for selected in selected_groups
            ]
        )
        replicate = metadata.iloc[indices]
        labels = replicate["target"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        values.append(
            float(
                roc_auc_score(
                    labels,
                    perturbed[indices],
                    sample_weight=weights,
                )
            )
            - float(
                roc_auc_score(
                    labels,
                    clean[indices],
                    sample_weight=weights,
                )
            )
        )
    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        float(np.mean(array >= 0)),
    )


def load_condition(
    dataset: str,
    condition: str,
    clean: pd.DataFrame,
) -> pd.DataFrame:
    if condition == "clean":
        return clean
    frame = pd.read_csv(
        INPUT_DIR / f"features_{dataset}_{condition}.csv"
    )
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["target"] = (frame["label"] == "malignant").astype(np.int64)
    frame = frame.sort_values(["patient_id", "image"]).reset_index(drop=True)
    keys = ["image", "patient_id", "target"]
    if frame[keys].to_dict("records") != clean[keys].to_dict("records"):
        raise RuntimeError(f"{dataset} {condition}: order/metadata mismatch")
    return frame


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "busuclm": prepare_metadata("busuclm"),
        "busi": prepare_metadata("busi"),
    }
    clean_targets = {
        dataset: external_frame(dataset) for dataset in DATASETS
    }
    blocks = feature_blocks(sources["busuclm"])
    models = {}
    tuning = {}
    for method in METHODS:
        columns = blocks[method]
        for source_name, source in sources.items():
            model, best_c = build_model(
                source,
                columns,
                4 if source_name == "busuclm" else 5,
            )
            models[(method, source_name)] = model
            tuning[(method, source_name)] = best_c

    records: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    for dataset in DATASETS:
        clean = clean_targets[dataset]
        output = clean[
            ["image", "patient_id", "label", "target", "birads", "device"]
        ].copy()
        condition_frames = {
            condition: load_condition(dataset, condition, clean)
            for condition in CONDITIONS
        }
        clean_predictions: dict[str, np.ndarray] = {}
        for method in METHODS:
            columns = blocks[method]
            for condition in CONDITIONS:
                target = condition_frames[condition]
                probability = np.mean(
                    [
                        models[(method, source)].predict_proba(
                            target[columns]
                        )[:, 1]
                        for source in ("busuclm", "busi")
                    ],
                    axis=0,
                )
                output[f"probability_{method}_{condition}"] = probability
                value = auc(clean, probability)
                if condition == "clean":
                    clean_predictions[method] = probability
                    low = high = 0.0
                    nonnegative = np.nan
                    delta = 0.0
                else:
                    delta = value - auc(clean, clean_predictions[method])
                    low, high, nonnegative = paired_bootstrap_delta(
                        clean,
                        clean_predictions[method],
                        probability,
                    )
                records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "condition": condition,
                        "auc": value,
                        "delta_vs_clean": delta,
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                        "probability_nonnegative_delta": nonnegative,
                        "features": len(columns),
                    }
                )
        output.to_csv(
            OUTPUT_DIR / f"{dataset}_mask_robustness_predictions.csv",
            index=False,
        )
        prediction_frames[dataset] = output

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / "mask_robustness_metrics.csv", index=False)
    summary = {
        "bootstrap_iterations": ITERATIONS,
        "tuning": {
            f"{method}_{source}": value
            for (method, source), value in tuning.items()
        },
        "metrics": result.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "mask_robustness_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
