from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_perturbed_features import CONDITIONS  # noqa: E402
from fractal_extrema import (  # noqa: E402
    MULTIZONE_FEATURE_GROUPS,
    feature_columns_for_group,
)
from run_busuclm_grouped_cv import patient_equal_weights  # noqa: E402


SEED = 20260717
BOOTSTRAP_ITERATIONS = 2000
OUT_DIR = ROOT / "results" / "cross_dataset_robustness"
TRANSFER_JSON = (
    ROOT
    / "results"
    / "cross_dataset_transfer"
    / "cross_dataset_transfer.json"
)
DATASETS = {
    "BUS-UCLM-clean": {
        "clean": (
            ROOT
            / "results"
            / "busuclm_multizone_features"
            / "features_multizone.csv"
        ),
        "perturbed": ROOT / "results" / "perturbed_features" / "busuclm",
    },
    "BUSI-valid": {
        "clean": ROOT / "results" / "busi_zenodo_features_v2" / "features.csv",
        "perturbed": ROOT / "results" / "perturbed_features" / "busi",
    },
}


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def load_clean(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATASETS[name]["clean"])
    if name == "BUS-UCLM-clean":
        frame = frame.loc[parse_bool(frame["is_clean_primary"])].copy()
    frame["target"] = (frame["label"] == "malignant").astype(np.int64)
    return frame.sort_values(["patient_id", "image"]).reset_index(drop=True)


def load_condition(name: str, condition: str) -> pd.DataFrame:
    if condition == "clean":
        return load_clean(name)
    path = DATASETS[name]["perturbed"] / f"features_{condition}.csv"
    frame = pd.read_csv(path)
    frame["target"] = (frame["label"] == "malignant").astype(np.int64)
    return frame.sort_values(["patient_id", "image"]).reset_index(drop=True)


def group_bootstrap_indices(
    frame: pd.DataFrame,
    iterations: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    groups = frame["patient_id"].unique()
    by_group = {
        group: frame.index[frame["patient_id"] == group].to_numpy()
        for group in groups
    }
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(iterations):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        indices: list[int] = []
        weights: list[float] = []
        for group in drawn:
            selected = by_group[group]
            indices.extend(selected.tolist())
            weights.extend([1.0 / len(selected)] * len(selected))
        samples.append((np.asarray(indices), np.asarray(weights)))
    return samples


def summarize_bootstrap(
    predictions: pd.DataFrame,
    groups: tuple[str, ...],
    conditions: tuple[str, ...],
) -> list[dict[str, object]]:
    clean_labels = predictions["target"].to_numpy()
    samples = group_bootstrap_indices(predictions, BOOTSTRAP_ITERATIONS)
    values: dict[tuple[str, str], list[float]] = {
        (group, condition): [] for group in groups for condition in conditions
    }
    for indices, weights in samples:
        labels = clean_labels[indices]
        if np.unique(labels).size < 2:
            continue
        for group in groups:
            for condition in conditions:
                values[(group, condition)].append(
                    float(
                        roc_auc_score(
                            labels,
                            predictions[
                                f"probability_{group}_{condition}"
                            ].to_numpy()[indices],
                            sample_weight=weights,
                        )
                    )
                )

    rows: list[dict[str, object]] = []
    sample_weight = patient_equal_weights(predictions)
    for group in groups:
        clean_samples = np.asarray(values[(group, "clean")])
        for condition in conditions:
            probability = predictions[
                f"probability_{group}_{condition}"
            ].to_numpy()
            auc = float(
                roc_auc_score(
                    predictions["target"],
                    probability,
                    sample_weight=sample_weight,
                )
            )
            auc_samples = np.asarray(values[(group, condition)])
            degradation = auc_samples - clean_samples
            basic_delta = auc_samples - np.asarray(
                values[("basic", condition)]
            )
            fractal_delta = auc_samples - np.asarray(
                values[("basic_fractal", condition)]
            )
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "auc": auc,
                    "auc_ci_low": float(np.quantile(auc_samples, 0.025)),
                    "auc_ci_high": float(np.quantile(auc_samples, 0.975)),
                    "delta_from_clean_mean": float(np.mean(degradation)),
                    "delta_from_clean_ci_low": float(
                        np.quantile(degradation, 0.025)
                    ),
                    "delta_from_clean_ci_high": float(
                        np.quantile(degradation, 0.975)
                    ),
                    "delta_vs_basic_mean": float(np.mean(basic_delta)),
                    "delta_vs_basic_ci_low": float(
                        np.quantile(basic_delta, 0.025)
                    ),
                    "delta_vs_basic_ci_high": float(
                        np.quantile(basic_delta, 0.975)
                    ),
                    "delta_vs_basic_fractal_mean": float(
                        np.mean(fractal_delta)
                    ),
                    "delta_vs_basic_fractal_ci_low": float(
                        np.quantile(fractal_delta, 0.025)
                    ),
                    "delta_vs_basic_fractal_ci_high": float(
                        np.quantile(fractal_delta, 0.975)
                    ),
                }
            )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transfer = json.loads(TRANSFER_JSON.read_text())
    all_results: list[dict[str, object]] = []
    conditions = ("clean",) + CONDITIONS

    for direction in transfer["directions"]:
        source_name = str(direction["source"])
        target_name = str(direction["target"])
        source = load_clean(source_name)
        target_clean = load_clean(target_name)
        c_by_group = {
            str(row["group"]): float(row["best_c"])
            for row in direction["groups"]
        }
        condition_frames = {
            condition: load_condition(target_name, condition)
            for condition in conditions
        }
        expected_images = target_clean["image"].tolist()
        for condition, frame in condition_frames.items():
            if frame["image"].tolist() != expected_images:
                raise RuntimeError(
                    f"{target_name} {condition}: target row alignment mismatch"
                )

        predictions = target_clean[
            ["image", "patient_id", "label", "target"]
        ].copy()
        for group in MULTIZONE_FEATURE_GROUPS:
            columns = feature_columns_for_group(source.columns, group)
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=c_by_group[group],
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=SEED,
                            solver="liblinear",
                        ),
                    ),
                ]
            )
            model.fit(source[columns], source["target"])
            for condition, frame in condition_frames.items():
                predictions[f"probability_{group}_{condition}"] = (
                    model.predict_proba(frame[columns])[:, 1]
                )
            print(
                f"{source_name} -> {target_name} {group}: scored "
                f"{len(conditions)} conditions",
                flush=True,
            )

        rows = summarize_bootstrap(
            predictions,
            MULTIZONE_FEATURE_GROUPS,
            conditions,
        )
        direction_name = (
            f"{source_name}_to_{target_name}".replace(" ", "_")
        )
        predictions.to_csv(
            OUT_DIR / f"{direction_name}_predictions.csv",
            index=False,
        )
        pd.DataFrame(rows).to_csv(
            OUT_DIR / f"{direction_name}_summary.csv",
            index=False,
        )
        all_results.append(
            {
                "source": source_name,
                "target": target_name,
                "rows": rows,
            }
        )

    result = {
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "conditions": list(conditions),
        "directions": all_results,
    }
    (OUT_DIR / "robustness_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print("robustness complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
