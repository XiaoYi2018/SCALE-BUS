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

from fractal_extrema import (  # noqa: E402
    MULTIZONE_FEATURE_GROUPS,
    feature_columns_for_group,
)
from run_busuclm_grouped_cv import (  # noqa: E402
    C_GRID,
    SEED,
    cluster_bootstrap,
    parse_bool,
    patient_equal_weights,
    valid_group_splits,
)


BUSUCLM_FEATURES = (
    ROOT / "results" / "busuclm_multizone_features" / "features_multizone.csv"
)
BUSI_FEATURES = ROOT / "results" / "busi_zenodo_features_v2" / "features.csv"
OUT_DIR = ROOT / "results" / "cross_dataset_transfer"
TUNING_REPETITIONS = 5
BOOTSTRAP_ITERATIONS = 5000


def prepare_datasets() -> dict[str, pd.DataFrame]:
    busuclm = pd.read_csv(BUSUCLM_FEATURES)
    busuclm = busuclm.loc[parse_bool(busuclm["is_clean_primary"])].copy()
    busuclm["target"] = (busuclm["label"] == "malignant").astype(np.int64)
    busuclm["domain_group"] = busuclm["patient_id"]
    busuclm = busuclm.sort_values(["domain_group", "image"]).reset_index(drop=True)

    busi = pd.read_csv(BUSI_FEATURES)
    busi["target"] = (busi["label"] == "malignant").astype(np.int64)
    busi["domain_group"] = busi["cv_group_id"]
    busi = busi.sort_values(["domain_group", "image"]).reset_index(drop=True)
    return {"BUS-UCLM-clean": busuclm, "BUSI-valid": busi}


def source_tune_c(
    source: pd.DataFrame,
    columns: list[str],
    n_splits: int,
) -> tuple[float, list[dict[str, float | int]]]:
    scores: dict[float, list[float]] = {float(value): [] for value in C_GRID}
    rows: list[dict[str, float | int]] = []
    working = source.copy()
    working["patient_id"] = working["domain_group"]
    for repeat in range(TUNING_REPETITIONS):
        folds, actual_seed = valid_group_splits(
            working,
            n_splits,
            SEED + repeat * 1000,
        )
        for fold, (train_idx, valid_idx) in enumerate(folds, 1):
            train = working.iloc[train_idx]
            valid = working.iloc[valid_idx]
            for c_value in C_GRID:
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=float(c_value),
                                class_weight="balanced",
                                max_iter=5000,
                                random_state=SEED,
                                solver="liblinear",
                            ),
                        ),
                    ]
                )
                model.fit(train[columns], train["target"])
                probability = model.predict_proba(valid[columns])[:, 1]
                auc = float(roc_auc_score(valid["target"], probability))
                scores[float(c_value)].append(auc)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "seed": actual_seed,
                        "c": float(c_value),
                        "auc": auc,
                    }
                )
    mean_scores = {
        c_value: float(np.mean(values)) for c_value, values in scores.items()
    }
    best_c = max(sorted(mean_scores), key=lambda value: mean_scores[value])
    return best_c, rows


def evaluate_direction(
    source_name: str,
    target_name: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> dict[str, object]:
    source_splits = 4 if source_name.startswith("BUS-UCLM") else 5
    prediction_frame = target[
        ["image", "label", "target", "domain_group"]
    ].copy()
    prediction_frame = prediction_frame.rename(
        columns={"domain_group": "patient_id"}
    )
    tuning_rows: list[pd.DataFrame] = []
    group_summaries: list[dict[str, object]] = []

    for group in MULTIZONE_FEATURE_GROUPS:
        columns = feature_columns_for_group(source.columns, group)
        best_c, tuning = source_tune_c(source, columns, source_splits)
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=best_c,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=SEED,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        model.fit(source[columns], source["target"])
        probability = model.predict_proba(target[columns])[:, 1]
        prediction_frame[f"probability_{group}"] = probability
        weights = patient_equal_weights(prediction_frame)
        group_summaries.append(
            {
                "group": group,
                "n_features": len(columns),
                "best_c": best_c,
                "source_cv_auc": float(
                    np.mean([row["auc"] for row in tuning if row["c"] == best_c])
                ),
                "target_image_auc": float(
                    roc_auc_score(target["target"], probability)
                ),
                "target_group_balanced_auc": float(
                    roc_auc_score(
                        target["target"],
                        probability,
                        sample_weight=weights,
                    )
                ),
            }
        )
        tuning_frame = pd.DataFrame(tuning)
        tuning_frame.insert(0, "group", group)
        tuning_rows.append(tuning_frame)
        print(
            f"{source_name} -> {target_name} {group}: "
            f"C={best_c} target_auc={group_summaries[-1]['target_group_balanced_auc']:.4f}",
            flush=True,
        )

    intervals, delta_basic = cluster_bootstrap(
        prediction_frame,
        list(MULTIZONE_FEATURE_GROUPS),
        BOOTSTRAP_ITERATIONS,
        "basic",
    )
    _, delta_fractal = cluster_bootstrap(
        prediction_frame,
        ["basic_fractal", "fused_multizone"],
        BOOTSTRAP_ITERATIONS,
        "basic_fractal",
    )
    for row in group_summaries:
        group = str(row["group"])
        row["target_group_balanced_auc_ci_low"] = intervals[group][0]
        row["target_group_balanced_auc_ci_high"] = intervals[group][1]
        if group in delta_basic:
            row["delta_vs_basic"] = delta_basic[group]
    direction = f"{source_name}_to_{target_name}".replace(" ", "_")
    prediction_frame.to_csv(OUT_DIR / f"{direction}_predictions.csv", index=False)
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        OUT_DIR / f"{direction}_source_tuning.csv",
        index=False,
    )
    pd.DataFrame(group_summaries).to_csv(
        OUT_DIR / f"{direction}_summary.csv",
        index=False,
    )
    return {
        "source": source_name,
        "target": target_name,
        "source_rows": int(len(source)),
        "source_groups": int(source["domain_group"].nunique()),
        "target_rows": int(len(target)),
        "target_groups": int(target["domain_group"].nunique()),
        "groups": group_summaries,
        "fused_multizone_vs_basic_fractal": delta_fractal["fused_multizone"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = prepare_datasets()
    results = [
        evaluate_direction(
            "BUS-UCLM-clean",
            "BUSI-valid",
            datasets["BUS-UCLM-clean"],
            datasets["BUSI-valid"],
        ),
        evaluate_direction(
            "BUSI-valid",
            "BUS-UCLM-clean",
            datasets["BUSI-valid"],
            datasets["BUS-UCLM-clean"],
        ),
    ]
    result = {
        "seed": SEED,
        "c_grid": list(C_GRID),
        "tuning_repetitions": TUNING_REPETITIONS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "directions": results,
    }
    (OUT_DIR / "cross_dataset_transfer.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
