from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_busuclm_grouped_cv import (  # noqa: E402
    SEED,
    evaluate_group,
    patient_equal_weights,
)
from run_cross_dataset_transfer import source_tune_c  # noqa: E402


INPUT_DIR = ROOT / "results" / "v04_advanced_features"
OUTPUT_DIR = ROOT / "results" / "v04_handcrafted_screen"
GROUPS = (
    "frozen98",
    "advanced76",
    "frozen98_advanced",
    "multifractal30",
    "wavelet36",
    "shape10",
    "frozen98_multifractal",
    "frozen98_wavelet",
    "frozen98_shape",
    "frozen98_multifractal_shape",
    "frozen98_component_tree",
)


def feature_columns(frame: pd.DataFrame, group: str) -> list[str]:
    conventional = [
        column for column in frame if column.startswith("basic_")
    ]
    fractal = [
        column for column in frame if column.startswith("fractal_")
    ]
    extrema = [column for column in frame if column.startswith("zone_")]
    component_tree = [
        column for column in frame if column.startswith("ct_")
    ]
    multifractal = [
        column for column in frame if column.startswith("advanced_mf_")
    ]
    wavelet = [
        column for column in frame if column.startswith("advanced_wavelet_")
    ]
    shape = [
        column for column in frame if column.startswith("advanced_shape_")
    ]
    frozen = conventional + fractal + extrema
    mapping = {
        "frozen98": frozen,
        "advanced76": multifractal + wavelet + shape,
        "frozen98_advanced": frozen + multifractal + wavelet + shape,
        "multifractal30": multifractal,
        "wavelet36": wavelet,
        "shape10": shape,
        "frozen98_multifractal": frozen + multifractal,
        "frozen98_wavelet": frozen + wavelet,
        "frozen98_shape": frozen + shape,
        "frozen98_multifractal_shape": frozen + multifractal + shape,
        "frozen98_component_tree": frozen + component_tree,
    }
    selected = mapping[group]
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"duplicate columns in {group}")
    return selected


def prepare(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    result = frame.copy()
    if dataset == "busi":
        result["patient_id"] = result["cv_group_id"].astype(str)
    else:
        result["patient_id"] = result["patient_id"].astype(str)
    result["target"] = (result["label"] == "malignant").astype(np.int64)
    result["domain_group"] = result["patient_id"]
    return result.sort_values(["patient_id", "image"]).reset_index(drop=True)


def within_auc(
    frame: pd.DataFrame,
    columns: list[str],
    dataset: str,
    output: Path,
) -> tuple[float, dict[str, object]]:
    if dataset == "busi":
        outer_splits, inner_splits = 5, 4
    else:
        outer_splits, inner_splits = 4, 3
    predictions, tuning, folds = evaluate_group(
        frame,
        columns,
        outer_splits,
        inner_splits,
        repetitions=3,
    )
    predictions.to_csv(output / f"{dataset}_oof_predictions.csv", index=False)
    tuning.to_csv(output / f"{dataset}_tuning.csv", index=False)
    folds.to_csv(output / f"{dataset}_fold_assignments.csv", index=False)
    means = (
        predictions.groupby(
            ["image", "patient_id", "label", "target"],
            as_index=False,
        )["probability"]
        .mean()
    )
    means.to_csv(output / f"{dataset}_mean_oof_predictions.csv", index=False)
    auc = float(
        roc_auc_score(
            means["target"],
            means["probability"],
            sample_weight=patient_equal_weights(means),
        )
    )
    fold_overlap = 0
    for _, assignment in folds.groupby(["repeat", "fold"]):
        train_ids = set(
            str(
                assignment.loc[
                    assignment["split"] == "train",
                    "patient_ids",
                ].iloc[0]
            ).split("|")
        )
        test_ids = set(
            str(
                assignment.loc[
                    assignment["split"] == "test",
                    "patient_ids",
                ].iloc[0]
            ).split("|")
        )
        fold_overlap += len(train_ids & test_ids)
    return auc, {
        "auc": auc,
        "rows": int(len(frame)),
        "groups": int(frame["patient_id"].nunique()),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "repetitions": 3,
        "fold_group_overlaps": fold_overlap,
    }


def transfer_auc(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: list[str],
    source_name: str,
    target_name: str,
    output: Path,
) -> tuple[float, dict[str, object]]:
    source_splits = 5 if source_name == "busi" else 4
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
    prediction = target[
        ["image", "patient_id", "label", "target"]
    ].copy()
    prediction["probability"] = probability
    direction = f"{source_name}_to_{target_name}"
    prediction.to_csv(output / f"{direction}_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(
        output / f"{direction}_tuning.csv",
        index=False,
    )
    auc = float(
        roc_auc_score(
            prediction["target"],
            probability,
            sample_weight=patient_equal_weights(prediction),
        )
    )
    return auc, {
        "source": source_name,
        "target": target_name,
        "auc": auc,
        "best_c": best_c,
        "source_rows": int(len(source)),
        "source_groups": int(source["patient_id"].nunique()),
        "target_rows": int(len(target)),
        "target_groups": int(target["patient_id"].nunique()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    args = parser.parse_args()

    output = OUTPUT_DIR / args.group
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    busuclm = prepare(
        pd.read_csv(INPUT_DIR / "features_busuclm_advanced.csv"),
        "busuclm",
    )
    busi = prepare(
        pd.read_csv(INPUT_DIR / "features_busi_advanced.csv"),
        "busi",
    )
    columns = feature_columns(busuclm, args.group)
    if columns != feature_columns(busi, args.group):
        raise RuntimeError("feature columns differ between development cohorts")
    values = np.concatenate(
        [
            busuclm[columns].to_numpy(dtype=np.float64).ravel(),
            busi[columns].to_numpy(dtype=np.float64).ravel(),
        ]
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{args.group} contains non-finite values")

    uclm_internal, uclm_internal_record = within_auc(
        busuclm,
        columns,
        "busuclm",
        output,
    )
    busi_internal, busi_internal_record = within_auc(
        busi,
        columns,
        "busi",
        output,
    )
    uclm_to_busi, uclm_to_busi_record = transfer_auc(
        busuclm,
        busi,
        columns,
        "busuclm",
        "busi",
        output,
    )
    busi_to_uclm, busi_to_uclm_record = transfer_auc(
        busi,
        busuclm,
        columns,
        "busi",
        "busuclm",
        output,
    )
    score = float(
        np.mean(
            [
                uclm_internal,
                busi_internal,
                uclm_to_busi,
                busi_to_uclm,
            ]
        )
    )
    result = {
        "protocol": 'V04_DEVELOPMENT_PROTOCOL_V1',
        "group": args.group,
        "n_features": len(columns),
        "feature_columns": columns,
        "within": {
            "busuclm": uclm_internal_record,
            "busi": busi_internal_record,
        },
        "transfer": {
            "busuclm_to_busi": uclm_to_busi_record,
            "busi_to_busuclm": busi_to_uclm_record,
        },
        "development_score": score,
        "minimum_transfer_auc": float(
            min(uclm_to_busi, busi_to_uclm)
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
