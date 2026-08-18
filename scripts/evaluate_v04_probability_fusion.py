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
sys.path.insert(0, str(ROOT / "scripts"))

from run_busuclm_grouped_cv import (  # noqa: E402
    SEED,
    evaluate_group,
    patient_equal_weights,
)
from run_cross_dataset_transfer import source_tune_c  # noqa: E402


HAND_DIR = ROOT / "results" / "v04_handcrafted_screen" / "advanced76"
DEEP_DIR = (
    ROOT
    / "results"
    / "v04_embedding_screen"
    / "resnet18_inner_only"
)
OUTPUT_DIR = ROOT / "results" / "v04_probability_fusion"
FEATURE_COLUMNS = ["probability_handcrafted", "probability_deep"]


def merge_predictions(
    handcrafted_path: Path,
    deep_path: Path,
) -> pd.DataFrame:
    handcrafted = pd.read_csv(handcrafted_path).rename(
        columns={"probability": "probability_handcrafted"}
    )
    deep = pd.read_csv(deep_path).rename(
        columns={"probability": "probability_deep"}
    )
    keys = ["image", "patient_id", "label", "target"]
    merged = handcrafted.merge(
        deep,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(handcrafted) or len(merged) != len(deep):
        raise RuntimeError("probability-fusion row alignment failed")
    merged["probability_late_mean"] = (
        merged["probability_handcrafted"] + merged["probability_deep"]
    ) / 2.0
    return merged


def weighted_auc(
    frame: pd.DataFrame,
    probability_column: str,
) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            frame[probability_column],
            sample_weight=patient_equal_weights(frame),
        )
    )


def within_dataset(dataset: str) -> dict[str, float]:
    merged = merge_predictions(
        HAND_DIR / f"{dataset}_mean_oof_predictions.csv",
        DEEP_DIR / f"{dataset}_mean_oof_predictions.csv",
    )
    late_auc = weighted_auc(merged, "probability_late_mean")

    outer_splits, inner_splits = (
        (5, 4) if dataset == "busi" else (4, 3)
    )
    predictions, tuning, folds = evaluate_group(
        merged,
        FEATURE_COLUMNS,
        outer_splits,
        inner_splits,
        repetitions=3,
    )
    stacked = (
        predictions.groupby(
            ["image", "patient_id", "label", "target"],
            as_index=False,
        )["probability"]
        .mean()
    )
    stacked_auc = weighted_auc(stacked, "probability")

    merged.to_csv(
        OUTPUT_DIR / f"{dataset}_late_mean_predictions.csv",
        index=False,
    )
    predictions.to_csv(
        OUTPUT_DIR / f"{dataset}_stacked_oof_predictions.csv",
        index=False,
    )
    tuning.to_csv(
        OUTPUT_DIR / f"{dataset}_stacked_tuning.csv",
        index=False,
    )
    folds.to_csv(
        OUTPUT_DIR / f"{dataset}_stacked_folds.csv",
        index=False,
    )
    return {
        "handcrafted_auc": weighted_auc(
            merged,
            "probability_handcrafted",
        ),
        "deep_auc": weighted_auc(merged, "probability_deep"),
        "late_mean_auc": late_auc,
        "stacked_auc": stacked_auc,
    }


def transfer_direction(
    source_name: str,
    target_name: str,
) -> dict[str, float]:
    source = merge_predictions(
        HAND_DIR / f"{source_name}_mean_oof_predictions.csv",
        DEEP_DIR / f"{source_name}_mean_oof_predictions.csv",
    )
    target = merge_predictions(
        HAND_DIR / f"{source_name}_to_{target_name}_predictions.csv",
        DEEP_DIR / f"{source_name}_to_{target_name}_predictions.csv",
    )
    source["target"] = source["target"].astype(np.int64)
    target["target"] = target["target"].astype(np.int64)
    source["domain_group"] = source["patient_id"]
    source_splits = 5 if source_name == "busi" else 4
    best_c, tuning = source_tune_c(
        source,
        FEATURE_COLUMNS,
        source_splits,
    )
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
    model.fit(source[FEATURE_COLUMNS], source["target"])
    target["probability_stacked"] = model.predict_proba(
        target[FEATURE_COLUMNS]
    )[:, 1]
    direction = f"{source_name}_to_{target_name}"
    target.to_csv(
        OUTPUT_DIR / f"{direction}_fusion_predictions.csv",
        index=False,
    )
    pd.DataFrame(tuning).to_csv(
        OUTPUT_DIR / f"{direction}_stacked_tuning.csv",
        index=False,
    )
    return {
        "handcrafted_auc": weighted_auc(
            target,
            "probability_handcrafted",
        ),
        "deep_auc": weighted_auc(target, "probability_deep"),
        "late_mean_auc": weighted_auc(
            target,
            "probability_late_mean",
        ),
        "stacked_auc": weighted_auc(target, "probability_stacked"),
        "stacked_best_c": float(best_c),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    within = {
        "busuclm": within_dataset("busuclm"),
        "busi": within_dataset("busi"),
    }
    transfer = {
        "busuclm_to_busi": transfer_direction("busuclm", "busi"),
        "busi_to_busuclm": transfer_direction("busi", "busuclm"),
    }
    methods = ("handcrafted", "deep", "late_mean", "stacked")
    development_scores = {}
    for method in methods:
        development_scores[method] = float(
            np.mean(
                [
                    within["busuclm"][f"{method}_auc"],
                    within["busi"][f"{method}_auc"],
                    transfer["busuclm_to_busi"][f"{method}_auc"],
                    transfer["busi_to_busuclm"][f"{method}_auc"],
                ]
            )
        )
    result = {
        "protocol": 'V04_DEVELOPMENT_PROTOCOL_V1',
        "handcrafted": "advanced76",
        "deep": "resnet18_inner_only_pca32",
        "within": within,
        "transfer": transfer,
        "development_scores": development_scores,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
