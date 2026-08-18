from __future__ import annotations

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

from evaluate_v04_locked_external import external_frame  # noqa: E402
from run_busuclm_grouped_cv import SEED, patient_equal_weights  # noqa: E402
from run_cross_dataset_transfer import source_tune_c  # noqa: E402
from screen_v04_embedding import prepare_metadata  # noqa: E402
from screen_v04_handcrafted import feature_columns  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "v04_four_dataset_transfer"
DATASET_ORDER = ("busuclm", "busi", "busbra", "breast")
METHODS = {
    "frozen98": "frozen98",
    "advanced76": "advanced76",
    "frozen98_shape": "frozen98_shape",
}


def build_model(
    source: pd.DataFrame,
    columns: list[str],
    splits: int,
) -> tuple[Pipeline, float, list[dict[str, object]]]:
    working = source.copy()
    working["domain_group"] = working["patient_id"]
    best_c, tuning = source_tune_c(working, columns, splits)
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
    return model, float(best_c), tuning


def auc(frame: pd.DataFrame, column: str) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            frame[column],
            sample_weight=patient_equal_weights(frame),
        )
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    datasets = {
        "busuclm": prepare_metadata("busuclm"),
        "busi": prepare_metadata("busi"),
        "busbra": external_frame("busbra"),
        "breast": external_frame("breast"),
    }
    for name, frame in datasets.items():
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["domain_group"] = frame["patient_id"]
        frame["dataset"] = name
    column_map = {
        method: feature_columns(datasets["busuclm"], group)
        for method, group in METHODS.items()
    }
    for dataset in DATASET_ORDER:
        for method, group in METHODS.items():
            if column_map[method] != feature_columns(
                datasets[dataset],
                group,
            ):
                raise RuntimeError(
                    f"{dataset}: columns differ for {method}"
                )

    pair_records: list[dict[str, object]] = []
    matrices = {
        method: pd.DataFrame(
            np.nan,
            index=DATASET_ORDER,
            columns=DATASET_ORDER,
        )
        for method in METHODS
    }
    for source_name in DATASET_ORDER:
        source = datasets[source_name]
        splits = 4 if source_name == "busuclm" else 5
        fitted = {}
        for method, columns in column_map.items():
            model, best_c, tuning = build_model(
                source,
                columns,
                splits,
            )
            fitted[method] = model
            tuning_frame = pd.DataFrame(tuning)
            tuning_frame.insert(0, "method", method)
            tuning_frame.to_csv(
                OUTPUT_DIR
                / f"{source_name}_source_tuning_{method}.csv",
                index=False,
            )
            fitted[f"{method}_best_c"] = best_c
        for target_name in DATASET_ORDER:
            if source_name == target_name:
                continue
            target = datasets[target_name]
            prediction = target[
                ["image", "patient_id", "label", "target"]
            ].copy()
            for method, columns in column_map.items():
                column = f"probability_{method}"
                prediction[column] = fitted[method].predict_proba(
                    target[columns]
                )[:, 1]
                value = auc(prediction, column)
                matrices[method].loc[source_name, target_name] = value
                pair_records.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "method": method,
                        "auc": value,
                        "best_c": fitted[f"{method}_best_c"],
                        "source_rows": int(len(source)),
                        "source_groups": int(
                            source["patient_id"].nunique()
                        ),
                        "target_rows": int(len(target)),
                        "target_groups": int(
                            target["patient_id"].nunique()
                        ),
                    }
                )
            prediction.to_csv(
                OUTPUT_DIR
                / f"{source_name}_to_{target_name}_predictions.csv",
                index=False,
            )

    lodo_records: list[dict[str, object]] = []
    lodo_predictions: dict[str, pd.DataFrame] = {}
    for held_out in DATASET_ORDER:
        target = datasets[held_out]
        prediction = target[
            ["image", "patient_id", "label", "target"]
        ].copy()
        source_names = [
            dataset for dataset in DATASET_ORDER if dataset != held_out
        ]
        for method, columns in column_map.items():
            source_parts = []
            for source_name in source_names:
                part = datasets[source_name][
                    ["image", "patient_id", "label", "target"] + columns
                ].copy()
                part["patient_id"] = (
                    source_name + ":" + part["patient_id"].astype(str)
                )
                part["domain_group"] = part["patient_id"]
                source_parts.append(part)
            pooled = pd.concat(source_parts, ignore_index=True)
            model, best_c, tuning = build_model(pooled, columns, 5)
            probability = model.predict_proba(target[columns])[:, 1]
            column = f"probability_{method}"
            prediction[column] = probability
            value = auc(prediction, column)
            lodo_records.append(
                {
                    "held_out": held_out,
                    "sources": "|".join(source_names),
                    "method": method,
                    "auc": value,
                    "best_c": best_c,
                    "source_rows": int(len(pooled)),
                    "source_groups": int(
                        pooled["patient_id"].nunique()
                    ),
                    "target_rows": int(len(target)),
                    "target_groups": int(
                        target["patient_id"].nunique()
                    ),
                }
            )
            tuning_frame = pd.DataFrame(tuning)
            tuning_frame.insert(0, "method", method)
            tuning_frame.to_csv(
                OUTPUT_DIR / f"lodo_{held_out}_tuning_{method}.csv",
                index=False,
            )
        prediction.to_csv(
            OUTPUT_DIR / f"lodo_{held_out}_predictions.csv",
            index=False,
        )
        lodo_predictions[held_out] = prediction

    pair_frame = pd.DataFrame(pair_records)
    pair_frame.to_csv(OUTPUT_DIR / "all_pair_transfer.csv", index=False)
    lodo_frame = pd.DataFrame(lodo_records)
    lodo_frame.to_csv(OUTPUT_DIR / "lodo_summary.csv", index=False)
    for method, matrix in matrices.items():
        matrix.to_csv(OUTPUT_DIR / f"transfer_matrix_{method}.csv")
    result = {
        "protocol": 'V04_LOCKED_EXTERNAL_PROTOCOL_V1',
        "datasets": {
            name: {
                "rows": int(len(frame)),
                "groups": int(frame["patient_id"].nunique()),
            }
            for name, frame in datasets.items()
        },
        "methods": METHODS,
        "all_pair": pair_records,
        "lodo": lodo_records,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (OUTPUT_DIR / "four_dataset_transfer_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "lodo": lodo_records,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
