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


OUTPUT_DIR = ROOT / "results" / "v04_feature_domain_ablation"
DATASET_ORDER = ("busuclm", "busi", "busbra", "breast")
DATASET_LABELS = {
    "busuclm": "BUS-UCLM",
    "busi": "BUSI",
    "busbra": "BUS-BRA",
    "breast": "BrEaST",
}
BLOCK_LABELS = {
    "multifractal30": "Multifractal (30)",
    "wavelet36": "Wavelet (36)",
    "shape10": "Boundary shape (10)",
    "lesion22": "Lesion region (22)",
    "inner22": "Inner band (22)",
    "outer22": "Outer band (22)",
    "texture66": "Multiscale texture (66)",
    "advanced76": "Texture + shape (76)",
}
BLOCK_ORDER = tuple(BLOCK_LABELS)


def feature_blocks(frame: pd.DataFrame) -> dict[str, list[str]]:
    multifractal = [
        column for column in frame if column.startswith("advanced_mf_")
    ]
    wavelet = [
        column for column in frame if column.startswith("advanced_wavelet_")
    ]
    shape = [
        column for column in frame if column.startswith("advanced_shape_")
    ]

    def region_columns(region: str) -> list[str]:
        mf_prefix = f"advanced_mf_{region}_"
        wavelet_prefix = f"advanced_wavelet_{region}_"
        return [
            column
            for column in frame
            if column.startswith((mf_prefix, wavelet_prefix))
        ]

    blocks = {
        "multifractal30": multifractal,
        "wavelet36": wavelet,
        "shape10": shape,
        "lesion22": region_columns("lesion"),
        "inner22": region_columns("inner"),
        "outer22": region_columns("outer"),
        "texture66": multifractal + wavelet,
        "advanced76": multifractal + wavelet + shape,
    }
    expected = {
        "multifractal30": 30,
        "wavelet36": 36,
        "shape10": 10,
        "lesion22": 22,
        "inner22": 22,
        "outer22": 22,
        "texture66": 66,
        "advanced76": 76,
    }
    for name, columns in blocks.items():
        if len(columns) != expected[name] or len(columns) != len(set(columns)):
            raise RuntimeError(
                f"{name}: expected {expected[name]} unique columns, "
                f"found {len(columns)}"
            )
    return blocks


def build_model(
    source: pd.DataFrame,
    columns: list[str],
    splits: int,
) -> tuple[Pipeline, float]:
    working = source.copy()
    working["domain_group"] = working["patient_id"].astype(str)
    best_c, _ = source_tune_c(working, columns, splits)
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
    return model, float(best_c)


def weighted_auc(frame: pd.DataFrame, probability: np.ndarray) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            probability,
            sample_weight=patient_equal_weights(frame),
        )
    )


def lodo_ablation(
    datasets: dict[str, pd.DataFrame],
    blocks: dict[str, list[str]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for held_out in DATASET_ORDER:
        target = datasets[held_out]
        sources = [name for name in DATASET_ORDER if name != held_out]
        for block in BLOCK_ORDER:
            columns = blocks[block]
            parts = []
            for source_name in sources:
                part = datasets[source_name][
                    ["image", "patient_id", "label", "target"] + columns
                ].copy()
                part["patient_id"] = (
                    source_name + ":" + part["patient_id"].astype(str)
                )
                part["domain_group"] = part["patient_id"]
                parts.append(part)
            pooled = pd.concat(parts, ignore_index=True)
            model, best_c = build_model(pooled, columns, 5)
            probability = model.predict_proba(target[columns])[:, 1]
            records.append(
                {
                    "held_out": held_out,
                    "block": block,
                    "features": len(columns),
                    "auc": weighted_auc(target, probability),
                    "best_c": best_c,
                    "source_rows": len(pooled),
                    "source_groups": pooled["patient_id"].nunique(),
                }
            )
    return pd.DataFrame(records)


def locked_external_ablation(
    datasets: dict[str, pd.DataFrame],
    blocks: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {
        target: datasets[target][
            ["image", "patient_id", "label", "target", "birads", "device"]
        ].copy()
        for target in ("busbra", "breast")
    }
    for block in BLOCK_ORDER:
        columns = blocks[block]
        source_probabilities: dict[str, dict[str, np.ndarray]] = {}
        source_cs: dict[str, float] = {}
        for source_name in ("busuclm", "busi"):
            source = datasets[source_name]
            splits = 4 if source_name == "busuclm" else 5
            model, best_c = build_model(source, columns, splits)
            source_cs[source_name] = best_c
            source_probabilities[source_name] = {
                target: model.predict_proba(datasets[target][columns])[:, 1]
                for target in ("busbra", "breast")
            }
        for target in ("busbra", "breast"):
            probability = np.mean(
                [
                    source_probabilities[source][target]
                    for source in ("busuclm", "busi")
                ],
                axis=0,
            )
            prediction_frames[target][f"probability_{block}"] = probability
            records.append(
                {
                    "target": target,
                    "block": block,
                    "features": len(columns),
                    "auc": weighted_auc(datasets[target], probability),
                    "busuclm_best_c": source_cs["busuclm"],
                    "busi_best_c": source_cs["busi"],
                }
            )
    return pd.DataFrame(records), prediction_frames


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
    blocks = feature_blocks(datasets["busuclm"])
    for dataset in DATASET_ORDER:
        candidate = feature_blocks(datasets[dataset])
        if candidate != blocks:
            raise RuntimeError(f"{dataset}: feature block columns differ")

    lodo = lodo_ablation(datasets, blocks)
    lodo.to_csv(OUTPUT_DIR / "lodo_feature_domain_ablation.csv", index=False)
    external, predictions = locked_external_ablation(datasets, blocks)
    external.to_csv(
        OUTPUT_DIR / "locked_external_feature_domain_ablation.csv",
        index=False,
    )
    for target, frame in predictions.items():
        frame.to_csv(
            OUTPUT_DIR / f"{target}_feature_domain_predictions.csv",
            index=False,
        )

    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "feature_blocks": {
            name: {"features": len(columns), "columns": columns}
            for name, columns in blocks.items()
        },
        "lodo_mean": (
            lodo.groupby("block")["auc"].mean().sort_values(ascending=False)
        ).to_dict(),
        "lodo_minimum": (
            lodo.groupby("block")["auc"].min().sort_values(ascending=False)
        ).to_dict(),
        "locked_external": external.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "feature_domain_ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
