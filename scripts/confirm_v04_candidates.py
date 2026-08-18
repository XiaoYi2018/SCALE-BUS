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

from run_busuclm_grouped_cv import (  # noqa: E402
    SEED,
    cluster_bootstrap,
    evaluate_group,
    patient_equal_weights,
)
from run_cross_dataset_transfer import source_tune_c  # noqa: E402
from screen_v04_classifiers import (  # noqa: E402
    fit_predict as classifier_fit_predict,
    transfer as classifier_transfer,
    within as classifier_within,
)
from screen_v04_early_fusion import (  # noqa: E402
    transfer as early_transfer,
    within_dataset as early_within,
)
from screen_v04_embedding import (  # noqa: E402
    load_embedding,
    transfer as embedding_transfer,
    within_dataset as embedding_within,
)
from screen_v04_handcrafted import feature_columns  # noqa: E402
from screen_v04_embedding import prepare_metadata  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "v04_confirmation"
REPETITIONS = 10
BOOTSTRAP_ITERATIONS = 2000
ENCODER = "resnet18"
VIEW = "inner_only"
CONFIGURATIONS = (
    "frozen98_logistic",
    "advanced76_logistic",
    "frozen98_shape_logistic",
    "advanced76_linear_svm",
    "resnet18_inner_only",
    "advanced76_resnet18_early",
    "frozen98_advanced_resnet18_early",
    "frozen98_shape_resnet18_early",
)
REFERENCE = "frozen98_logistic"


def mean_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(
            ["image", "patient_id", "label", "target"],
            as_index=False,
        )["probability"]
        .mean()
    )


def weighted_auc(predictions: pd.DataFrame) -> float:
    return float(
        roc_auc_score(
            predictions["target"],
            predictions["probability"],
            sample_weight=patient_equal_weights(predictions),
        )
    )


def run_handcrafted(
    name: str,
    group: str,
    frames: dict[str, pd.DataFrame],
) -> dict[str, object]:
    output = OUTPUT_DIR / name
    output.mkdir(parents=True, exist_ok=True)
    columns = feature_columns(frames["busuclm"], group)
    if columns != feature_columns(frames["busi"], group):
        raise RuntimeError(f"{group} feature columns differ")
    result: dict[str, object] = {
        "configuration": name,
        "feature_group": group,
        "n_features": len(columns),
        "classifier": "logistic",
        "within": {},
        "transfer": {},
    }
    for dataset, outer_splits, inner_splits in (
        ("busuclm", 4, 3),
        ("busi", 5, 4),
    ):
        predictions, tuning, folds = evaluate_group(
            frames[dataset],
            columns,
            outer_splits,
            inner_splits,
            repetitions=REPETITIONS,
        )
        means = mean_predictions(predictions)
        predictions.to_csv(
            output / f"{dataset}_oof_predictions.csv",
            index=False,
        )
        means.to_csv(
            output / f"{dataset}_mean_oof_predictions.csv",
            index=False,
        )
        tuning.to_csv(output / f"{dataset}_tuning.csv", index=False)
        folds.to_csv(
            output / f"{dataset}_fold_assignments.csv",
            index=False,
        )
        result["within"][dataset] = {
            "auc": weighted_auc(means),
            "rows": int(len(means)),
            "groups": int(means["patient_id"].nunique()),
            "repetitions": REPETITIONS,
        }
    for source_name, target_name in (
        ("busuclm", "busi"),
        ("busi", "busuclm"),
    ):
        source = frames[source_name].copy()
        target = frames[target_name].copy()
        source["domain_group"] = source["patient_id"]
        best_c, tuning = source_tune_c(
            source,
            columns,
            5 if source_name == "busi" else 4,
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
        model.fit(source[columns], source["target"])
        prediction = target[
            ["image", "patient_id", "label", "target"]
        ].copy()
        prediction["probability"] = model.predict_proba(
            target[columns]
        )[:, 1]
        direction = f"{source_name}_to_{target_name}"
        prediction.to_csv(
            output / f"{direction}_predictions.csv",
            index=False,
        )
        pd.DataFrame(tuning).to_csv(
            output / f"{direction}_tuning.csv",
            index=False,
        )
        result["transfer"][direction] = {
            "auc": weighted_auc(prediction),
            "best_c": float(best_c),
        }
    return result


def run_linear_svm(
    frames: dict[str, pd.DataFrame],
) -> dict[str, object]:
    name = "advanced76_linear_svm"
    output = OUTPUT_DIR / name
    output.mkdir(parents=True, exist_ok=True)
    columns = feature_columns(frames["busuclm"], "advanced76")
    values = {
        dataset: frames[dataset][columns].to_numpy(dtype=np.float64)
        for dataset in frames
    }
    uclm_auc, uclm_record = classifier_within(
        "linear_svm",
        "busuclm",
        frames["busuclm"],
        values["busuclm"],
        output,
        repetitions=REPETITIONS,
    )
    busi_auc, busi_record = classifier_within(
        "linear_svm",
        "busi",
        frames["busi"],
        values["busi"],
        output,
        repetitions=REPETITIONS,
    )
    u2b_auc, u2b_record = classifier_transfer(
        "linear_svm",
        "busuclm",
        "busi",
        frames["busuclm"],
        values["busuclm"],
        frames["busi"],
        values["busi"],
        output,
    )
    b2u_auc, b2u_record = classifier_transfer(
        "linear_svm",
        "busi",
        "busuclm",
        frames["busi"],
        values["busi"],
        frames["busuclm"],
        values["busuclm"],
        output,
    )
    return {
        "configuration": name,
        "feature_group": "advanced76",
        "n_features": len(columns),
        "classifier": "linear_svm",
        "within": {
            "busuclm": uclm_record,
            "busi": busi_record,
        },
        "transfer": {
            "busuclm_to_busi": u2b_record,
            "busi_to_busuclm": b2u_record,
        },
        "development_score": float(
            np.mean([uclm_auc, busi_auc, u2b_auc, b2u_auc])
        ),
    }


def run_embedding(
    frames: dict[str, pd.DataFrame],
    embeddings: dict[str, np.ndarray],
) -> dict[str, object]:
    name = "resnet18_inner_only"
    output = OUTPUT_DIR / name
    output.mkdir(parents=True, exist_ok=True)
    uclm_auc, uclm_record = embedding_within(
        "busuclm",
        frames["busuclm"],
        embeddings["busuclm"],
        output,
        repetitions=REPETITIONS,
    )
    busi_auc, busi_record = embedding_within(
        "busi",
        frames["busi"],
        embeddings["busi"],
        output,
        repetitions=REPETITIONS,
    )
    u2b_auc, u2b_record = embedding_transfer(
        "busuclm",
        "busi",
        frames["busuclm"],
        embeddings["busuclm"],
        frames["busi"],
        embeddings["busi"],
        output,
    )
    b2u_auc, b2u_record = embedding_transfer(
        "busi",
        "busuclm",
        frames["busi"],
        embeddings["busi"],
        frames["busuclm"],
        embeddings["busuclm"],
        output,
    )
    return {
        "configuration": name,
        "encoder": ENCODER,
        "view": VIEW,
        "pca_components": 32,
        "within": {
            "busuclm": uclm_record,
            "busi": busi_record,
        },
        "transfer": {
            "busuclm_to_busi": u2b_record,
            "busi_to_busuclm": b2u_record,
        },
        "development_score": float(
            np.mean([uclm_auc, busi_auc, u2b_auc, b2u_auc])
        ),
    }


def run_early(
    name: str,
    group: str,
    frames: dict[str, pd.DataFrame],
    embeddings: dict[str, np.ndarray],
) -> dict[str, object]:
    output = OUTPUT_DIR / name
    output.mkdir(parents=True, exist_ok=True)
    columns = feature_columns(frames["busuclm"], group)
    handcrafted = {
        dataset: frames[dataset][columns].to_numpy(dtype=np.float64)
        for dataset in frames
    }
    uclm_auc, uclm_record = early_within(
        "busuclm",
        frames["busuclm"],
        handcrafted["busuclm"],
        embeddings["busuclm"],
        output,
        repetitions=REPETITIONS,
    )
    busi_auc, busi_record = early_within(
        "busi",
        frames["busi"],
        handcrafted["busi"],
        embeddings["busi"],
        output,
        repetitions=REPETITIONS,
    )
    u2b_auc, u2b_record = early_transfer(
        "busuclm",
        "busi",
        frames["busuclm"],
        handcrafted["busuclm"],
        embeddings["busuclm"],
        frames["busi"],
        handcrafted["busi"],
        embeddings["busi"],
        output,
    )
    b2u_auc, b2u_record = early_transfer(
        "busi",
        "busuclm",
        frames["busi"],
        handcrafted["busi"],
        embeddings["busi"],
        frames["busuclm"],
        handcrafted["busuclm"],
        embeddings["busuclm"],
        output,
    )
    return {
        "configuration": name,
        "feature_group": group,
        "handcrafted_features": len(columns),
        "encoder": ENCODER,
        "view": VIEW,
        "pca_components": 32,
        "within": {
            "busuclm": uclm_record,
            "busi": busi_record,
        },
        "transfer": {
            "busuclm_to_busi": u2b_record,
            "busi_to_busuclm": b2u_record,
        },
        "development_score": float(
            np.mean([uclm_auc, busi_auc, u2b_auc, b2u_auc])
        ),
    }


def add_score(record: dict[str, object]) -> dict[str, object]:
    if "development_score" not in record:
        values = [
            record["within"]["busuclm"]["auc"],
            record["within"]["busi"]["auc"],
            record["transfer"]["busuclm_to_busi"]["auc"],
            record["transfer"]["busi_to_busuclm"]["auc"],
        ]
        record["development_score"] = float(np.mean(values))
    record["minimum_transfer_auc"] = float(
        min(
            record["transfer"]["busuclm_to_busi"]["auc"],
            record["transfer"]["busi_to_busuclm"]["auc"],
        )
    )
    return record


def combine_and_bootstrap(setting: str) -> dict[str, object]:
    if setting in {"busuclm", "busi"}:
        filename = f"{setting}_mean_oof_predictions.csv"
    else:
        filename = f"{setting}_predictions.csv"
    keys = ["image", "patient_id", "label", "target"]
    combined: pd.DataFrame | None = None
    for configuration in CONFIGURATIONS:
        prediction = pd.read_csv(
            OUTPUT_DIR / configuration / filename
        ).rename(columns={"probability": f"probability_{configuration}"})
        keep = keys + [f"probability_{configuration}"]
        if combined is None:
            combined = prediction[keep].copy()
        else:
            combined = combined.merge(
                prediction[keep],
                on=keys,
                how="inner",
                validate="one_to_one",
            )
    if combined is None:
        raise RuntimeError("no predictions to combine")
    expected = len(
        pd.read_csv(OUTPUT_DIR / CONFIGURATIONS[0] / filename)
    )
    if len(combined) != expected:
        raise RuntimeError(f"{setting} confirmation row mismatch")
    intervals, deltas = cluster_bootstrap(
        combined,
        list(CONFIGURATIONS),
        BOOTSTRAP_ITERATIONS,
        REFERENCE,
    )
    combined.to_csv(
        OUTPUT_DIR / f"{setting}_combined_predictions.csv",
        index=False,
    )
    return {
        "setting": setting,
        "rows": int(len(combined)),
        "groups": int(combined["patient_id"].nunique()),
        "reference": REFERENCE,
        "auc_intervals": intervals,
        "paired_deltas": deltas,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frames = {
        dataset: prepare_metadata(dataset)
        for dataset in ("busuclm", "busi")
    }
    for frame in frames.values():
        frame["domain_group"] = frame["patient_id"]
    embeddings = {
        dataset: load_embedding(
            dataset,
            ENCODER,
            VIEW,
            frames[dataset],
        )
        for dataset in frames
    }
    records = [
        run_handcrafted(
            "frozen98_logistic",
            "frozen98",
            frames,
        ),
        run_handcrafted(
            "advanced76_logistic",
            "advanced76",
            frames,
        ),
        run_handcrafted(
            "frozen98_shape_logistic",
            "frozen98_shape",
            frames,
        ),
        run_linear_svm(frames),
        run_embedding(frames, embeddings),
        run_early(
            "advanced76_resnet18_early",
            "advanced76",
            frames,
            embeddings,
        ),
        run_early(
            "frozen98_advanced_resnet18_early",
            "frozen98_advanced",
            frames,
            embeddings,
        ),
        run_early(
            "frozen98_shape_resnet18_early",
            "frozen98_shape",
            frames,
            embeddings,
        ),
    ]
    records = [add_score(record) for record in records]
    ranking = sorted(
        records,
        key=lambda record: (
            -record["development_score"],
            -record["minimum_transfer_auc"],
            record["configuration"],
        ),
    )
    for record in records:
        output = OUTPUT_DIR / record["configuration"]
        (output / "summary.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
    bootstrap = {
        setting: combine_and_bootstrap(setting)
        for setting in (
            "busuclm",
            "busi",
            "busuclm_to_busi",
            "busi_to_busuclm",
        )
    }
    result = {
        "protocol": 'V04_DEVELOPMENT_PROTOCOL_V1',
        "repetitions": REPETITIONS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "reference": REFERENCE,
        "ranking": ranking,
        "bootstrap": bootstrap,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (OUTPUT_DIR / "confirmation_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "rank": rank,
                "configuration": record["configuration"],
                "development_score": record["development_score"],
                "minimum_transfer_auc": record[
                    "minimum_transfer_auc"
                ],
                "busuclm_internal_auc": record["within"]["busuclm"][
                    "auc"
                ],
                "busi_internal_auc": record["within"]["busi"]["auc"],
                "busuclm_to_busi_auc": record["transfer"][
                    "busuclm_to_busi"
                ]["auc"],
                "busi_to_busuclm_auc": record["transfer"][
                    "busi_to_busuclm"
                ]["auc"],
            }
            for rank, record in enumerate(ranking, 1)
        ]
    ).to_csv(OUTPUT_DIR / "development_ranking.csv", index=False)
    print(
        json.dumps(
            {
                "ranking": [
                    {
                        "configuration": record["configuration"],
                        "development_score": record["development_score"],
                        "minimum_transfer_auc": record[
                            "minimum_transfer_auc"
                        ],
                    }
                    for record in ranking
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
