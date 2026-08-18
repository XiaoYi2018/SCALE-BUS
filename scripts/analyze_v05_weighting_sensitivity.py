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

from analyze_v05_marginal_block_ablation import (  # noqa: E402
    corrected_group_bootstrap,
)
from evaluate_v04_locked_external import external_frame  # noqa: E402
from run_busuclm_grouped_cv import (  # noqa: E402
    C_GRID,
    SEED,
    patient_equal_weights,
    valid_group_splits,
)
from screen_v04_embedding import prepare_metadata  # noqa: E402
from screen_v04_handcrafted import feature_columns  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "v05_weighting_sensitivity"
DATASET_ORDER = ("busuclm", "busi", "busbra", "breast")
EXTERNAL_TARGETS = ("busbra", "breast")
MODELS = ("fre98", "gfwb76")
BOOTSTRAP_ITERATIONS = 3000
TUNING_REPETITIONS = 5


def equal_dataset_group_class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equalize dataset-class strata and groups within every stratum."""
    working = frame.copy()
    if "dataset" not in working:
        working["dataset"] = "single"
    dataset = working["dataset"].astype(str)
    group = working["patient_id"].astype(str)
    class_label = working["target"].astype(str)
    composite = dataset + "::" + group + "::class" + class_label
    group_sizes = composite.groupby(composite).transform("size").astype(float)
    stratum = dataset + "::class" + class_label
    stratum_groups = (
        pd.DataFrame({"stratum": stratum, "group": composite})
        .drop_duplicates()
        .groupby("stratum")["group"]
        .count()
    )
    base = 1.0 / group_sizes.to_numpy()
    base /= stratum.map(stratum_groups).to_numpy(dtype=float)
    weights = base / float(np.mean(base))
    audit = pd.DataFrame(
        {
            "stratum": stratum.to_numpy(),
            "group": composite.to_numpy(),
            "weight": weights,
        }
    )
    stratum_totals = audit.groupby("stratum")["weight"].sum().to_numpy()
    if float(np.ptp(stratum_totals)) > 1e-8:
        raise RuntimeError("dataset-class stratum weights are not equal")
    for _, part in audit.groupby("stratum"):
        group_totals = part.groupby("group")["weight"].sum().to_numpy()
        if float(np.ptp(group_totals)) > 1e-8:
            raise RuntimeError("group weights differ within a stratum")
    return weights


def tune_weighted_c(
    source: pd.DataFrame,
    columns: list[str],
    n_splits: int,
) -> tuple[float, float]:
    scores: dict[float, list[float]] = {
        float(value): [] for value in C_GRID
    }
    working = source.copy().reset_index(drop=True)
    working["patient_id"] = working["patient_id"].astype(str)
    for repeat in range(TUNING_REPETITIONS):
        folds, _ = valid_group_splits(
            working,
            n_splits,
            SEED + repeat * 1000,
        )
        for train_idx, valid_idx in folds:
            train = working.iloc[train_idx].copy()
            valid = working.iloc[valid_idx].copy()
            train_weight = equal_dataset_group_class_weights(train)
            valid_weight = equal_dataset_group_class_weights(valid)
            for c_value in C_GRID:
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=float(c_value),
                                class_weight=None,
                                max_iter=5000,
                                random_state=SEED,
                                solver="liblinear",
                            ),
                        ),
                    ]
                )
                model.fit(
                    train[columns],
                    train["target"],
                    model__sample_weight=train_weight,
                )
                probability = model.predict_proba(valid[columns])[:, 1]
                scores[float(c_value)].append(
                    float(
                        roc_auc_score(
                            valid["target"],
                            probability,
                            sample_weight=valid_weight,
                        )
                    )
                )
    mean_scores = {
        c_value: float(np.mean(values)) for c_value, values in scores.items()
    }
    best_c = max(sorted(mean_scores), key=lambda value: mean_scores[value])
    return best_c, mean_scores[best_c]


def build_weighted_model(
    source: pd.DataFrame,
    columns: list[str],
    splits: int,
) -> tuple[Pipeline, float, float]:
    best_c, selected_auc = tune_weighted_c(source, columns, splits)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=best_c,
                    class_weight=None,
                    max_iter=5000,
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(
        source[columns],
        source["target"],
        model__sample_weight=equal_dataset_group_class_weights(source),
    )
    return model, float(best_c), selected_auc


def target_auc(frame: pd.DataFrame, probability: np.ndarray) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            probability,
            sample_weight=patient_equal_weights(frame),
        )
    )


def model_columns(frame: pd.DataFrame, model: str) -> list[str]:
    group = {"fre98": "frozen98", "gfwb76": "advanced76"}[model]
    return feature_columns(frame, group)


def locked_external(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    predictions = {
        target: datasets[target][
            ["image", "patient_id", "label", "target"]
        ].copy()
        for target in EXTERNAL_TARGETS
    }
    for model_name in MODELS:
        columns = model_columns(datasets["busuclm"], model_name)
        source_probabilities: dict[str, dict[str, np.ndarray]] = {}
        cs: dict[str, float] = {}
        for source_name in ("busuclm", "busi"):
            model, best_c, selected_auc = build_weighted_model(
                datasets[source_name],
                columns,
                4 if source_name == "busuclm" else 5,
            )
            cs[source_name] = best_c
            source_probabilities[source_name] = {
                target: model.predict_proba(datasets[target][columns])[:, 1]
                for target in EXTERNAL_TARGETS
            }
            tuning_rows.append(
                {
                    "context": "locked_external",
                    "held_out": "",
                    "source": source_name,
                    "model": model_name,
                    "best_c": best_c,
                    "weighted_source_cv_auc": selected_auc,
                }
            )
        for target in EXTERNAL_TARGETS:
            probability = np.mean(
                [
                    source_probabilities[source][target]
                    for source in ("busuclm", "busi")
                ],
                axis=0,
            )
            predictions[target][f"probability_{model_name}"] = probability
            rows.append(
                {
                    "context": "locked_external",
                    "target": target,
                    "model": model_name,
                    "features": len(columns),
                    "auc": target_auc(datasets[target], probability),
                    "busuclm_best_c": cs["busuclm"],
                    "busi_best_c": cs["busi"],
                }
            )
    return pd.DataFrame(rows), predictions, pd.DataFrame(tuning_rows)


def lodo(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for held_out in DATASET_ORDER:
        target = datasets[held_out]
        prediction_frame = target[
            ["image", "patient_id", "label", "target"]
        ].copy()
        source_names = [name for name in DATASET_ORDER if name != held_out]
        for model_name in MODELS:
            columns = model_columns(datasets["busuclm"], model_name)
            parts = []
            for source_name in source_names:
                part = datasets[source_name][
                    ["image", "patient_id", "label", "target"] + columns
                ].copy()
                part["dataset"] = source_name
                part["patient_id"] = (
                    source_name + ":" + part["patient_id"].astype(str)
                )
                parts.append(part)
            pooled = pd.concat(parts, ignore_index=True)
            model, best_c, selected_auc = build_weighted_model(
                pooled,
                columns,
                5,
            )
            probability = model.predict_proba(target[columns])[:, 1]
            prediction_frame[f"probability_{model_name}"] = probability
            rows.append(
                {
                    "context": "lodo",
                    "held_out": held_out,
                    "model": model_name,
                    "features": len(columns),
                    "auc": target_auc(target, probability),
                    "best_c": best_c,
                }
            )
            tuning_rows.append(
                {
                    "context": "lodo",
                    "held_out": held_out,
                    "source": "+".join(source_names),
                    "model": model_name,
                    "best_c": best_c,
                    "weighted_source_cv_auc": selected_auc,
                }
            )
        predictions[held_out] = prediction_frame
    return pd.DataFrame(rows), predictions, pd.DataFrame(tuning_rows)


def add_bootstrap(
    metrics: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    context: str,
    target_column: str,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for offset, (target, frame) in enumerate(predictions.items()):
        # The helper's full-model constant is gfwb76, which is present here.
        intervals, deltas, valid = corrected_group_bootstrap(
            frame,
            MODELS,
            BOOTSTRAP_ITERATIONS,
            seed_offset + offset,
        )
        point = {
            row.model: float(row.auc)
            for row in metrics.loc[
                metrics[target_column] == target
            ].itertuples(index=False)
        }
        for model, (low, high) in intervals.items():
            interval_rows.append(
                {
                    target_column: target,
                    "model": model,
                    "auc_ci_low": low,
                    "auc_ci_high": high,
                }
            )
        for delta in deltas:
            delta_rows.append(
                {
                    "context": context,
                    target_column: target,
                    **delta,
                    "point_delta": point["gfwb76"] - point["fre98"],
                    "bootstrap_iterations_valid": valid,
                }
            )
    enriched = metrics.merge(
        pd.DataFrame(interval_rows),
        on=[target_column, "model"],
        how="left",
        validate="one_to_one",
    )
    return enriched, pd.DataFrame(delta_rows)


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
    for model_name in MODELS:
        reference = model_columns(datasets["busuclm"], model_name)
        for dataset in DATASET_ORDER:
            if model_columns(datasets[dataset], model_name) != reference:
                raise RuntimeError(
                    f"{dataset}/{model_name}: feature columns differ"
                )

    external_metrics, external_predictions, external_tuning = (
        locked_external(datasets)
    )
    external_metrics, external_deltas = add_bootstrap(
        external_metrics,
        external_predictions,
        "locked_external",
        "target",
        500,
    )
    external_metrics.to_csv(
        OUTPUT_DIR / "weighted_locked_external_metrics.csv",
        index=False,
    )
    for target, frame in external_predictions.items():
        frame.to_csv(
            OUTPUT_DIR / f"weighted_locked_external_{target}_predictions.csv",
            index=False,
        )

    lodo_metrics, lodo_predictions, lodo_tuning = lodo(datasets)
    lodo_metrics, lodo_deltas = add_bootstrap(
        lodo_metrics,
        lodo_predictions,
        "lodo",
        "held_out",
        600,
    )
    lodo_metrics.to_csv(
        OUTPUT_DIR / "weighted_lodo_metrics.csv",
        index=False,
    )
    for target, frame in lodo_predictions.items():
        frame.to_csv(
            OUTPUT_DIR / f"weighted_lodo_{target}_predictions.csv",
            index=False,
        )

    deltas = pd.concat(
        [external_deltas, lodo_deltas],
        ignore_index=True,
        sort=False,
    )
    deltas.to_csv(
        OUTPUT_DIR / "weighted_paired_differences.csv",
        index=False,
    )
    pd.concat(
        [external_tuning, lodo_tuning],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "weighted_model_tuning.csv", index=False)

    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "weighting": (
            "equal total weight per dataset-by-class stratum and equal total "
            "weight per patient/visual group within each stratum"
        ),
        "model_selection_metric": (
            "weighted ROC AUC using the same dataset/group/class weighting"
        ),
        "models": {
            model: len(model_columns(datasets["busuclm"], model))
            for model in MODELS
        },
        "locked_external": external_metrics.to_dict(orient="records"),
        "lodo": lodo_metrics.to_dict(orient="records"),
        "deltas": deltas.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "weighting_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "elapsed_seconds": summary["elapsed_seconds"],
                "locked_external": summary["locked_external"],
                "lodo": summary["lodo"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
