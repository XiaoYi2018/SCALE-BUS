from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fractal_extrema import (  # noqa: E402
    FORMAL_FEATURE_GROUPS,
    MULTIZONE_FEATURE_GROUPS,
    feature_columns_for_group,
)


DEFAULT_FEATURES = ROOT / "results" / "busuclm_features" / "features_all_valid.csv"
SEED = 20260717
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def valid_group_splits(
    frame: pd.DataFrame,
    n_splits: int,
    seed: int,
    max_attempts: int = 200,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    y = frame["target"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    for attempt in range(max_attempts):
        actual_seed = seed + attempt
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=actual_seed,
        )
        splits = list(splitter.split(frame, y, groups))
        if all(
            np.unique(y[train_idx]).size == 2
            and np.unique(y[test_idx]).size == 2
            for train_idx, test_idx in splits
        ):
            return splits, actual_seed
    raise RuntimeError(
        f"could not construct {n_splits} class-valid grouped folds "
        f"after {max_attempts} deterministic attempts"
    )


def patient_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("patient_id")["patient_id"].transform("size")
    return (1.0 / counts).to_numpy(dtype=np.float64)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    binary = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(
        labels,
        binary,
        labels=[0, 1],
        sample_weight=weights,
    ).ravel()
    return {
        "auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, binary, sample_weight=weights)
        ),
        "sensitivity": float(tp / max(tp + fn, np.finfo(float).eps)),
        "specificity": float(tn / max(tn + fp, np.finfo(float).eps)),
        "f1": float(f1_score(labels, binary, sample_weight=weights)),
        "mcc": float(matthews_corrcoef(labels, binary, sample_weight=weights)),
    }


def evaluate_group(
    frame: pd.DataFrame,
    columns: list[str],
    outer_splits: int,
    inner_splits: int,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for repeat in range(repetitions):
        outer, outer_seed = valid_group_splits(
            frame,
            outer_splits,
            SEED + repeat * 1000,
        )
        for fold, (train_idx, test_idx) in enumerate(outer, 1):
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            inner, inner_seed = valid_group_splits(
                train.reset_index(drop=True),
                inner_splits,
                SEED + repeat * 1000 + fold * 100,
            )
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=SEED,
                            solver="liblinear",
                        ),
                    ),
                ]
            )
            search = GridSearchCV(
                estimator=model,
                param_grid={"model__C": list(C_GRID)},
                scoring="roc_auc",
                cv=inner,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )
            search.fit(
                train[columns],
                train["target"],
            )
            probabilities = search.predict_proba(test[columns])[:, 1]
            for row, probability in zip(
                test.itertuples(index=False),
                probabilities,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "image": row.image,
                        "patient_id": row.patient_id,
                        "label": row.label,
                        "target": int(row.target),
                        "probability": float(probability),
                    }
                )
            tuning_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "outer_seed": outer_seed,
                    "inner_seed": inner_seed,
                    "best_c": float(search.best_params_["model__C"]),
                    "inner_auc": float(search.best_score_),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_patients": int(train["patient_id"].nunique()),
                    "test_patients": int(test["patient_id"].nunique()),
                }
            )
            for split, subset in (("train", train), ("test", test)):
                counts = subset["target"].value_counts()
                fold_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "outer_seed": outer_seed,
                        "split": split,
                        "rows": int(len(subset)),
                        "patients": int(subset["patient_id"].nunique()),
                        "benign": int(counts.get(0, 0)),
                        "malignant": int(counts.get(1, 0)),
                        "patient_ids": "|".join(
                            sorted(subset["patient_id"].unique())
                        ),
                    }
                )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(fold_rows).drop_duplicates(),
    )


def cluster_bootstrap(
    mean_predictions: pd.DataFrame,
    groups: list[str],
    iterations: int,
    reference_group: str,
) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(SEED)
    patients = mean_predictions["patient_id"].unique()
    by_patient = {
        patient: mean_predictions.index[
            mean_predictions["patient_id"] == patient
        ].to_numpy()
        for patient in patients
    }
    auc_samples = {group: [] for group in groups}
    delta_samples = {group: [] for group in groups if group != reference_group}

    for _ in range(iterations):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        indices: list[int] = []
        weights: list[float] = []
        for patient in drawn:
            selected = by_patient[patient]
            indices.extend(selected.tolist())
            weights.extend([1.0 / len(selected)] * len(selected))
        boot = mean_predictions.loc[indices]
        labels = boot["target"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        sample_weight = np.asarray(weights)
        current: dict[str, float] = {}
        for group in groups:
            auc = float(
                roc_auc_score(
                    labels,
                    boot[f"probability_{group}"],
                    sample_weight=sample_weight,
                )
            )
            auc_samples[group].append(auc)
            current[group] = auc
        reference = current[reference_group]
        for group in delta_samples:
            delta_samples[group].append(current[group] - reference)

    intervals = {
        group: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for group, values in auc_samples.items()
    }
    deltas = {
        group: {
            "mean": float(np.mean(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "probability_positive": float(np.mean(np.asarray(values) > 0)),
        }
        for group, values in delta_samples.items()
    }
    return intervals, deltas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--cohort",
        choices=("primary", "sensitivity", "external_busi"),
        required=True,
    )
    parser.add_argument(
        "--experiment",
        choices=("formal", "multizone"),
        default="formal",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--reference-group")
    parser.add_argument("--candidate-group")
    args = parser.parse_args()

    frame = pd.read_csv(args.features.resolve())
    frame["target"] = (frame["label"] == "malignant").astype(np.int64)
    if args.cohort == "primary":
        frame = frame.loc[parse_bool(frame["is_clean_primary"])].copy()
        outer_splits, inner_splits, repetitions = 4, 3, 10
    elif args.cohort == "sensitivity":
        outer_splits, inner_splits, repetitions = 5, 4, 5
    else:
        outer_splits, inner_splits, repetitions = 5, 4, 10
    if args.experiment == "formal":
        experiment_groups = FORMAL_FEATURE_GROUPS
        reference_group = "basic_fractal"
        candidate_group = "fused_compact"
    else:
        experiment_groups = MULTIZONE_FEATURE_GROUPS
        reference_group = "basic"
        candidate_group = "basic_multizone_extrema"
    if args.reference_group:
        reference_group = args.reference_group
    if args.candidate_group:
        candidate_group = args.candidate_group
    frame = frame.sort_values(["patient_id", "image"]).reset_index(drop=True)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else (
            ROOT
            / "results"
            / (
                f"busi_cv_{args.experiment}"
                if args.cohort == "external_busi"
                else (
                    f"busuclm_cv_{args.cohort}"
                    if args.experiment == "formal"
                    else f"busuclm_cv_{args.cohort}_multizone"
                )
            )
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    all_tuning: list[pd.DataFrame] = []
    shared_folds: pd.DataFrame | None = None
    feature_counts: dict[str, int] = {}

    for group in experiment_groups:
        columns = feature_columns_for_group(frame.columns, group)
        feature_counts[group] = len(columns)
        predictions, tuning, folds = evaluate_group(
            frame,
            columns,
            outer_splits,
            inner_splits,
            repetitions,
        )
        predictions.insert(0, "group", group)
        tuning.insert(0, "group", group)
        all_predictions.append(predictions)
        all_tuning.append(tuning)
        if shared_folds is None:
            shared_folds = folds
        print(
            f"{args.cohort} {group}: {len(columns)} features, "
            f"{len(predictions)} predictions",
            flush=True,
        )

    prediction_table = pd.concat(all_predictions, ignore_index=True)
    tuning_table = pd.concat(all_tuning, ignore_index=True)
    prediction_table.to_csv(out_dir / "oof_predictions.csv", index=False)
    tuning_table.to_csv(out_dir / "tuning.csv", index=False)
    assert shared_folds is not None
    shared_folds.to_csv(out_dir / "fold_assignments.csv", index=False)

    repeat_metrics: list[dict[str, object]] = []
    for (group, repeat), subset in prediction_table.groupby(["group", "repeat"]):
        metrics = classification_metrics(
            subset["target"].to_numpy(),
            subset["probability"].to_numpy(),
        )
        metrics.update({"group": group, "repeat": int(repeat)})
        repeat_metrics.append(metrics)
    repeat_frame = pd.DataFrame(repeat_metrics)
    repeat_frame.to_csv(out_dir / "repeat_metrics.csv", index=False)

    means = (
        prediction_table.groupby(
            ["group", "image", "patient_id", "label", "target"],
            as_index=False,
        )["probability"]
        .mean()
    )
    mean_wide = means.pivot(
        index=["image", "patient_id", "label", "target"],
        columns="group",
        values="probability",
    ).reset_index()
    mean_wide.columns.name = None
    mean_wide = mean_wide.rename(
        columns={group: f"probability_{group}" for group in experiment_groups}
    )
    mean_wide.to_csv(out_dir / "mean_oof_predictions.csv", index=False)

    intervals, deltas = cluster_bootstrap(
        mean_wide,
        list(experiment_groups),
        args.bootstrap_iterations,
        reference_group,
    )
    summary_rows: list[dict[str, object]] = []
    labels = mean_wide["target"].to_numpy()
    weights = patient_equal_weights(mean_wide)
    for group in experiment_groups:
        unweighted = classification_metrics(
            labels,
            mean_wide[f"probability_{group}"].to_numpy(),
        )
        patient_balanced = classification_metrics(
            labels,
            mean_wide[f"probability_{group}"].to_numpy(),
            weights=weights,
        )
        repeated = repeat_frame.loc[repeat_frame["group"] == group, "auc"]
        row: dict[str, object] = {
            "group": group,
            "n_features": feature_counts[group],
            **{f"image_{key}": value for key, value in unweighted.items()},
            **{
                f"patient_balanced_{key}": value
                for key, value in patient_balanced.items()
            },
            "repeat_auc_mean": float(repeated.mean()),
            "repeat_auc_std": float(repeated.std(ddof=1)),
            "patient_balanced_auc_ci_low": intervals[group][0],
            "patient_balanced_auc_ci_high": intervals[group][1],
        }
        if group in deltas:
            row.update(
                {
                    "reference_group": reference_group,
                    "delta_vs_reference_mean": deltas[group]["mean"],
                    "delta_vs_reference_ci_low": deltas[group]["ci_low"],
                    "delta_vs_reference_ci_high": deltas[group]["ci_high"],
                    "delta_vs_reference_probability_positive": deltas[group][
                        "probability_positive"
                    ],
                }
            )
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out_dir / "cv_summary.csv", index=False)

    result = {
        "cohort": args.cohort,
        "experiment": args.experiment,
        "features_file": str(args.features.resolve()),
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "grouping_unit": (
            "near_duplicate_visual_group"
            if args.cohort == "external_busi"
            else "patient"
        ),
        "class_counts": {
            key: int(value) for key, value in frame["label"].value_counts().items()
        },
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "repetitions": repetitions,
        "seed": SEED,
        "c_grid": list(C_GRID),
        "bootstrap_iterations": args.bootstrap_iterations,
        "summary": summary_rows,
        "primary_comparison": {
            "reference": reference_group,
            "candidate": candidate_group,
            **deltas[candidate_group],
        },
    }
    (out_dir / "cv_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["primary_comparison"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
