from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_busuclm_grouped_cv import (  # noqa: E402
    C_GRID,
    SEED,
    patient_equal_weights,
    valid_group_splits,
)
from screen_v04_handcrafted import feature_columns, prepare  # noqa: E402


INPUT_DIR = ROOT / "results" / "v04_advanced_features"
OUTPUT_DIR = ROOT / "results" / "v04_classifier_screen"
CLASSIFIERS = (
    "logistic",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "gradient_boosting",
    "adaboost",
)
FEATURE_GROUP = "advanced76"


def parameter_grid(classifier: str) -> list[dict[str, object]]:
    if classifier in {"logistic", "linear_svm"}:
        return [{"c": float(c)} for c in C_GRID]
    if classifier == "rbf_svm":
        return [
            {"c": c, "gamma": gamma}
            for c, gamma in itertools.product(
                (0.1, 1.0, 10.0, 100.0),
                ("scale", 0.01, 0.1),
            )
        ]
    if classifier in {"random_forest", "extra_trees"}:
        return [
            {
                "max_features": max_features,
                "min_samples_leaf": min_samples_leaf,
                "max_depth": max_depth,
            }
            for max_features, min_samples_leaf, max_depth in itertools.product(
                ("sqrt", 0.5),
                (1, 3),
                (None, 8),
            )
        ]
    if classifier == "hist_gradient_boosting":
        return [
            {
                "learning_rate": learning_rate,
                "max_leaf_nodes": max_leaf_nodes,
                "l2_regularization": l2,
            }
            for learning_rate, max_leaf_nodes, l2 in itertools.product(
                (0.03, 0.1),
                (7, 15),
                (0.0, 1.0, 10.0),
            )
        ]
    if classifier == "gradient_boosting":
        return [
            {
                "learning_rate": learning_rate,
                "n_estimators": estimators,
                "max_depth": depth,
            }
            for learning_rate, estimators, depth in itertools.product(
                (0.03, 0.1),
                (100, 300),
                (1, 2),
            )
        ]
    if classifier == "adaboost":
        return [
            {"learning_rate": learning_rate, "n_estimators": estimators}
            for learning_rate, estimators in itertools.product(
                (0.03, 0.1, 0.5),
                (100, 300),
            )
        ]
    raise ValueError(classifier)


def make_model(
    classifier: str,
    parameters: dict[str, object],
    seed: int,
):
    if classifier == "logistic":
        return LogisticRegression(
            C=float(parameters["c"]),
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            solver="liblinear",
        )
    if classifier == "linear_svm":
        return SVC(
            C=float(parameters["c"]),
            kernel="linear",
            class_weight="balanced",
            probability=False,
            random_state=seed,
        )
    if classifier == "rbf_svm":
        return SVC(
            C=float(parameters["c"]),
            gamma=parameters["gamma"],
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=seed,
        )
    if classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features=parameters["max_features"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_depth=parameters["max_depth"],
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )
    if classifier == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features=parameters["max_features"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_depth=parameters["max_depth"],
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    if classifier == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            l2_regularization=float(parameters["l2_regularization"]),
            max_iter=300,
            early_stopping=False,
            random_state=seed,
        )
    if classifier == "gradient_boosting":
        return GradientBoostingClassifier(
            learning_rate=float(parameters["learning_rate"]),
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            random_state=seed,
        )
    if classifier == "adaboost":
        return AdaBoostClassifier(
            learning_rate=float(parameters["learning_rate"]),
            n_estimators=int(parameters["n_estimators"]),
            random_state=seed,
        )
    raise ValueError(classifier)


def needs_sample_weight(classifier: str) -> bool:
    return classifier in {
        "hist_gradient_boosting",
        "gradient_boosting",
        "adaboost",
    }


def score_values(model, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values)[:, 1], dtype=np.float64)
    return np.asarray(model.decision_function(values), dtype=np.float64)


def fit_model(
    classifier: str,
    parameters: dict[str, object],
    train_values: np.ndarray,
    train_target: np.ndarray,
    seed: int,
):
    model = make_model(classifier, parameters, seed)
    if needs_sample_weight(classifier):
        weights = compute_sample_weight("balanced", train_target)
        model.fit(train_values, train_target, sample_weight=weights)
    else:
        model.fit(train_values, train_target)
    return model


def tune(
    classifier: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    splits: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    folds, split_seed = valid_group_splits(frame, splits, seed)
    grid = parameter_grid(classifier)
    scores: list[list[float]] = [[] for _ in grid]
    for fold_number, (train_index, validation_index) in enumerate(folds, 1):
        scaler = StandardScaler()
        train_values = scaler.fit_transform(values[train_index])
        validation_values = scaler.transform(values[validation_index])
        train_target = frame.iloc[train_index]["target"].to_numpy()
        validation_target = frame.iloc[validation_index]["target"].to_numpy()
        for parameter_index, parameters in enumerate(grid):
            model = fit_model(
                classifier,
                parameters,
                train_values,
                train_target,
                seed + fold_number,
            )
            probability = score_values(model, validation_values)
            scores[parameter_index].append(
                float(roc_auc_score(validation_target, probability))
            )
    rows = [
        {
            "parameters": json.dumps(parameters, sort_keys=True),
            "auc": float(np.mean(parameter_scores)),
            "split_seed": int(split_seed),
        }
        for parameters, parameter_scores in zip(grid, scores, strict=True)
    ]
    best_index = sorted(
        range(len(grid)),
        key=lambda index: (
            -rows[index]["auc"],
            rows[index]["parameters"],
        ),
    )[0]
    return grid[best_index], rows


def tune_repeated(
    classifier: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    splits: int,
    seed: int,
    repetitions: int = 5,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for repeat in range(repetitions):
        _, repeat_rows = tune(
            classifier,
            frame,
            values,
            splits,
            seed + repeat * 1000,
        )
        for row in repeat_rows:
            rows.append({"repeat": repeat + 1, **row})
    means: dict[str, list[float]] = {}
    for row in rows:
        key = str(row["parameters"])
        means.setdefault(key, []).append(float(row["auc"]))
    ranked = sorted(
        means,
        key=lambda key: (-float(np.mean(means[key])), key),
    )
    return json.loads(ranked[0]), rows


def fit_predict(
    classifier: str,
    parameters: dict[str, object],
    train_values: np.ndarray,
    train_target: np.ndarray,
    test_values: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_values)
    scaled_test = scaler.transform(test_values)
    model = fit_model(
        classifier,
        parameters,
        scaled_train,
        train_target,
        seed,
    )
    return score_values(model, scaled_test)


def within(
    classifier: str,
    dataset: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    output: Path,
    repetitions: int = 3,
) -> tuple[float, dict[str, object]]:
    outer_splits, inner_splits = (
        (5, 4) if dataset == "busi" else (4, 3)
    )
    predictions: list[dict[str, object]] = []
    tuning_records: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    overlap = 0
    for repeat in range(repetitions):
        folds, outer_seed = valid_group_splits(
            frame,
            outer_splits,
            SEED + repeat * 1000,
        )
        for fold_number, (train_index, test_index) in enumerate(folds, 1):
            train = frame.iloc[train_index].reset_index(drop=True)
            parameters, tuning = tune(
                classifier,
                train,
                values[train_index],
                inner_splits,
                SEED + repeat * 1000 + fold_number * 100,
            )
            score = fit_predict(
                classifier,
                parameters,
                values[train_index],
                train["target"].to_numpy(),
                values[test_index],
                SEED + repeat * 1000 + fold_number,
            )
            test = frame.iloc[test_index]
            for record, value in zip(
                test.itertuples(index=False),
                score,
                strict=True,
            ):
                predictions.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold_number,
                        "image": record.image,
                        "patient_id": record.patient_id,
                        "label": record.label,
                        "target": int(record.target),
                        "probability": float(value),
                    }
                )
            train_ids = set(train["patient_id"])
            test_ids = set(test["patient_id"])
            overlap += len(train_ids & test_ids)
            fold_records.extend(
                [
                    {
                        "repeat": repeat + 1,
                        "fold": fold_number,
                        "outer_seed": outer_seed,
                        "split": "train",
                        "rows": int(len(train)),
                        "groups": int(train["patient_id"].nunique()),
                        "patient_ids": "|".join(sorted(train_ids)),
                    },
                    {
                        "repeat": repeat + 1,
                        "fold": fold_number,
                        "outer_seed": outer_seed,
                        "split": "test",
                        "rows": int(len(test)),
                        "groups": int(test["patient_id"].nunique()),
                        "patient_ids": "|".join(sorted(test_ids)),
                    },
                ]
            )
            tuning_records.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold_number,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "inner_auc": max(item["auc"] for item in tuning),
                }
            )
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(
        output / f"{dataset}_oof_predictions.csv",
        index=False,
    )
    pd.DataFrame(tuning_records).to_csv(
        output / f"{dataset}_tuning.csv",
        index=False,
    )
    pd.DataFrame(fold_records).drop_duplicates().to_csv(
        output / f"{dataset}_fold_assignments.csv",
        index=False,
    )
    means = (
        prediction_frame.groupby(
            ["image", "patient_id", "label", "target"],
            as_index=False,
        )["probability"]
        .mean()
    )
    means.to_csv(
        output / f"{dataset}_mean_oof_predictions.csv",
        index=False,
    )
    auc = float(
        roc_auc_score(
            means["target"],
            means["probability"],
            sample_weight=patient_equal_weights(means),
        )
    )
    return auc, {
        "auc": auc,
        "rows": int(len(frame)),
        "groups": int(frame["patient_id"].nunique()),
        "repetitions": repetitions,
        "fold_group_overlaps": int(overlap),
    }


def transfer(
    classifier: str,
    source_name: str,
    target_name: str,
    source: pd.DataFrame,
    source_values: np.ndarray,
    target: pd.DataFrame,
    target_values: np.ndarray,
    output: Path,
) -> tuple[float, dict[str, object]]:
    source_splits = 5 if source_name == "busi" else 4
    parameters, tuning = tune_repeated(
        classifier,
        source,
        source_values,
        source_splits,
        SEED,
    )
    score = fit_predict(
        classifier,
        parameters,
        source_values,
        source["target"].to_numpy(),
        target_values,
        SEED,
    )
    prediction = target[
        ["image", "patient_id", "label", "target"]
    ].copy()
    prediction["probability"] = score
    direction = f"{source_name}_to_{target_name}"
    prediction.to_csv(
        output / f"{direction}_predictions.csv",
        index=False,
    )
    pd.DataFrame(tuning).to_csv(
        output / f"{direction}_tuning.csv",
        index=False,
    )
    auc = float(
        roc_auc_score(
            prediction["target"],
            score,
            sample_weight=patient_equal_weights(prediction),
        )
    )
    return auc, {
        "source": source_name,
        "target": target_name,
        "auc": auc,
        "parameters": parameters,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=CLASSIFIERS, required=True)
    args = parser.parse_args()
    output = OUTPUT_DIR / f"{FEATURE_GROUP}_{args.classifier}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frames = {
        "busuclm": prepare(
            pd.read_csv(
                INPUT_DIR / "features_busuclm_advanced.csv"
            ),
            "busuclm",
        ),
        "busi": prepare(
            pd.read_csv(INPUT_DIR / "features_busi_advanced.csv"),
            "busi",
        ),
    }
    columns = feature_columns(frames["busuclm"], FEATURE_GROUP)
    if columns != feature_columns(frames["busi"], FEATURE_GROUP):
        raise RuntimeError("feature columns differ between cohorts")
    values = {
        dataset: frames[dataset][columns].to_numpy(dtype=np.float64)
        for dataset in frames
    }
    uclm_internal, uclm_record = within(
        args.classifier,
        "busuclm",
        frames["busuclm"],
        values["busuclm"],
        output,
    )
    busi_internal, busi_record = within(
        args.classifier,
        "busi",
        frames["busi"],
        values["busi"],
        output,
    )
    uclm_to_busi, u2b_record = transfer(
        args.classifier,
        "busuclm",
        "busi",
        frames["busuclm"],
        values["busuclm"],
        frames["busi"],
        values["busi"],
        output,
    )
    busi_to_uclm, b2u_record = transfer(
        args.classifier,
        "busi",
        "busuclm",
        frames["busi"],
        values["busi"],
        frames["busuclm"],
        values["busuclm"],
        output,
    )
    result = {
        "protocol": 'V04_DEVELOPMENT_PROTOCOL_V1',
        "feature_group": FEATURE_GROUP,
        "n_features": len(columns),
        "classifier": args.classifier,
        "grid": parameter_grid(args.classifier),
        "within": {
            "busuclm": uclm_record,
            "busi": busi_record,
        },
        "transfer": {
            "busuclm_to_busi": u2b_record,
            "busi_to_busuclm": b2u_record,
        },
        "development_score": float(
            np.mean(
                [
                    uclm_internal,
                    busi_internal,
                    uclm_to_busi,
                    busi_to_uclm,
                ]
            )
        ),
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
