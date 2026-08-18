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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_busuclm_grouped_cv import (  # noqa: E402
    SEED,
    patient_equal_weights,
)
from run_cross_dataset_transfer import source_tune_c  # noqa: E402
from screen_v04_classifiers import (  # noqa: E402
    fit_predict as classifier_fit_predict,
    tune_repeated as classifier_tune_repeated,
)
from screen_v04_early_fusion import (  # noqa: E402
    fit_predict as early_fit_predict,
    tune_c as early_tune_c,
)
from screen_v04_embedding import (  # noqa: E402
    fit_predict as embedding_fit_predict,
    load_embedding,
    prepare_metadata,
    tune_c as embedding_tune_c,
)
from screen_v04_handcrafted import feature_columns  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "v04_locked_external"
EXTERNAL_FEATURE_DIR = ROOT / "results" / "v04_external_features"
EMBEDDING_DIR = ROOT / "results" / "v04_deep_embeddings"
CONFIRMATION_DIR = ROOT / "results" / "v04_confirmation"
BOOTSTRAP_ITERATIONS = 5000
METHODS = (
    "frozen98_logistic",
    "advanced76_logistic",
    "frozen98_shape_logistic",
    "advanced76_linear_svm",
    "resnet18_inner_only",
    "advanced76_resnet18_early",
)
REFERENCE = "frozen98_logistic"
PRIMARY = "advanced76_logistic"
EARLY = "advanced76_resnet18_early"
SOURCES = ("busuclm", "busi")
TARGETS = ("busbra", "breast")


def external_frame(dataset: str) -> pd.DataFrame:
    advanced = pd.read_csv(
        EXTERNAL_FEATURE_DIR / f"features_{dataset}_advanced.csv"
    )
    frozen = pd.read_csv(
        EXTERNAL_FEATURE_DIR / f"features_{dataset}_frozen.csv"
    )
    keys = ["image", "patient_id", "label"]
    frozen_columns = [
        column
        for column in frozen
        if column.startswith(("basic_", "fractal_", "zone_"))
    ]
    frame = advanced.merge(
        frozen[keys + frozen_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(advanced) or len(frame) != len(frozen):
        raise RuntimeError(f"{dataset}: external feature merge failed")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["target"] = (frame["label"] == "malignant").astype(np.int64)
    frame["domain_group"] = frame["patient_id"]
    frame["original_order"] = np.arange(len(frame))
    return frame.sort_values(["patient_id", "image"]).reset_index(drop=True)


def load_external_embedding(
    dataset: str,
    frame: pd.DataFrame,
) -> np.ndarray:
    path = EMBEDDING_DIR / f"{dataset}_resnet18_inner_only.npz"
    bundle = np.load(path, allow_pickle=True)
    embedding = np.asarray(bundle["embedding"], dtype=np.float64)
    image = bundle["image"].astype(str)
    original = frame.sort_values("original_order")
    if image.tolist() != original["image"].astype(str).tolist():
        raise RuntimeError(f"{dataset}: external embedding order mismatch")
    embedding = embedding[
        frame["original_order"].to_numpy(dtype=np.int64)
    ]
    if embedding.shape != (len(frame), 512):
        raise RuntimeError(f"{dataset}: invalid embedding shape")
    if not np.isfinite(embedding).all():
        raise RuntimeError(f"{dataset}: non-finite embedding")
    return embedding


def source_oof(method: str, source: str) -> pd.DataFrame:
    path = (
        CONFIRMATION_DIR
        / method
        / f"{source}_mean_oof_predictions.csv"
    )
    frame = pd.read_csv(path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame


def fit_platt(oof: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression(
        C=1e6,
        max_iter=5000,
        random_state=SEED,
        solver="lbfgs",
    )
    model.fit(
        oof[["probability"]],
        oof["target"],
        sample_weight=patient_equal_weights(oof),
    )
    return model


def youden_threshold(frame: pd.DataFrame) -> float:
    weights = patient_equal_weights(frame)
    false_positive, true_positive, thresholds = roc_curve(
        frame["target"],
        frame["probability"],
        sample_weight=weights,
    )
    valid = np.isfinite(thresholds)
    if not valid.any():
        return 0.5
    indices = np.flatnonzero(valid)
    best = indices[
        int(np.argmax((true_positive - false_positive)[valid]))
    ]
    return float(np.clip(thresholds[best], 0.0, 1.0))


def build_logistic(
    source: pd.DataFrame,
    columns: list[str],
) -> tuple[Pipeline, float, list[dict[str, object]]]:
    working = source.copy()
    working["domain_group"] = working["patient_id"]
    best_c, tuning = source_tune_c(
        working,
        columns,
        5 if len(source) > 200 else 4,
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
    return model, float(best_c), tuning


def predict_method(
    method: str,
    source_name: str,
    source: pd.DataFrame,
    source_embedding: np.ndarray,
    targets: dict[str, pd.DataFrame],
    target_embeddings: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object], float]:
    combined_target = pd.concat(
        [targets[dataset] for dataset in TARGETS],
        ignore_index=True,
    )
    lengths = [len(targets[dataset]) for dataset in TARGETS]
    split_at = lengths[0]
    tuning_record: dict[str, object]

    if method in {
        "frozen98_logistic",
        "advanced76_logistic",
        "frozen98_shape_logistic",
    }:
        group = {
            "frozen98_logistic": "frozen98",
            "advanced76_logistic": "advanced76",
            "frozen98_shape_logistic": "frozen98_shape",
        }[method]
        columns = feature_columns(source, group)
        if columns != feature_columns(combined_target, group):
            raise RuntimeError(f"{method}: source-target columns differ")
        model, best_c, tuning = build_logistic(source, columns)
        raw = model.predict_proba(combined_target[columns])[:, 1]
        tuning_record = {
            "family": "logistic",
            "group": group,
            "features": len(columns),
            "best_c": best_c,
            "source_tuning": tuning,
        }
    elif method == "advanced76_linear_svm":
        columns = feature_columns(source, "advanced76")
        parameters, tuning = classifier_tune_repeated(
            "linear_svm",
            source,
            source[columns].to_numpy(dtype=np.float64),
            5 if source_name == "busi" else 4,
            SEED,
        )
        raw_score = classifier_fit_predict(
            "linear_svm",
            parameters,
            source[columns].to_numpy(dtype=np.float64),
            source["target"].to_numpy(),
            combined_target[columns].to_numpy(dtype=np.float64),
            SEED,
        )
        calibrator = fit_platt(source_oof(method, source_name))
        raw = calibrator.predict_proba(raw_score.reshape(-1, 1))[:, 1]
        tuning_record = {
            "family": "linear_svm_source_oof_platt",
            "group": "advanced76",
            "features": len(columns),
            "parameters": parameters,
            "source_tuning": tuning,
            "platt_intercept": float(calibrator.intercept_[0]),
            "platt_slope": float(calibrator.coef_[0, 0]),
        }
    elif method == "resnet18_inner_only":
        combined_embedding = np.concatenate(
            [target_embeddings[dataset] for dataset in TARGETS],
            axis=0,
        )
        best_c, tuning = embedding_tune_c(
            source,
            source_embedding,
            5 if source_name == "busi" else 4,
            SEED,
        )
        raw = embedding_fit_predict(
            source,
            source_embedding,
            combined_embedding,
            best_c,
            SEED,
        )
        tuning_record = {
            "family": "resnet18_inner_only_pca32_logistic",
            "features": 32,
            "best_c": best_c,
            "source_tuning": tuning,
        }
    elif method == "advanced76_resnet18_early":
        columns = feature_columns(source, "advanced76")
        source_handcrafted = source[columns].to_numpy(dtype=np.float64)
        target_handcrafted = combined_target[columns].to_numpy(
            dtype=np.float64
        )
        combined_embedding = np.concatenate(
            [target_embeddings[dataset] for dataset in TARGETS],
            axis=0,
        )
        best_c, tuning = early_tune_c(
            source,
            source_handcrafted,
            source_embedding,
            5 if source_name == "busi" else 4,
            SEED,
        )
        raw = early_fit_predict(
            source,
            source_handcrafted,
            source_embedding,
            target_handcrafted,
            combined_embedding,
            best_c,
            SEED,
        )
        tuning_record = {
            "family": "advanced76_resnet18_pca32_early_logistic",
            "features": 108,
            "best_c": best_c,
            "source_tuning": tuning,
        }
    else:
        raise ValueError(method)

    oof = source_oof(method, source_name)
    if method == "advanced76_linear_svm":
        calibrator = fit_platt(oof)
        calibrated_oof = oof.copy()
        calibrated_oof["probability"] = calibrator.predict_proba(
            oof[["probability"]]
        )[:, 1]
        threshold = youden_threshold(calibrated_oof)
    else:
        threshold = youden_threshold(oof)
    predictions = {
        TARGETS[0]: raw[:split_at],
        TARGETS[1]: raw[split_at:],
    }
    return predictions, tuning_record, threshold


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (
                (probabilities >= edges[index])
                & (probabilities <= edges[index + 1])
            )
        else:
            selected = (
                (probabilities >= edges[index])
                & (probabilities < edges[index + 1])
            )
        if not selected.any():
            continue
        current_weight = weights[selected]
        fraction = float(current_weight.sum()) / total
        observed = float(
            np.average(labels[selected], weights=current_weight)
        )
        predicted = float(
            np.average(probabilities[selected], weights=current_weight)
        )
        value += fraction * abs(observed - predicted)
    return float(value)


def metrics(
    frame: pd.DataFrame,
    probability_column: str,
    threshold: float,
) -> dict[str, float]:
    labels = frame["target"].to_numpy(dtype=np.int64)
    probabilities = frame[probability_column].to_numpy(dtype=np.float64)
    weights = patient_equal_weights(frame)
    binary = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(
        labels,
        binary,
        labels=[0, 1],
        sample_weight=weights,
    ).ravel()
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(
        C=1e6,
        max_iter=5000,
        random_state=SEED,
        solver="lbfgs",
    )
    calibration.fit(logit, labels, sample_weight=weights)
    return {
        "group_balanced_roc_auc": float(
            roc_auc_score(labels, probabilities, sample_weight=weights)
        ),
        "image_roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(
                labels,
                probabilities,
                sample_weight=weights,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                binary,
                sample_weight=weights,
            )
        ),
        "sensitivity": float(tp / max(tp + fn, np.finfo(float).eps)),
        "specificity": float(tn / max(tn + fp, np.finfo(float).eps)),
        "f1": float(f1_score(labels, binary, sample_weight=weights)),
        "mcc": float(
            matthews_corrcoef(
                labels,
                binary,
                sample_weight=weights,
            )
        ),
        "brier": float(
            np.average(np.square(probabilities - labels), weights=weights)
        ),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "ece_10": expected_calibration_error(
            labels,
            probabilities,
            weights,
        ),
        "threshold": float(threshold),
    }


def bootstrap(
    frame: pd.DataFrame,
    probability_columns: dict[str, str],
    reference: str,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    patients = frame["patient_id"].unique()
    by_patient = {
        patient: frame.index[
            frame["patient_id"] == patient
        ].to_numpy()
        for patient in patients
    }
    samples = {method: [] for method in probability_columns}
    deltas = {
        method: []
        for method in probability_columns
        if method != reference
    }
    for _ in range(BOOTSTRAP_ITERATIONS):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        indices: list[int] = []
        weights: list[float] = []
        for patient in drawn:
            selected = by_patient[patient]
            indices.extend(selected.tolist())
            weights.extend([1.0 / len(selected)] * len(selected))
        boot = frame.loc[indices]
        labels = boot["target"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        sample_weight = np.asarray(weights)
        current = {}
        for method, column in probability_columns.items():
            value = float(
                roc_auc_score(
                    labels,
                    boot[column],
                    sample_weight=sample_weight,
                )
            )
            samples[method].append(value)
            current[method] = value
        for method in deltas:
            deltas[method].append(
                current[method] - current[reference]
            )
    return {
        "auc_intervals": {
            method: {
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975)),
            }
            for method, values in samples.items()
        },
        "paired_deltas": {
            method: {
                "mean": float(np.mean(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "probability_positive": float(
                    np.mean(np.asarray(values) > 0)
                ),
                "two_sided_p": float(
                    min(
                        1.0,
                        2.0
                        * min(
                            np.mean(np.asarray(values) <= 0),
                            np.mean(np.asarray(values) >= 0),
                        ),
                    )
                ),
            }
            for method, values in deltas.items()
        },
        "iterations_requested": BOOTSTRAP_ITERATIONS,
        "iterations_valid": int(len(next(iter(samples.values())))),
    }


def benjamini_hochberg(
    records: list[tuple[str, str, float]],
) -> dict[str, float]:
    ordered = sorted(records, key=lambda item: item[2])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        dataset, method, value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, value * count / rank)
        adjusted[f"{dataset}:{method}"] = float(min(1.0, running))
    return adjusted


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sources = {
        dataset: prepare_metadata(dataset)
        for dataset in SOURCES
    }
    for frame in sources.values():
        frame["domain_group"] = frame["patient_id"]
    targets = {
        dataset: external_frame(dataset)
        for dataset in TARGETS
    }
    source_embeddings = {
        dataset: load_embedding(
            dataset,
            "resnet18",
            "inner_only",
            sources[dataset],
        )
        for dataset in SOURCES
    }
    target_embeddings = {
        dataset: load_external_embedding(dataset, targets[dataset])
        for dataset in TARGETS
    }

    predictions = {
        dataset: targets[dataset][
            [
                "image",
                "patient_id",
                "label",
                "target",
                "birads",
                "device",
            ]
        ].copy()
        for dataset in TARGETS
    }
    tuning: dict[str, dict[str, object]] = {}
    thresholds: dict[str, dict[str, float]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        tuning[method] = {}
        for source_name in SOURCES:
            predicted, tuning_record, threshold = predict_method(
                method,
                source_name,
                sources[source_name],
                source_embeddings[source_name],
                targets,
                target_embeddings,
            )
            tuning[method][source_name] = tuning_record
            thresholds[method][source_name] = threshold
            for target_name in TARGETS:
                predictions[target_name][
                    f"probability_{method}_{source_name}"
                ] = predicted[target_name]
        for target_name in TARGETS:
            predictions[target_name][
                f"probability_{method}_ensemble"
            ] = np.mean(
                [
                    predictions[target_name][
                        f"probability_{method}_{source_name}"
                    ].to_numpy(dtype=np.float64)
                    for source_name in SOURCES
                ],
                axis=0,
            )
        thresholds[method]["ensemble"] = float(
            np.mean(
                [thresholds[method][source] for source in SOURCES]
            )
        )

    metric_summary: dict[str, dict[str, object]] = {}
    bootstrap_reference: dict[str, object] = {}
    bootstrap_early: dict[str, object] = {}
    secondary_p_values: list[tuple[str, str, float]] = []
    for dataset in TARGETS:
        frame = predictions[dataset]
        frame.to_csv(
            OUTPUT_DIR / f"{dataset}_locked_predictions.csv",
            index=False,
        )
        metric_summary[dataset] = {}
        for method in METHODS:
            metric_summary[dataset][method] = {
                source: metrics(
                    frame,
                    f"probability_{method}_{source}",
                    thresholds[method][source],
                )
                for source in (*SOURCES, "ensemble")
            }
        columns = {
            method: f"probability_{method}_ensemble"
            for method in METHODS
        }
        bootstrap_reference[dataset] = bootstrap(
            frame,
            columns,
            REFERENCE,
            SEED + (0 if dataset == "busbra" else 10000),
        )
        bootstrap_early[dataset] = bootstrap(
            frame,
            {
                PRIMARY: columns[PRIMARY],
                EARLY: columns[EARLY],
            },
            PRIMARY,
            SEED + (20000 if dataset == "busbra" else 30000),
        )
        for method, record in bootstrap_reference[dataset][
            "paired_deltas"
        ].items():
            if method != PRIMARY:
                secondary_p_values.append(
                    (dataset, method, record["two_sided_p"])
                )
    adjusted_p = benjamini_hochberg(secondary_p_values)
    for dataset in TARGETS:
        for method, record in bootstrap_reference[dataset][
            "paired_deltas"
        ].items():
            key = f"{dataset}:{method}"
            if key in adjusted_p:
                record["bh_adjusted_p_secondary_family"] = adjusted_p[key]

    primary_auc = {
        dataset: metric_summary[dataset][PRIMARY]["ensemble"][
            "group_balanced_roc_auc"
        ]
        for dataset in TARGETS
    }
    reference_auc = {
        dataset: metric_summary[dataset][REFERENCE]["ensemble"][
            "group_balanced_roc_auc"
        ]
        for dataset in TARGETS
    }
    primary_delta = {
        dataset: bootstrap_reference[dataset]["paired_deltas"][PRIMARY]
        for dataset in TARGETS
    }
    primary_supported = (
        all(
            primary_auc[dataset] >= reference_auc[dataset]
            for dataset in TARGETS
        )
        and any(
            primary_delta[dataset]["ci_low"] > 0
            for dataset in TARGETS
        )
    )
    early_delta = {
        dataset: bootstrap_early[dataset]["paired_deltas"][EARLY]
        for dataset in TARGETS
    }
    early_promoted = (
        any(early_delta[dataset]["mean"] >= 0.015 for dataset in TARGETS)
        and all(early_delta[dataset]["mean"] >= -0.010 for dataset in TARGETS)
        and any(
            early_delta[dataset]["probability_positive"] >= 0.95
            for dataset in TARGETS
        )
    )
    result = {
        "protocol": 'V04_LOCKED_EXTERNAL_PROTOCOL_V1',
        "methods": list(METHODS),
        "sources": list(SOURCES),
        "targets": {
            dataset: {
                "rows": int(len(targets[dataset])),
                "groups": int(targets[dataset]["patient_id"].nunique()),
                "benign": int((targets[dataset]["target"] == 0).sum()),
                "malignant": int((targets[dataset]["target"] == 1).sum()),
            }
            for dataset in TARGETS
        },
        "thresholds": thresholds,
        "tuning": tuning,
        "metrics": metric_summary,
        "bootstrap_vs_frozen98": bootstrap_reference,
        "bootstrap_early_vs_advanced76": bootstrap_early,
        "decision": {
            "primary_method": PRIMARY,
            "primary_supported": bool(primary_supported),
            "primary_auc": primary_auc,
            "reference_auc": reference_auc,
            "primary_paired_deltas": primary_delta,
            "early_fusion_promoted": bool(early_promoted),
            "early_fusion_paired_deltas_vs_primary": early_delta,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (OUTPUT_DIR / "locked_external_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    rows = []
    for dataset in TARGETS:
        for method in METHODS:
            record = metric_summary[dataset][method]["ensemble"]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **record,
                    "auc_ci_low": bootstrap_reference[dataset][
                        "auc_intervals"
                    ][method]["low"],
                    "auc_ci_high": bootstrap_reference[dataset][
                        "auc_intervals"
                    ][method]["high"],
                    "delta_vs_frozen98": (
                        0.0
                        if method == REFERENCE
                        else bootstrap_reference[dataset][
                            "paired_deltas"
                        ][method]["mean"]
                    ),
                    "delta_ci_low": (
                        0.0
                        if method == REFERENCE
                        else bootstrap_reference[dataset][
                            "paired_deltas"
                        ][method]["ci_low"]
                    ),
                    "delta_ci_high": (
                        0.0
                        if method == REFERENCE
                        else bootstrap_reference[dataset][
                            "paired_deltas"
                        ][method]["ci_high"]
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "locked_external_metrics.csv",
        index=False,
    )
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
