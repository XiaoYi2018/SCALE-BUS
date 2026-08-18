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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_busuclm_grouped_cv import (  # noqa: E402
    C_GRID,
    SEED,
    patient_equal_weights,
    valid_group_splits,
)


FEATURE_DIR = ROOT / "results" / "v04_advanced_features"
EMBEDDING_DIR = ROOT / "results" / "v04_deep_embeddings"
OUTPUT_DIR = ROOT / "results" / "v04_embedding_screen"
VIEWS = ("roi_context", "lesion_only", "inner_only", "lesion_outer")
ENCODERS = ("resnet18", "efficientnet_b0", "convnext_tiny")
PCA_COMPONENTS = 32


def prepare_metadata(dataset: str) -> pd.DataFrame:
    path = FEATURE_DIR / (
        "features_busuclm_advanced.csv"
        if dataset == "busuclm"
        else "features_busi_advanced.csv"
    )
    frame = pd.read_csv(path)
    if dataset == "busi":
        frame["patient_id"] = frame["cv_group_id"].astype(str)
    else:
        frame["patient_id"] = frame["patient_id"].astype(str)
    frame["target"] = (frame["label"] == "malignant").astype(np.int64)
    frame["original_order"] = np.arange(len(frame))
    return frame.sort_values(["patient_id", "image"]).reset_index(drop=True)


def load_embedding(
    dataset: str,
    encoder: str,
    view: str,
    frame: pd.DataFrame,
) -> np.ndarray:
    path = EMBEDDING_DIR / f"{dataset}_{encoder}_{view}.npz"
    bundle = np.load(path, allow_pickle=True)
    embedding = np.asarray(bundle["embedding"], dtype=np.float64)
    image = bundle["image"].astype(str)
    original = frame.sort_values("original_order")
    if image.tolist() != original["image"].astype(str).tolist():
        raise RuntimeError(f"{dataset} embedding image order mismatch")
    embedding = embedding[frame["original_order"].to_numpy(dtype=np.int64)]
    if embedding.shape[0] != len(frame) or not np.isfinite(embedding).all():
        raise RuntimeError(f"{dataset} invalid embedding matrix")
    return embedding


def preprocess_fit(
    matrix: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    components = min(PCA_COMPONENTS, scaled.shape[0] - 2, scaled.shape[1])
    if components < 2:
        raise RuntimeError("too few samples for embedding PCA")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=seed,
    )
    transformed = pca.fit_transform(scaled)
    return scaler, pca, transformed


def transform(
    matrix: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    return pca.transform(scaler.transform(matrix))


def tune_c(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    splits: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    inner, split_seed = valid_group_splits(frame, splits, seed)
    scores: dict[float, list[float]] = {
        float(c): [] for c in C_GRID
    }
    for split_number, (train_index, validation_index) in enumerate(inner, 1):
        scaler, pca, train_values = preprocess_fit(
            matrix[train_index],
            seed + split_number,
        )
        validation_values = transform(
            matrix[validation_index],
            scaler,
            pca,
        )
        for c in C_GRID:
            model = LogisticRegression(
                C=float(c),
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
                solver="liblinear",
            )
            model.fit(train_values, frame.iloc[train_index]["target"])
            probability = model.predict_proba(validation_values)[:, 1]
            scores[float(c)].append(
                float(
                    roc_auc_score(
                        frame.iloc[validation_index]["target"],
                        probability,
                    )
                )
            )
    rows = [
        {
            "c": c,
            "auc": float(np.mean(values)),
            "split_seed": float(split_seed),
        }
        for c, values in scores.items()
    ]
    best = sorted(rows, key=lambda item: (-item["auc"], item["c"]))[0]
    return float(best["c"]), rows


def fit_predict(
    train_frame: pd.DataFrame,
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    c: float,
    seed: int,
) -> np.ndarray:
    scaler, pca, train_values = preprocess_fit(train_matrix, seed)
    test_values = transform(test_matrix, scaler, pca)
    model = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=5000,
        random_state=SEED,
        solver="liblinear",
    )
    model.fit(train_values, train_frame["target"])
    return model.predict_proba(test_values)[:, 1]


def within_dataset(
    dataset: str,
    frame: pd.DataFrame,
    matrix: np.ndarray,
    output: Path,
    repetitions: int = 3,
) -> tuple[float, dict[str, object]]:
    if dataset == "busi":
        outer_splits, inner_splits = 5, 4
    else:
        outer_splits, inner_splits = 4, 3
    predictions: list[dict[str, object]] = []
    tuning_records: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    overlap = 0

    for repeat in range(repetitions):
        outer, outer_seed = valid_group_splits(
            frame,
            outer_splits,
            SEED + repeat * 1000,
        )
        for fold, (train_index, test_index) in enumerate(outer, 1):
            train = frame.iloc[train_index].reset_index(drop=True)
            best_c, tuning = tune_c(
                train,
                matrix[train_index],
                inner_splits,
                SEED + repeat * 1000 + fold * 100,
            )
            probability = fit_predict(
                train,
                matrix[train_index],
                matrix[test_index],
                best_c,
                SEED + repeat * 1000 + fold,
            )
            test = frame.iloc[test_index]
            for record, value in zip(
                test.itertuples(index=False),
                probability,
                strict=True,
            ):
                predictions.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
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
                        "fold": fold,
                        "outer_seed": outer_seed,
                        "split": "train",
                        "rows": int(len(train)),
                        "groups": int(train["patient_id"].nunique()),
                        "patient_ids": "|".join(sorted(train_ids)),
                    },
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
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
                    "fold": fold,
                    "best_c": best_c,
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
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "repetitions": repetitions,
        "fold_group_overlaps": int(overlap),
    }


def transfer(
    source_name: str,
    target_name: str,
    source_frame: pd.DataFrame,
    source_matrix: np.ndarray,
    target_frame: pd.DataFrame,
    target_matrix: np.ndarray,
    output: Path,
) -> tuple[float, dict[str, object]]:
    source_splits = 5 if source_name == "busi" else 4
    best_c, tuning = tune_c(
        source_frame,
        source_matrix,
        source_splits,
        SEED,
    )
    probability = fit_predict(
        source_frame,
        source_matrix,
        target_matrix,
        best_c,
        SEED,
    )
    prediction = target_frame[
        ["image", "patient_id", "label", "target"]
    ].copy()
    prediction["probability"] = probability
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
            probability,
            sample_weight=patient_equal_weights(prediction),
        )
    )
    return auc, {
        "source": source_name,
        "target": target_name,
        "auc": auc,
        "best_c": best_c,
        "pca_components": PCA_COMPONENTS,
        "source_rows": int(len(source_frame)),
        "source_groups": int(source_frame["patient_id"].nunique()),
        "target_rows": int(len(target_frame)),
        "target_groups": int(target_frame["patient_id"].nunique()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    configuration = f"{args.encoder}_{args.view}"
    output = OUTPUT_DIR / configuration
    output.mkdir(parents=True, exist_ok=True)
    busuclm = prepare_metadata("busuclm")
    busi = prepare_metadata("busi")
    busuclm_embedding = load_embedding(
        "busuclm",
        args.encoder,
        args.view,
        busuclm,
    )
    busi_embedding = load_embedding(
        "busi",
        args.encoder,
        args.view,
        busi,
    )

    uclm_internal, uclm_record = within_dataset(
        "busuclm",
        busuclm,
        busuclm_embedding,
        output,
    )
    busi_internal, busi_record = within_dataset(
        "busi",
        busi,
        busi_embedding,
        output,
    )
    uclm_to_busi, uclm_to_busi_record = transfer(
        "busuclm",
        "busi",
        busuclm,
        busuclm_embedding,
        busi,
        busi_embedding,
        output,
    )
    busi_to_uclm, busi_to_uclm_record = transfer(
        "busi",
        "busuclm",
        busi,
        busi_embedding,
        busuclm,
        busuclm_embedding,
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
        "configuration": configuration,
        "encoder": args.encoder,
        "view": args.view,
        "raw_dimension": int(busuclm_embedding.shape[1]),
        "pca_components": PCA_COMPONENTS,
        "within": {"busuclm": uclm_record, "busi": busi_record},
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
