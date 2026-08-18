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
from screen_v04_embedding import (  # noqa: E402
    load_embedding,
    prepare_metadata,
)
from screen_v04_handcrafted import feature_columns  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "v04_early_fusion_screen"
GROUPS = ("advanced76", "frozen98_shape", "frozen98_advanced")
ENCODER = "resnet18"
VIEW = "inner_only"
PCA_COMPONENTS = 32


def fit_preprocessor(
    handcrafted: np.ndarray,
    embedding: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, StandardScaler, PCA, np.ndarray]:
    handcrafted_scaler = StandardScaler()
    handcrafted_values = handcrafted_scaler.fit_transform(handcrafted)
    embedding_scaler = StandardScaler()
    embedding_scaled = embedding_scaler.fit_transform(embedding)
    components = min(
        PCA_COMPONENTS,
        embedding_scaled.shape[0] - 2,
        embedding_scaled.shape[1],
    )
    if components < 2:
        raise RuntimeError("too few samples for embedding PCA")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=seed,
    )
    embedding_values = pca.fit_transform(embedding_scaled)
    values = np.concatenate(
        [handcrafted_values, embedding_values],
        axis=1,
    )
    return handcrafted_scaler, embedding_scaler, pca, values


def transform(
    handcrafted: np.ndarray,
    embedding: np.ndarray,
    handcrafted_scaler: StandardScaler,
    embedding_scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    return np.concatenate(
        [
            handcrafted_scaler.transform(handcrafted),
            pca.transform(embedding_scaler.transform(embedding)),
        ],
        axis=1,
    )


def tune_c(
    frame: pd.DataFrame,
    handcrafted: np.ndarray,
    embedding: np.ndarray,
    splits: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    inner, split_seed = valid_group_splits(frame, splits, seed)
    scores = {float(c): [] for c in C_GRID}
    for split_number, (train_index, validation_index) in enumerate(inner, 1):
        hs, es, pca, train_values = fit_preprocessor(
            handcrafted[train_index],
            embedding[train_index],
            seed + split_number,
        )
        validation_values = transform(
            handcrafted[validation_index],
            embedding[validation_index],
            hs,
            es,
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
    train_handcrafted: np.ndarray,
    train_embedding: np.ndarray,
    test_handcrafted: np.ndarray,
    test_embedding: np.ndarray,
    c: float,
    seed: int,
) -> np.ndarray:
    hs, es, pca, train_values = fit_preprocessor(
        train_handcrafted,
        train_embedding,
        seed,
    )
    test_values = transform(
        test_handcrafted,
        test_embedding,
        hs,
        es,
        pca,
    )
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
    handcrafted: np.ndarray,
    embedding: np.ndarray,
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
        outer, outer_seed = valid_group_splits(
            frame,
            outer_splits,
            SEED + repeat * 1000,
        )
        for fold, (train_index, test_index) in enumerate(outer, 1):
            train = frame.iloc[train_index].reset_index(drop=True)
            best_c, tuning = tune_c(
                train,
                handcrafted[train_index],
                embedding[train_index],
                inner_splits,
                SEED + repeat * 1000 + fold * 100,
            )
            probability = fit_predict(
                train,
                handcrafted[train_index],
                embedding[train_index],
                handcrafted[test_index],
                embedding[test_index],
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
    source_handcrafted: np.ndarray,
    source_embedding: np.ndarray,
    target_frame: pd.DataFrame,
    target_handcrafted: np.ndarray,
    target_embedding: np.ndarray,
    output: Path,
) -> tuple[float, dict[str, object]]:
    source_splits = 5 if source_name == "busi" else 4
    best_c, tuning = tune_c(
        source_frame,
        source_handcrafted,
        source_embedding,
        source_splits,
        SEED,
    )
    probability = fit_predict(
        source_frame,
        source_handcrafted,
        source_embedding,
        target_handcrafted,
        target_embedding,
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
        "source_rows": int(len(source_frame)),
        "source_groups": int(source_frame["patient_id"].nunique()),
        "target_rows": int(len(target_frame)),
        "target_groups": int(target_frame["patient_id"].nunique()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    args = parser.parse_args()
    output = OUTPUT_DIR / f"{args.group}_{ENCODER}_{VIEW}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    frames = {
        dataset: prepare_metadata(dataset)
        for dataset in ("busuclm", "busi")
    }
    columns = feature_columns(frames["busuclm"], args.group)
    if columns != feature_columns(frames["busi"], args.group):
        raise RuntimeError("feature columns differ between cohorts")
    handcrafted = {
        dataset: frames[dataset][columns].to_numpy(dtype=np.float64)
        for dataset in frames
    }
    embedding = {
        dataset: load_embedding(
            dataset,
            ENCODER,
            VIEW,
            frames[dataset],
        )
        for dataset in frames
    }
    for dataset in frames:
        if not np.isfinite(handcrafted[dataset]).all():
            raise RuntimeError(f"{dataset} handcrafted values are invalid")

    uclm_internal, uclm_record = within_dataset(
        "busuclm",
        frames["busuclm"],
        handcrafted["busuclm"],
        embedding["busuclm"],
        output,
    )
    busi_internal, busi_record = within_dataset(
        "busi",
        frames["busi"],
        handcrafted["busi"],
        embedding["busi"],
        output,
    )
    uclm_to_busi, u2b_record = transfer(
        "busuclm",
        "busi",
        frames["busuclm"],
        handcrafted["busuclm"],
        embedding["busuclm"],
        frames["busi"],
        handcrafted["busi"],
        embedding["busi"],
        output,
    )
    busi_to_uclm, b2u_record = transfer(
        "busi",
        "busuclm",
        frames["busi"],
        handcrafted["busi"],
        embedding["busi"],
        frames["busuclm"],
        handcrafted["busuclm"],
        embedding["busuclm"],
        output,
    )
    result = {
        "protocol": 'V04_DEVELOPMENT_PROTOCOL_V1',
        "group": args.group,
        "handcrafted_features": len(columns),
        "embedding": f"{ENCODER}_{VIEW}",
        "embedding_pca_components": PCA_COMPONENTS,
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
