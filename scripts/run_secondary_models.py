from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fractal_extrema import feature_columns_for_group  # noqa: E402
from run_busuclm_grouped_cv import (  # noqa: E402
    cluster_bootstrap,
    patient_equal_weights,
    valid_group_splits,
)
from run_cross_dataset_transfer import prepare_datasets  # noqa: E402


SEED = 20260717
REPETITIONS = 5
BOOTSTRAP_ITERATIONS = 2000
SECONDARY_GROUPS = (
    "basic",
    "basic_fractal",
    "basic_margin_extrema",
    "fused_multizone",
)
MODELS = ("rbf_svm", "random_forest")
OUT_DIR = ROOT / "results" / "secondary_models"


def fit_predict(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    inner_splits: int,
    repeat: int,
    fold: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if model_name == "rbf_svm":
        inner_frame = train.reset_index(drop=True).copy()
        inner_frame["patient_id"] = inner_frame["domain_group"]
        inner, inner_seed = valid_group_splits(
            inner_frame,
            inner_splits,
            SEED + repeat * 1000 + fold * 100,
        )
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            model,
            param_grid={
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.1],
            },
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        search.fit(train[columns], train["target"])
        scores = search.decision_function(test[columns])
        tuning = {
            "best_c": float(search.best_params_["model__C"]),
            "best_gamma": str(search.best_params_["model__gamma"]),
            "inner_auc": float(search.best_score_),
            "inner_seed": inner_seed,
        }
        return np.asarray(scores, dtype=np.float64), tuning

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED + repeat * 100 + fold,
    )
    model.fit(train[columns], train["target"])
    return (
        model.predict_proba(test[columns])[:, 1],
        {
            "best_c": None,
            "best_gamma": None,
            "inner_auc": None,
            "inner_seed": None,
        },
    )


def evaluate_dataset(
    dataset_name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    outer_splits = 4 if dataset_name.startswith("BUS-UCLM") else 5
    inner_splits = 3 if dataset_name.startswith("BUS-UCLM") else 4
    working = frame.copy()
    working["patient_id"] = working["domain_group"]
    prediction_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    outer_by_repeat: dict[int, tuple[list[tuple[np.ndarray, np.ndarray]], int]] = {}
    for repeat in range(REPETITIONS):
        outer_by_repeat[repeat] = valid_group_splits(
            working,
            outer_splits,
            SEED + repeat * 1000,
        )

    for model_name in MODELS:
        for group in SECONDARY_GROUPS:
            columns = feature_columns_for_group(working.columns, group)
            for repeat in range(REPETITIONS):
                outer, outer_seed = outer_by_repeat[repeat]
                for fold, (train_idx, test_idx) in enumerate(outer, 1):
                    train = working.iloc[train_idx]
                    test = working.iloc[test_idx]
                    scores, tuning = fit_predict(
                        model_name,
                        train,
                        test,
                        columns,
                        inner_splits,
                        repeat,
                        fold,
                    )
                    for row, score in zip(
                        test.itertuples(index=False),
                        scores,
                        strict=True,
                    ):
                        prediction_rows.append(
                            {
                                "model": model_name,
                                "group": group,
                                "repeat": repeat + 1,
                                "fold": fold,
                                "image": row.image,
                                "patient_id": row.patient_id,
                                "label": row.label,
                                "target": int(row.target),
                                "score": float(score),
                            }
                        )
                    tuning_rows.append(
                        {
                            "model": model_name,
                            "group": group,
                            "repeat": repeat + 1,
                            "fold": fold,
                            "outer_seed": outer_seed,
                            **tuning,
                        }
                    )
            print(
                f"{dataset_name} {model_name} {group}: complete",
                flush=True,
            )

    predictions = pd.DataFrame(prediction_rows)
    tuning = pd.DataFrame(tuning_rows)
    safe_name = dataset_name.replace(" ", "_")
    predictions.to_csv(
        OUT_DIR / f"{safe_name}_predictions.csv",
        index=False,
    )
    tuning.to_csv(OUT_DIR / f"{safe_name}_tuning.csv", index=False)
    summary_rows: list[dict[str, object]] = []

    for model_name in MODELS:
        subset = predictions.loc[predictions["model"] == model_name]
        means = (
            subset.groupby(
                ["group", "image", "patient_id", "label", "target"],
                as_index=False,
            )["score"]
            .mean()
        )
        wide = means.pivot(
            index=["image", "patient_id", "label", "target"],
            columns="group",
            values="score",
        ).reset_index()
        wide.columns.name = None
        wide = wide.rename(
            columns={group: f"probability_{group}" for group in SECONDARY_GROUPS}
        )
        intervals, delta_basic = cluster_bootstrap(
            wide,
            list(SECONDARY_GROUPS),
            BOOTSTRAP_ITERATIONS,
            "basic",
        )
        _, delta_fractal = cluster_bootstrap(
            wide,
            ["basic_fractal", "fused_multizone"],
            BOOTSTRAP_ITERATIONS,
            "basic_fractal",
        )
        weights = patient_equal_weights(wide)
        for group in SECONDARY_GROUPS:
            score = wide[f"probability_{group}"]
            repeat_aucs = []
            for _, repeat_frame in subset.loc[
                subset["group"] == group
            ].groupby("repeat"):
                repeat_aucs.append(
                    roc_auc_score(
                        repeat_frame["target"],
                        repeat_frame["score"],
                    )
                )
            row: dict[str, object] = {
                "dataset": dataset_name,
                "model": model_name,
                "group": group,
                "auc": float(roc_auc_score(wide["target"], score)),
                "group_balanced_auc": float(
                    roc_auc_score(
                        wide["target"],
                        score,
                        sample_weight=weights,
                    )
                ),
                "auc_ci_low": intervals[group][0],
                "auc_ci_high": intervals[group][1],
                "repeat_auc_mean": float(np.mean(repeat_aucs)),
                "repeat_auc_std": float(np.std(repeat_aucs, ddof=1)),
            }
            if group != "basic":
                row["delta_vs_basic"] = delta_basic[group]
            if group == "fused_multizone":
                row["delta_vs_basic_fractal"] = delta_fractal[group]
            summary_rows.append(row)

    return {
        "dataset": dataset_name,
        "rows": int(len(frame)),
        "groups": int(frame["domain_group"].nunique()),
        "summary": summary_rows,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = prepare_datasets()
    results = [
        evaluate_dataset(name, frame)
        for name, frame in datasets.items()
    ]
    summary_rows = [
        row for result in results for row in result["summary"]
    ]
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "secondary_model_summary.csv",
        index=False,
    )
    result = {
        "seed": SEED,
        "repetitions": REPETITIONS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "feature_groups": list(SECONDARY_GROUPS),
        "models": list(MODELS),
        "datasets": results,
    }
    (OUT_DIR / "secondary_model_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print("secondary models complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
