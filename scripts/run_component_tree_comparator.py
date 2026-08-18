from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fractal_extrema.component_tree import (  # noqa: E402
    DEFAULT_AREA_THRESHOLDS,
    extract_component_tree_feature_dict,
)
from run_busuclm_grouped_cv import (  # noqa: E402
    C_GRID,
    SEED,
    classification_metrics,
    cluster_bootstrap,
    evaluate_group,
    parse_bool,
    patient_equal_weights,
)
from run_cross_dataset_transfer import source_tune_c  # noqa: E402


BUSUCLM_SOURCE = (
    ROOT / "results" / "busuclm_multizone_features" / "features_multizone.csv"
)
BUSI_SOURCE = ROOT / "results" / "busi_zenodo_features_v2" / "features.csv"
OUT_DIR = ROOT / "results" / "component_tree_comparator"
BOOTSTRAP_ITERATIONS = 2000

GROUPS = (
    "conventional",
    "conv_fractal",
    "conv_component_tree",
    "conv_fractal_component_tree",
    "fused_extrema",
    "fused_extrema_component_tree",
)


def columns_for_group(frame: pd.DataFrame, group: str) -> list[str]:
    columns = list(frame.columns)
    conventional = [name for name in columns if name.startswith("basic_")]
    fractal = [name for name in columns if name.startswith("fractal_")]
    extrema = [name for name in columns if name.startswith("zone_")]
    component_tree = [name for name in columns if name.startswith("ct_")]
    mapping = {
        "conventional": conventional,
        "conv_fractal": conventional + fractal,
        "conv_component_tree": conventional + component_tree,
        "conv_fractal_component_tree": (
            conventional + fractal + component_tree
        ),
        "fused_extrema": conventional + fractal + extrema,
        "fused_extrema_component_tree": (
            conventional + fractal + extrema + component_tree
        ),
    }
    selected = mapping[group]
    if not selected:
        raise ValueError(f"group {group} selected no columns")
    return selected


def extract_dataset(
    source_path: Path,
    dataset_name: str,
    clean_only: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.read_csv(source_path)
    if clean_only:
        source = source.loc[parse_bool(source["is_clean_primary"])].copy()
    source = source.reset_index(drop=True)
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for position, record in enumerate(source.to_dict(orient="records"), 1):
        with Image.open(record["image_path"]) as handle:
            image = np.asarray(handle.convert("L"))
        with Image.open(record["mask_path"]) as handle:
            mask = np.asarray(handle.convert("L"))
        rows.append(extract_component_tree_feature_dict(image, mask))
        if position == 1 or position % 25 == 0 or position == len(source):
            print(
                f"[extract] {dataset_name}: {position}/{len(source)}",
                flush=True,
            )
    component_tree = pd.DataFrame(rows)
    if component_tree.shape != (len(source), 54):
        raise RuntimeError(
            f"{dataset_name} component-tree matrix has shape "
            f"{component_tree.shape}, expected {(len(source), 54)}"
        )
    merged = pd.concat([source, component_tree], axis=1)
    ct_values = merged.filter(regex=r"^ct_").to_numpy(dtype=np.float64)
    if not np.isfinite(ct_values).all():
        raise RuntimeError(f"{dataset_name} has non-finite comparator values")
    summary = {
        "dataset": dataset_name,
        "source": str(source_path.resolve()),
        "rows": int(len(merged)),
        "groups": int(
            merged[
                "cv_group_id" if dataset_name == "BUSI-valid" else "patient_id"
            ].nunique()
        ),
        "features": 54,
        "area_thresholds": list(DEFAULT_AREA_THRESHOLDS),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return merged, summary


def within_dataset(
    frame: pd.DataFrame,
    dataset_name: str,
) -> dict[str, object]:
    working = frame.copy()
    if dataset_name == "BUSI-valid":
        working["patient_id"] = working["cv_group_id"]
        outer_splits, inner_splits, repetitions = 5, 4, 10
    else:
        outer_splits, inner_splits, repetitions = 4, 3, 10
    working["target"] = (working["label"] == "malignant").astype(np.int64)
    working = working.sort_values(["patient_id", "image"]).reset_index(drop=True)

    output = OUT_DIR / (
        "busi_grouped_cv" if dataset_name == "BUSI-valid"
        else "busuclm_grouped_cv"
    )
    output.mkdir(parents=True, exist_ok=True)
    predictions_all: list[pd.DataFrame] = []
    tuning_all: list[pd.DataFrame] = []
    shared_folds: pd.DataFrame | None = None
    feature_counts: dict[str, int] = {}

    for group in GROUPS:
        columns = columns_for_group(working, group)
        feature_counts[group] = len(columns)
        predictions, tuning, folds = evaluate_group(
            working,
            columns,
            outer_splits,
            inner_splits,
            repetitions,
        )
        predictions.insert(0, "group", group)
        tuning.insert(0, "group", group)
        predictions_all.append(predictions)
        tuning_all.append(tuning)
        if shared_folds is None:
            shared_folds = folds
        elif not shared_folds.equals(folds):
            raise RuntimeError("fold assignments changed across feature groups")
        print(
            f"[within] {dataset_name} {group}: {len(columns)} features",
            flush=True,
        )

    predictions = pd.concat(predictions_all, ignore_index=True)
    tuning = pd.concat(tuning_all, ignore_index=True)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    tuning.to_csv(output / "tuning.csv", index=False)
    assert shared_folds is not None
    shared_folds.to_csv(output / "fold_assignments.csv", index=False)

    means = (
        predictions.groupby(
            ["group", "image", "patient_id", "label", "target"],
            as_index=False,
        )["probability"]
        .mean()
    )
    wide = means.pivot(
        index=["image", "patient_id", "label", "target"],
        columns="group",
        values="probability",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={group: f"probability_{group}" for group in GROUPS}
    )
    wide.to_csv(output / "mean_oof_predictions.csv", index=False)

    intervals, delta_conventional = cluster_bootstrap(
        wide,
        list(GROUPS),
        BOOTSTRAP_ITERATIONS,
        "conventional",
    )
    _, delta_ct_vs_extrema = cluster_bootstrap(
        wide,
        ["conv_fractal_component_tree", "fused_extrema"],
        BOOTSTRAP_ITERATIONS,
        "conv_fractal_component_tree",
    )
    _, delta_added_ct = cluster_bootstrap(
        wide,
        ["fused_extrema", "fused_extrema_component_tree"],
        BOOTSTRAP_ITERATIONS,
        "fused_extrema",
    )
    weights = patient_equal_weights(wide)
    labels = wide["target"].to_numpy()
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        metrics = classification_metrics(
            labels,
            wide[f"probability_{group}"].to_numpy(),
            weights,
        )
        rows.append(
            {
                "group": group,
                "n_features": feature_counts[group],
                **metrics,
                "auc_ci_low": intervals[group][0],
                "auc_ci_high": intervals[group][1],
                "delta_vs_conventional": (
                    delta_conventional[group] if group != "conventional" else None
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    result = {
        "dataset": dataset_name,
        "rows": int(len(working)),
        "groups": int(working["patient_id"].nunique()),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "repetitions": repetitions,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "summary": rows,
        "fused_extrema_vs_conv_fractal_component_tree": delta_ct_vs_extrema[
            "fused_extrema"
        ],
        "added_component_tree_vs_fused_extrema": delta_added_ct[
            "fused_extrema_component_tree"
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def transfer_direction(
    source_name: str,
    target_name: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> dict[str, object]:
    source_splits = 4 if source_name.startswith("BUS-UCLM") else 5
    prediction = target[
        ["image", "label", "target", "domain_group"]
    ].copy()
    prediction = prediction.rename(columns={"domain_group": "patient_id"})
    rows: list[dict[str, object]] = []
    tuning_all: list[pd.DataFrame] = []

    for group in GROUPS:
        columns = columns_for_group(source, group)
        best_c, tuning = source_tune_c(source, columns, source_splits)
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
        probability = model.predict_proba(target[columns])[:, 1]
        prediction[f"probability_{group}"] = probability
        weights = patient_equal_weights(prediction)
        rows.append(
            {
                "group": group,
                "n_features": len(columns),
                "best_c": best_c,
                "source_cv_auc": float(
                    np.mean([item["auc"] for item in tuning if item["c"] == best_c])
                ),
                "target_auc": float(
                    roc_auc_score(
                        target["target"],
                        probability,
                        sample_weight=weights,
                    )
                ),
            }
        )
        tuning_frame = pd.DataFrame(tuning)
        tuning_frame.insert(0, "group", group)
        tuning_all.append(tuning_frame)
        print(
            f"[transfer] {source_name} -> {target_name} {group}: "
            f"{rows[-1]['target_auc']:.4f}",
            flush=True,
        )

    intervals, delta_conventional = cluster_bootstrap(
        prediction,
        list(GROUPS),
        BOOTSTRAP_ITERATIONS,
        "conventional",
    )
    _, delta_ct_vs_extrema = cluster_bootstrap(
        prediction,
        ["conv_fractal_component_tree", "fused_extrema"],
        BOOTSTRAP_ITERATIONS,
        "conv_fractal_component_tree",
    )
    _, delta_added_ct = cluster_bootstrap(
        prediction,
        ["fused_extrema", "fused_extrema_component_tree"],
        BOOTSTRAP_ITERATIONS,
        "fused_extrema",
    )
    for row in rows:
        group = str(row["group"])
        row["auc_ci_low"] = intervals[group][0]
        row["auc_ci_high"] = intervals[group][1]
        if group != "conventional":
            row["delta_vs_conventional"] = delta_conventional[group]

    direction = f"{source_name}_to_{target_name}"
    output = OUT_DIR / "transfer"
    output.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output / f"{direction}_predictions.csv", index=False)
    pd.concat(tuning_all, ignore_index=True).to_csv(
        output / f"{direction}_tuning.csv",
        index=False,
    )
    pd.DataFrame(rows).to_csv(
        output / f"{direction}_summary.csv",
        index=False,
    )
    return {
        "source": source_name,
        "target": target_name,
        "source_rows": int(len(source)),
        "source_groups": int(source["domain_group"].nunique()),
        "target_rows": int(len(target)),
        "target_groups": int(target["domain_group"].nunique()),
        "summary": rows,
        "fused_extrema_vs_conv_fractal_component_tree": delta_ct_vs_extrema[
            "fused_extrema"
        ],
        "added_component_tree_vs_fused_extrema": delta_added_ct[
            "fused_extrema_component_tree"
        ],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    busuclm, busuclm_extract = extract_dataset(
        BUSUCLM_SOURCE,
        "BUS-UCLM-clean",
        clean_only=True,
    )
    busi, busi_extract = extract_dataset(
        BUSI_SOURCE,
        "BUSI-valid",
        clean_only=False,
    )
    busuclm.to_csv(OUT_DIR / "features_busuclm_clean.csv", index=False)
    busi.to_csv(OUT_DIR / "features_busi_valid.csv", index=False)
    (OUT_DIR / "extraction_summary.json").write_text(
        json.dumps(
            {"datasets": [busuclm_extract, busi_extract]},
            indent=2,
        ),
        encoding="utf-8",
    )

    within_results = [
        within_dataset(busuclm, "BUS-UCLM-clean"),
        within_dataset(busi, "BUSI-valid"),
    ]

    busuclm["target"] = (
        busuclm["label"] == "malignant"
    ).astype(np.int64)
    busuclm["domain_group"] = busuclm["patient_id"]
    busi["target"] = (busi["label"] == "malignant").astype(np.int64)
    busi["domain_group"] = busi["cv_group_id"]
    transfer_results = [
        transfer_direction(
            "BUS-UCLM-clean",
            "BUSI-valid",
            busuclm,
            busi,
        ),
        transfer_direction(
            "BUSI-valid",
            "BUS-UCLM-clean",
            busi,
            busuclm,
        ),
    ]
    result = {
        "protocol": 'COMPONENT_TREE_COMPARATOR_PROTOCOL_V1',
        "seed": SEED,
        "area_thresholds": list(DEFAULT_AREA_THRESHOLDS),
        "groups": list(GROUPS),
        "within_dataset": within_results,
        "transfer": transfer_results,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (OUT_DIR / "component_tree_comparator.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

