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


OUTPUT_DIR = ROOT / "results" / "v05_marginal_block_ablation"
DATASET_ORDER = ("busuclm", "busi", "busbra", "breast")
EXTERNAL_TARGETS = ("busbra", "breast")
BOOTSTRAP_ITERATIONS = 3000
FULL_MODEL = "gfwb76"
MODEL_ORDER = (
    "gf30",
    "w36",
    "s10",
    "gfw66",
    "gfs40",
    "ws46",
    FULL_MODEL,
)
MODEL_LABELS = {
    "gf30": "Generalized-fractal (GF)",
    "w36": "Wavelet (W)",
    "s10": "Boundary shape (S)",
    "gfw66": "GF + W",
    "gfs40": "GF + S",
    "ws46": "W + S",
    "gfwb76": "GF + W + S (GFWB-76)",
}


def feature_blocks(frame: pd.DataFrame) -> dict[str, list[str]]:
    gf = [column for column in frame if column.startswith("advanced_mf_")]
    wavelet = [
        column for column in frame if column.startswith("advanced_wavelet_")
    ]
    shape = [
        column for column in frame if column.startswith("advanced_shape_")
    ]
    blocks = {
        "gf30": gf,
        "w36": wavelet,
        "s10": shape,
        "gfw66": gf + wavelet,
        "gfs40": gf + shape,
        "ws46": wavelet + shape,
        "gfwb76": gf + wavelet + shape,
    }
    expected = {
        "gf30": 30,
        "w36": 36,
        "s10": 10,
        "gfw66": 66,
        "gfs40": 40,
        "ws46": 46,
        "gfwb76": 76,
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
) -> tuple[Pipeline, float, float]:
    working = source.copy()
    working["domain_group"] = working["patient_id"].astype(str)
    best_c, tuning = source_tune_c(working, columns, splits)
    selected_auc = float(
        np.mean([row["auc"] for row in tuning if row["c"] == best_c])
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
    return model, float(best_c), selected_auc


def weighted_auc(frame: pd.DataFrame, probability: np.ndarray) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            probability,
            sample_weight=patient_equal_weights(frame),
        )
    )


def corrected_group_bootstrap(
    frame: pd.DataFrame,
    model_names: tuple[str, ...],
    iterations: int,
    seed_offset: int,
) -> tuple[dict[str, tuple[float, float]], list[dict[str, object]], int]:
    """Bootstrap groups while preserving the multiplicity of repeated draws."""
    rng = np.random.default_rng(SEED + seed_offset)
    patients = frame["patient_id"].astype(str).unique()
    by_patient = {
        patient: frame.index[
            frame["patient_id"].astype(str) == patient
        ].to_numpy()
        for patient in patients
    }
    auc_samples = {model: [] for model in model_names}
    delta_samples = {
        model: [] for model in model_names if model != FULL_MODEL
    }
    valid_iterations = 0

    for _ in range(iterations):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        indices: list[int] = []
        weights: list[float] = []
        for patient in drawn:
            selected = by_patient[str(patient)]
            indices.extend(selected.tolist())
            weights.extend([1.0 / len(selected)] * len(selected))
        boot = frame.loc[indices]
        labels = boot["target"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        sample_weight = np.asarray(weights, dtype=np.float64)
        current: dict[str, float] = {}
        for model in model_names:
            auc = float(
                roc_auc_score(
                    labels,
                    boot[f"probability_{model}"],
                    sample_weight=sample_weight,
                )
            )
            auc_samples[model].append(auc)
            current[model] = auc
        for model in delta_samples:
            delta_samples[model].append(current[FULL_MODEL] - current[model])
        valid_iterations += 1

    if valid_iterations == 0:
        raise RuntimeError("no class-valid bootstrap iteration was generated")
    intervals = {
        model: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for model, values in auc_samples.items()
    }
    deltas = [
        {
            "full_model": FULL_MODEL,
            "ablated_model": model,
            "delta_mean": float(np.mean(values)),
            "delta_ci_low": float(np.quantile(values, 0.025)),
            "delta_ci_high": float(np.quantile(values, 0.975)),
            "probability_positive": float(np.mean(np.asarray(values) > 0)),
        }
        for model, values in delta_samples.items()
    ]
    return intervals, deltas, valid_iterations


def locked_external_factorial(
    datasets: dict[str, pd.DataFrame],
    blocks: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    auc_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    predictions = {
        target: datasets[target][
            ["image", "patient_id", "label", "target", "birads", "device"]
        ].copy()
        for target in EXTERNAL_TARGETS
    }
    for model_name in MODEL_ORDER:
        columns = blocks[model_name]
        source_predictions: dict[str, dict[str, np.ndarray]] = {}
        source_cs: dict[str, float] = {}
        for source_name in ("busuclm", "busi"):
            splits = 4 if source_name == "busuclm" else 5
            model, best_c, selected_auc = build_model(
                datasets[source_name],
                columns,
                splits,
            )
            source_cs[source_name] = best_c
            source_predictions[source_name] = {
                target: model.predict_proba(datasets[target][columns])[:, 1]
                for target in EXTERNAL_TARGETS
            }
            tuning_rows.append(
                {
                    "context": "locked_external",
                    "held_out": "",
                    "source": source_name,
                    "model": model_name,
                    "features": len(columns),
                    "best_c": best_c,
                    "selected_source_cv_auc": selected_auc,
                }
            )
        for target in EXTERNAL_TARGETS:
            probability = np.mean(
                [
                    source_predictions[source][target]
                    for source in ("busuclm", "busi")
                ],
                axis=0,
            )
            predictions[target][f"probability_{model_name}"] = probability
            auc_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "features": len(columns),
                    "auc": weighted_auc(datasets[target], probability),
                    "busuclm_best_c": source_cs["busuclm"],
                    "busi_best_c": source_cs["busi"],
                }
            )
    return pd.DataFrame(auc_rows), predictions, pd.DataFrame(tuning_rows)


def lodo_factorial(
    datasets: dict[str, pd.DataFrame],
    blocks: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    auc_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for held_out in DATASET_ORDER:
        target = datasets[held_out]
        prediction_frame = target[
            ["image", "patient_id", "label", "target"]
        ].copy()
        source_names = [name for name in DATASET_ORDER if name != held_out]
        for model_name in MODEL_ORDER:
            columns = blocks[model_name]
            source_parts = []
            for source_name in source_names:
                part = datasets[source_name][
                    ["image", "patient_id", "label", "target"] + columns
                ].copy()
                part["patient_id"] = (
                    source_name + ":" + part["patient_id"].astype(str)
                )
                part["domain_group"] = part["patient_id"]
                source_parts.append(part)
            pooled = pd.concat(source_parts, ignore_index=True)
            model, best_c, selected_auc = build_model(pooled, columns, 5)
            probability = model.predict_proba(target[columns])[:, 1]
            prediction_frame[f"probability_{model_name}"] = probability
            auc_rows.append(
                {
                    "held_out": held_out,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "features": len(columns),
                    "auc": weighted_auc(target, probability),
                    "best_c": best_c,
                    "source_rows": len(pooled),
                    "source_groups": pooled["patient_id"].nunique(),
                }
            )
            tuning_rows.append(
                {
                    "context": "lodo",
                    "held_out": held_out,
                    "source": "+".join(source_names),
                    "model": model_name,
                    "features": len(columns),
                    "best_c": best_c,
                    "selected_source_cv_auc": selected_auc,
                }
            )
        predictions[held_out] = prediction_frame
    return pd.DataFrame(auc_rows), predictions, pd.DataFrame(tuning_rows)


def add_intervals_and_differences(
    auc_table: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    context: str,
    target_column: str,
    seed_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    interval_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for offset, (target, frame) in enumerate(predictions.items()):
        intervals, deltas, valid = corrected_group_bootstrap(
            frame,
            MODEL_ORDER,
            BOOTSTRAP_ITERATIONS,
            seed_start + offset,
        )
        for model, (low, high) in intervals.items():
            interval_rows.append(
                {
                    "context": context,
                    target_column: target,
                    "model": model,
                    "auc_ci_low": low,
                    "auc_ci_high": high,
                }
            )
        point_aucs = {
            row.model: float(row.auc)
            for row in auc_table.loc[
                auc_table[target_column] == target
            ].itertuples(index=False)
        }
        for delta in deltas:
            ablated = str(delta["ablated_model"])
            delta_rows.append(
                {
                    "context": context,
                    target_column: target,
                    **delta,
                    "point_delta": point_aucs[FULL_MODEL] - point_aucs[ablated],
                    "bootstrap_iterations_requested": BOOTSTRAP_ITERATIONS,
                    "bootstrap_iterations_valid": valid,
                }
            )
        audit_rows.append(
            {
                "context": context,
                "target": target,
                "groups": int(frame["patient_id"].nunique()),
                "rows": int(len(frame)),
                "bootstrap_iterations_requested": BOOTSTRAP_ITERATIONS,
                "bootstrap_iterations_valid": valid,
                "multiplicity_preserved": True,
            }
        )
    intervals = pd.DataFrame(interval_rows)
    enriched = auc_table.merge(
        intervals,
        on=[target_column, "model"],
        how="left",
        validate="one_to_one",
    )
    return enriched, pd.DataFrame(delta_rows), audit_rows


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
        if feature_blocks(datasets[dataset]) != blocks:
            raise RuntimeError(f"{dataset}: feature block columns differ")

    external_auc, external_predictions, external_tuning = (
        locked_external_factorial(datasets, blocks)
    )
    external_auc, external_deltas, external_audit = (
        add_intervals_and_differences(
            external_auc,
            external_predictions,
            "locked_external",
            "target",
            100,
        )
    )
    external_auc.to_csv(
        OUTPUT_DIR / "locked_external_factorial_auc.csv",
        index=False,
    )
    for target, frame in external_predictions.items():
        frame.to_csv(
            OUTPUT_DIR / f"locked_external_{target}_predictions.csv",
            index=False,
        )

    lodo_auc, lodo_predictions, lodo_tuning = lodo_factorial(
        datasets,
        blocks,
    )
    lodo_auc, lodo_deltas, lodo_audit = add_intervals_and_differences(
        lodo_auc,
        lodo_predictions,
        "lodo",
        "held_out",
        200,
    )
    lodo_auc.to_csv(OUTPUT_DIR / "lodo_factorial_auc.csv", index=False)
    for target, frame in lodo_predictions.items():
        frame.to_csv(
            OUTPUT_DIR / f"lodo_{target}_predictions.csv",
            index=False,
        )

    deltas = pd.concat(
        [external_deltas, lodo_deltas],
        ignore_index=True,
        sort=False,
    )
    deltas.to_csv(
        OUTPUT_DIR / "paired_marginal_differences.csv",
        index=False,
    )
    tuning = pd.concat(
        [external_tuning, lodo_tuning],
        ignore_index=True,
    )
    tuning.to_csv(OUTPUT_DIR / "model_tuning.csv", index=False)

    lodo_means = (
        lodo_auc.groupby("model")["auc"].agg(["mean", "min"]).reset_index()
    )
    full_lodo = lodo_means.loc[lodo_means["model"] == FULL_MODEL].iloc[0]
    gf_marginal = deltas.loc[deltas["ablated_model"] == "ws46"].copy()
    gate = {
        "criterion_external_positive_ci": bool(
            (
                (gf_marginal["context"] == "locked_external")
                & (gf_marginal["delta_ci_low"] > 0)
            ).any()
        ),
        "criterion_mean_lodo_at_least_0_01": bool(
            lodo_auc.loc[lodo_auc["model"] == FULL_MODEL, "auc"].mean()
            - lodo_auc.loc[lodo_auc["model"] == "ws46", "auc"].mean()
            >= 0.01
        ),
        "criterion_no_lodo_target_below_minus_0_01": bool(
            (
                gf_marginal.loc[
                    gf_marginal["context"] == "lodo",
                    "point_delta",
                ]
                >= -0.01
            ).all()
        ),
    }
    gate["retain_strong_gf_contribution_claim"] = bool(
        gate["criterion_external_positive_ci"]
        or (
            gate["criterion_mean_lodo_at_least_0_01"]
            and gate["criterion_no_lodo_target_below_minus_0_01"]
        )
    )
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_method": (
            "patient/group resampling with replacement; each repeated draw "
            "is appended separately with within-group weights summing to one"
        ),
        "feature_blocks": {
            name: {
                "label": MODEL_LABELS[name],
                "features": len(columns),
                "columns": columns,
            }
            for name, columns in blocks.items()
        },
        "locked_external": external_auc.to_dict(orient="records"),
        "lodo": lodo_auc.to_dict(orient="records"),
        "lodo_model_summary": lodo_means.to_dict(orient="records"),
        "full_model_lodo_mean": float(full_lodo["mean"]),
        "full_model_lodo_minimum": float(full_lodo["min"]),
        "bootstrap_audit": external_audit + lodo_audit,
        "predeclared_gf_gate": gate,
    }
    (OUTPUT_DIR / "marginal_block_ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["predeclared_gf_gate"], indent=2))
    print(
        f"Completed in {summary['elapsed_seconds']:.1f} seconds. "
        f"Outputs: {OUTPUT_DIR}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
