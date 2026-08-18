from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "clinical_utility"
SEED = 20260717
BOOTSTRAPS = 2000
THRESHOLDS = np.round(np.arange(0.10, 0.801, 0.01), 2)
MODELS = {
    "basic": "probability_basic",
    "basic_fractal": "probability_basic_fractal",
    "fused_multizone": "probability_fused_multizone",
}
DATASETS = {
    "BUSI-valid": (
        ROOT / "results" / "busi_cv_multizone" / "mean_oof_predictions.csv"
    ),
    "BUS-UCLM-clean": (
        ROOT
        / "results"
        / "busuclm_cv_primary_multizone"
        / "mean_oof_predictions.csv"
    ),
}


def group_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("patient_id")["patient_id"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights * frame["patient_id"].nunique() / weights.sum()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_calibration(
    target: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))

    def objective(parameters: np.ndarray) -> float:
        linear = parameters[0] + parameters[1] * logit
        fitted = np.where(
            linear >= 0,
            1.0 / (1.0 + np.exp(-linear)),
            np.exp(linear) / (1.0 + np.exp(linear)),
        )
        fitted = np.clip(fitted, 1e-12, 1.0 - 1e-12)
        loss = -(target * np.log(fitted) + (1 - target) * np.log(1 - fitted))
        return weighted_mean(loss, weights)

    result = minimize(
        objective,
        x0=np.asarray([0.0, 1.0]),
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-9},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"calibration optimization failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def reliability_bins(
    target: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    order = np.argsort(probability, kind="stable")
    cumulative = np.cumsum(weights[order])
    bin_id_sorted = np.minimum(
        (n_bins * (cumulative - 0.5 * weights[order]) / weights.sum()).astype(int),
        n_bins - 1,
    )
    bin_ids = np.empty_like(bin_id_sorted)
    bin_ids[order] = bin_id_sorted
    rows: list[dict[str, float | int]] = []
    for bin_id in range(n_bins):
        selected = bin_ids == bin_id
        if not selected.any():
            continue
        bin_weights = weights[selected]
        mean_prediction = weighted_mean(probability[selected], bin_weights)
        observed_fraction = weighted_mean(target[selected], bin_weights)
        rows.append(
            {
                "bin": bin_id + 1,
                "n_images": int(selected.sum()),
                "weight": float(bin_weights.sum()),
                "mean_prediction": mean_prediction,
                "observed_fraction": observed_fraction,
                "absolute_gap": abs(mean_prediction - observed_fraction),
            }
        )
    return pd.DataFrame(rows)


def decision_curve(
    target: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    total = weights.sum()
    values = []
    for threshold in THRESHOLDS:
        positive = probability >= threshold
        true_positive = np.sum(weights * (positive & (target == 1)))
        false_positive = np.sum(weights * (positive & (target == 0)))
        values.append(
            true_positive / total
            - false_positive / total * threshold / (1.0 - threshold)
        )
    return np.asarray(values, dtype=float)


def group_metric_table(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    working = frame[["patient_id", "target"]].copy()
    working["probability"] = probability
    working["brier"] = (working["probability"] - working["target"]) ** 2
    rows = []
    for group_id, group in working.groupby("patient_id", sort=False):
        row: dict[str, float | str] = {
            "patient_id": group_id,
            "brier": float(group["brier"].mean()),
        }
        target = group["target"].to_numpy(dtype=int)
        score = group["probability"].to_numpy(dtype=float)
        for position, threshold in enumerate(THRESHOLDS):
            positive = score >= threshold
            row[f"tp_{position}"] = float(np.mean(positive & (target == 1)))
            row[f"fp_{position}"] = float(np.mean(positive & (target == 0)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("patient_id")


def bootstrap_differences(
    group_tables: dict[str, pd.DataFrame],
) -> dict[str, dict[str, list[float] | float]]:
    groups = group_tables["fused_multizone"].index.to_numpy()
    rng = np.random.default_rng(SEED)
    output: dict[str, dict[str, list[float] | float]] = {}
    fused = group_tables["fused_multizone"]
    for comparator in ("basic", "basic_fractal"):
        reference = group_tables[comparator]
        brier_delta = np.empty(BOOTSTRAPS, dtype=float)
        mean_nb_delta = np.empty(BOOTSTRAPS, dtype=float)
        for bootstrap in range(BOOTSTRAPS):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            fused_sample = fused.loc[sampled]
            reference_sample = reference.loc[sampled]
            brier_delta[bootstrap] = (
                fused_sample["brier"].mean() - reference_sample["brier"].mean()
            )
            fused_nb = []
            reference_nb = []
            for position, threshold in enumerate(THRESHOLDS):
                odds = threshold / (1.0 - threshold)
                fused_nb.append(
                    fused_sample[f"tp_{position}"].mean()
                    - fused_sample[f"fp_{position}"].mean() * odds
                )
                reference_nb.append(
                    reference_sample[f"tp_{position}"].mean()
                    - reference_sample[f"fp_{position}"].mean() * odds
                )
            mean_nb_delta[bootstrap] = float(
                np.mean(np.asarray(fused_nb) - np.asarray(reference_nb))
            )
        output[comparator] = {
            "brier_delta": float(
                fused["brier"].mean() - reference["brier"].mean()
            ),
            "brier_delta_ci": np.quantile(
                brier_delta, [0.025, 0.975]
            ).tolist(),
            "mean_net_benefit_delta": float(
                np.mean(
                    [
                        (
                            fused[f"tp_{position}"].mean()
                            - fused[f"fp_{position}"].mean()
                            * threshold
                            / (1.0 - threshold)
                        )
                        - (
                            reference[f"tp_{position}"].mean()
                            - reference[f"fp_{position}"].mean()
                            * threshold
                            / (1.0 - threshold)
                        )
                        for position, threshold in enumerate(THRESHOLDS)
                    ]
                )
            ),
            "mean_net_benefit_delta_ci": np.quantile(
                mean_nb_delta, [0.025, 0.975]
            ).tolist(),
        }
    return output


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float | int | str]] = []
    bin_rows: list[pd.DataFrame] = []
    curve_rows: list[dict[str, float | str]] = []
    bootstrap_summary: dict[str, object] = {}

    for dataset, path in DATASETS.items():
        frame = pd.read_csv(path)
        required = {
            "patient_id",
            "target",
            *MODELS.values(),
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{dataset} missing columns: {sorted(missing)}")
        weights = group_balanced_weights(frame)
        target = frame["target"].to_numpy(dtype=int)
        prevalence = weighted_mean(target, weights)
        treat_all = np.asarray(
            [
                prevalence - (1.0 - prevalence) * t / (1.0 - t)
                for t in THRESHOLDS
            ]
        )
        curves: dict[str, np.ndarray] = {}
        groups: dict[str, pd.DataFrame] = {}
        dataset_bins: dict[str, pd.DataFrame] = {}

        for model, column in MODELS.items():
            probability = frame[column].to_numpy(dtype=float)
            if (
                not np.isfinite(probability).all()
                or probability.min() < 0
                or probability.max() > 1
            ):
                raise ValueError(f"invalid probabilities for {dataset}/{model}")
            brier = weighted_mean((probability - target) ** 2, weights)
            clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
            log_loss = weighted_mean(
                -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)),
                weights,
            )
            intercept, slope = weighted_calibration(target, probability, weights)
            bins = reliability_bins(target, probability, weights)
            bins.insert(0, "model", model)
            bins.insert(0, "dataset", dataset)
            bin_rows.append(bins)
            dataset_bins[model] = bins
            ece = float(
                np.sum(bins["weight"] * bins["absolute_gap"])
                / bins["weight"].sum()
            )
            curve = decision_curve(target, probability, weights)
            curves[model] = curve
            useful = curve > np.maximum(treat_all, 0.0)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n_images": len(frame),
                    "n_groups": frame["patient_id"].nunique(),
                    "group_balanced_prevalence": prevalence,
                    "brier_score": brier,
                    "log_loss": log_loss,
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "ece_10_equal_frequency": ece,
                    "mean_net_benefit_0.10_0.80": float(curve.mean()),
                    "useful_threshold_count": int(useful.sum()),
                    "useful_threshold_fraction": float(useful.mean()),
                }
            )
            groups[model] = group_metric_table(frame, probability)
            for threshold, value in zip(THRESHOLDS, curve, strict=True):
                curve_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "threshold": threshold,
                        "net_benefit": float(value),
                    }
                )
        for threshold, none_value, all_value in zip(
            THRESHOLDS,
            np.zeros_like(THRESHOLDS),
            treat_all,
            strict=True,
        ):
            curve_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "model": "treat_none",
                        "threshold": threshold,
                        "net_benefit": float(none_value),
                    },
                    {
                        "dataset": dataset,
                        "model": "treat_all",
                        "threshold": threshold,
                        "net_benefit": float(all_value),
                    },
                ]
            )
        bootstrap_summary[dataset] = bootstrap_differences(groups)

    summary = pd.DataFrame(summary_rows)
    bins = pd.concat(bin_rows, ignore_index=True)
    curves = pd.DataFrame(curve_rows)
    summary.to_csv(OUT_DIR / "clinical_utility_summary.csv", index=False)
    bins.to_csv(OUT_DIR / "calibration_bins.csv", index=False)
    curves.to_csv(OUT_DIR / "decision_curves.csv", index=False)
    payload = {
        "seed": SEED,
        "bootstraps": BOOTSTRAPS,
        "thresholds": [float(x) for x in THRESHOLDS],
        "models": MODELS,
        "bootstrap_differences": bootstrap_summary,
        "summary": summary.to_dict(orient="records"),
    }
    (OUT_DIR / "clinical_utility_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(json.dumps(bootstrap_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
