from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fractal_extrema import feature_columns_for_group  # noqa: E402
from run_busuclm_grouped_cv import patient_equal_weights  # noqa: E402
from run_cross_dataset_transfer import prepare_datasets  # noqa: E402


SEED = 20260717
PERMUTATIONS = 1000
COEFFICIENT_BOOTSTRAPS = 300
OUT_DIR = ROOT / "results" / "feature_family_contribution"
TRANSFER_JSON = (
    ROOT
    / "results"
    / "cross_dataset_transfer"
    / "cross_dataset_transfer.json"
)


def family_for_column(column: str) -> str:
    if column.startswith("basic_"):
        return "basic"
    if column.startswith("fractal_"):
        return "fractal"
    if column.startswith("zone_lesion_"):
        return "lesion_extrema"
    if column.startswith("zone_inner_"):
        return "inner_margin_extrema"
    if column.startswith("zone_outer_"):
        return "outer_margin_extrema"
    raise KeyError(f"unassigned fused feature: {column}")


def build_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )


def coefficient_stability(
    source: pd.DataFrame,
    columns: list[str],
    c_value: float,
    full_coefficients: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    groups = source["domain_group"].unique()
    by_group = {
        group: source.index[source["domain_group"] == group].to_numpy()
        for group in groups
    }
    coefficients: list[np.ndarray] = []
    for _ in range(COEFFICIENT_BOOTSTRAPS):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[group] for group in drawn])
        sample = source.loc[indices]
        if sample["target"].nunique() < 2:
            continue
        model = build_model(c_value)
        model.fit(sample[columns], sample["target"])
        coefficients.append(model.named_steps["model"].coef_[0])
    values = np.asarray(coefficients)
    families = np.asarray([family_for_column(column) for column in columns])
    feature_rows: list[dict[str, object]] = []
    for index, column in enumerate(columns):
        sign = np.sign(full_coefficients[index])
        feature_rows.append(
            {
                "feature": column,
                "family": families[index],
                "full_coefficient": float(full_coefficients[index]),
                "absolute_coefficient": float(abs(full_coefficients[index])),
                "bootstrap_median": float(np.median(values[:, index])),
                "bootstrap_abs_median": float(
                    np.median(np.abs(values[:, index]))
                ),
                "sign_consistency": float(
                    np.mean(np.sign(values[:, index]) == sign)
                ),
            }
        )
    family_rows: list[dict[str, object]] = []
    for family in sorted(set(families)):
        selected = families == family
        magnitude = np.mean(np.abs(values[:, selected]), axis=1)
        family_rows.append(
            {
                "family": family,
                "features": int(selected.sum()),
                "mean_abs_coefficient": float(np.mean(magnitude)),
                "ci_low": float(np.quantile(magnitude, 0.025)),
                "ci_high": float(np.quantile(magnitude, 0.975)),
            }
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(family_rows)


def error_complementarity(
    target: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    labels = target["target"].to_numpy()
    fused_correct = (probabilities["fused_multizone"] >= 0.5) == labels
    rows: list[dict[str, object]] = []
    for reference in ("basic", "basic_fractal"):
        reference_correct = (probabilities[reference] >= 0.5) == labels
        corrected = fused_correct & ~reference_correct
        worsened = ~fused_correct & reference_correct
        rows.append(
            {
                "reference": reference,
                "corrected_total": int(corrected.sum()),
                "worsened_total": int(worsened.sum()),
                "net_corrected": int(corrected.sum() - worsened.sum()),
                "malignant_corrected": int((corrected & (labels == 1)).sum()),
                "malignant_worsened": int((worsened & (labels == 1)).sum()),
                "benign_corrected": int((corrected & (labels == 0)).sum()),
                "benign_worsened": int((worsened & (labels == 0)).sum()),
            }
        )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = prepare_datasets()
    transfer = json.loads(TRANSFER_JSON.read_text())
    result_directions: list[dict[str, object]] = []

    for direction_index, direction in enumerate(transfer["directions"]):
        source_name = str(direction["source"])
        target_name = str(direction["target"])
        source = datasets[source_name]
        target = datasets[target_name]
        c_by_group = {
            str(row["group"]): float(row["best_c"])
            for row in direction["groups"]
        }
        probabilities: dict[str, np.ndarray] = {}
        fitted: dict[str, Pipeline] = {}
        for group in ("basic", "basic_fractal", "fused_multizone"):
            columns = feature_columns_for_group(source.columns, group)
            model = build_model(c_by_group[group])
            model.fit(source[columns], source["target"])
            probabilities[group] = model.predict_proba(target[columns])[:, 1]
            fitted[group] = model

        fused_columns = feature_columns_for_group(
            source.columns,
            "fused_multizone",
        )
        fused_model = fitted["fused_multizone"]
        weights = patient_equal_weights(target)
        baseline_auc = float(
            roc_auc_score(
                target["target"],
                probabilities["fused_multizone"],
                sample_weight=weights,
            )
        )
        families: dict[str, list[str]] = {}
        for column in fused_columns:
            families.setdefault(family_for_column(column), []).append(column)

        rng = np.random.default_rng(SEED + direction_index * 10000)
        permutation_rows: list[dict[str, object]] = []
        for family, columns in families.items():
            drops: list[float] = []
            for _ in range(PERMUTATIONS):
                order = rng.permutation(len(target))
                permuted = target[fused_columns].copy()
                permuted.loc[:, columns] = target[columns].to_numpy()[order]
                probability = fused_model.predict_proba(permuted)[:, 1]
                auc = float(
                    roc_auc_score(
                        target["target"],
                        probability,
                        sample_weight=weights,
                    )
                )
                drops.append(baseline_auc - auc)
            permutation_rows.append(
                {
                    "family": family,
                    "features": len(columns),
                    "baseline_auc": baseline_auc,
                    "auc_drop_mean": float(np.mean(drops)),
                    "auc_drop_ci_low": float(np.quantile(drops, 0.025)),
                    "auc_drop_ci_high": float(np.quantile(drops, 0.975)),
                    "probability_positive": float(np.mean(np.asarray(drops) > 0)),
                }
            )

        coefficients = fused_model.named_steps["model"].coef_[0]
        feature_stability, family_stability = coefficient_stability(
            source,
            fused_columns,
            c_by_group["fused_multizone"],
            coefficients,
        )
        errors = error_complementarity(target, probabilities)
        direction_name = (
            f"{source_name}_to_{target_name}".replace(" ", "_")
        )
        pd.DataFrame(permutation_rows).to_csv(
            OUT_DIR / f"{direction_name}_permutation.csv",
            index=False,
        )
        feature_stability.sort_values(
            "absolute_coefficient",
            ascending=False,
        ).to_csv(
            OUT_DIR / f"{direction_name}_coefficient_stability.csv",
            index=False,
        )
        family_stability.to_csv(
            OUT_DIR / f"{direction_name}_family_coefficients.csv",
            index=False,
        )
        prediction_frame = target[
            ["image", "patient_id", "label", "target"]
        ].copy()
        for group, probability in probabilities.items():
            prediction_frame[f"probability_{group}"] = probability
        prediction_frame.to_csv(
            OUT_DIR / f"{direction_name}_predictions.csv",
            index=False,
        )
        result_directions.append(
            {
                "source": source_name,
                "target": target_name,
                "permutation": permutation_rows,
                "error_complementarity": errors,
                "top_coefficients": (
                    feature_stability.sort_values(
                        "absolute_coefficient",
                        ascending=False,
                    )
                    .head(20)
                    .to_dict(orient="records")
                ),
                "family_coefficients": family_stability.to_dict(
                    orient="records"
                ),
            }
        )
        print(
            f"{source_name} -> {target_name}: permutation, coefficients, "
            "and errors complete",
            flush=True,
        )

    result = {
        "seed": SEED,
        "permutations": PERMUTATIONS,
        "coefficient_bootstraps": COEFFICIENT_BOOTSTRAPS,
        "directions": result_directions,
    }
    (OUT_DIR / "feature_family_contribution.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print("feature family contribution complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
