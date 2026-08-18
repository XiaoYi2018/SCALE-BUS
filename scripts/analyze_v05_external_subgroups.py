from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "v04_locked_external"
OUTPUT_DIR = ROOT / "results" / "v05_external_subgroups"
ITERATIONS = 5000
SEED = 20260718
METHODS = {
    "frozen98_logistic": "FRE-98",
    "advanced76_logistic": "GFWB-76",
    "frozen98_shape_logistic": "FREB-108",
}


def patient_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    units = (
        frame["patient_id"].astype(str)
        + "|"
        + frame["target"].astype(str)
    )
    counts = units.map(units.value_counts())
    return 1.0 / counts.to_numpy(dtype=float)


def auc(frame: pd.DataFrame, column: str) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            frame[column],
            sample_weight=patient_equal_weights(frame),
        )
    )


def paired_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
) -> tuple[float, float, float, float]:
    working = frame.copy()
    working["bootstrap_unit"] = (
        working["patient_id"].astype(str)
        + "|"
        + working["target"].astype(str)
    )
    grouped = {
        unit: part.index.to_numpy(dtype=np.int64)
        for unit, part in working.groupby("bootstrap_unit", sort=True)
    }
    units = np.asarray(list(grouped), dtype=object)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(ITERATIONS):
        sampled = rng.choice(units, size=len(units), replace=True)
        selected_groups = [grouped[unit] for unit in sampled]
        indices = np.concatenate(selected_groups)
        weights = np.concatenate(
            [
                np.repeat(1.0 / len(selected), len(selected))
                for selected in selected_groups
            ]
        )
        replicate = working.loc[indices]
        if replicate["target"].nunique() < 2:
            continue
        labels = replicate["target"].to_numpy()
        differences.append(
            float(
                roc_auc_score(
                    labels,
                    replicate[candidate],
                    sample_weight=weights,
                )
            )
            - float(
                roc_auc_score(
                    labels,
                    replicate[reference],
                    sample_weight=weights,
                )
            )
        )
    values = np.asarray(differences, dtype=float)
    observed = auc(working, candidate) - auc(working, reference)
    return (
        observed,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values > 0)),
    )


def make_subgroups(
    busbra: pd.DataFrame,
    breast: pd.DataFrame,
) -> list[tuple[str, str, pd.DataFrame]]:
    groups: list[tuple[str, str, pd.DataFrame]] = [
        ("busbra_all", "BUS-BRA: all", busbra),
        (
            "busbra_birads34",
            "BUS-BRA: BI-RADS 3-4",
            busbra.loc[busbra["birads"].astype(str).isin(("3", "4"))].copy(),
        ),
        ("breast_all", "BrEaST: all", breast),
        (
            "breast_birads4",
            "BrEaST: BI-RADS 4a-4c",
            breast.loc[
                breast["birads"].astype(str).str.lower().isin(
                    ("4a", "4b", "4c")
                )
            ].copy(),
        ),
    ]
    for device, part in busbra.groupby("device", sort=True):
        counts = part["target"].value_counts()
        if len(part) < 100 or counts.get(0, 0) < 20 or counts.get(1, 0) < 20:
            continue
        short = (
            str(device)
            .replace(" @10-14MHz", "")
            .replace(" @10-12MHz", "")
            .replace(" @12-14 MHz", "")
        )
        groups.append(
            (
                "busbra_device_" + short.lower().replace(" ", "_"),
                f"BUS-BRA: {short}",
                part.copy(),
            )
        )
    return groups


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    busbra = pd.read_csv(INPUT_DIR / "busbra_locked_predictions.csv")
    breast = pd.read_csv(INPUT_DIR / "breast_locked_predictions.csv")
    records: list[dict[str, object]] = []
    for key, label, frame in make_subgroups(busbra, breast):
        benign = int((frame["target"] == 0).sum())
        malignant = int((frame["target"] == 1).sum())
        if benign == 0 or malignant == 0:
            continue
        base_column = "probability_frozen98_logistic_ensemble"
        base_auc = auc(frame, base_column)
        for method, method_label in METHODS.items():
            column = f"probability_{method}_ensemble"
            method_auc = auc(frame, column)
            record = {
                "subgroup": key,
                "subgroup_label": label,
                "method": method,
                "method_label": method_label,
                "rows": len(frame),
                "patients": frame["patient_id"].astype(str).nunique(),
                "benign_rows": benign,
                "malignant_rows": malignant,
                "auc": method_auc,
                "reference_auc": base_auc,
                "delta_vs_fre98": method_auc - base_auc,
            }
            if method == "frozen98_logistic":
                record.update(
                    {
                        "delta_ci_low": 0.0,
                        "delta_ci_high": 0.0,
                        "probability_positive_delta": np.nan,
                    }
                )
            else:
                observed, low, high, positive = paired_bootstrap(
                    frame,
                    column,
                    base_column,
                )
                record.update(
                    {
                        "delta_vs_fre98": observed,
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                        "probability_positive_delta": positive,
                    }
                )
            records.append(record)
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / "external_subgroup_metrics.csv", index=False)

    summary = {
        "bootstrap_iterations": ITERATIONS,
        "subgroups": result.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "external_subgroup_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        result[
            [
                "subgroup_label",
                "method_label",
                "rows",
                "auc",
                "delta_vs_fre98",
                "delta_ci_low",
                "delta_ci_high",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
