from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEEP_ROOT = ROOT / "results" / "v06_finetuned_deep"
HANDCRAFTED_LOCKED = ROOT / "results" / "v04_locked_external"
HANDCRAFTED_LODO = ROOT / "results" / "v04_four_dataset_transfer"
OUT = ROOT / "results" / "v06_deep_comparator_analysis"
ITERATIONS = 5000
SEED = 20260718


def group_weights(frame: pd.DataFrame) -> np.ndarray:
    return 1.0 / frame.groupby("patient_id")["patient_id"].transform("size").to_numpy(float)


def auc(frame: pd.DataFrame, probability: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = group_weights(frame)
    return float(roc_auc_score(frame["target"], probability, sample_weight=weights))


def paired_bootstrap(
    frame: pd.DataFrame,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
) -> tuple[float, float, float]:
    units = {
        str(unit): part.index.to_numpy(dtype=int)
        for unit, part in frame.groupby("patient_id", sort=True)
    }
    names = np.asarray(list(units), dtype=object)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(ITERATIONS):
        sampled = rng.choice(names, size=len(names), replace=True)
        selected = [units[name] for name in sampled]
        indices = np.concatenate(selected)
        weights = np.concatenate(
            [np.repeat(1.0 / len(group_indices), len(group_indices)) for group_indices in selected]
        )
        labels = frame.loc[indices, "target"].to_numpy(dtype=int)
        if np.unique(labels).size < 2:
            continue
        deltas.append(
            roc_auc_score(labels, probability_a[indices], sample_weight=weights)
            - roc_auc_score(labels, probability_b[indices], sample_weight=weights)
        )
    values = np.asarray(deltas, dtype=float)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values > 0)),
    )


def load_handcrafted(protocol: str, dataset: str) -> pd.DataFrame:
    stem = dataset.lower().replace("-", "")
    if protocol == "locked":
        path = HANDCRAFTED_LOCKED / f"{stem}_locked_predictions.csv"
        probability_column = "probability_advanced76_logistic_ensemble"
    else:
        path = HANDCRAFTED_LODO / f"lodo_{stem}_predictions.csv"
        probability_column = "probability_advanced76"
    frame = pd.read_csv(path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["handcrafted_probability"] = frame[probability_column].astype(float)
    return frame[["image", "patient_id", "target", "handcrafted_probability"]].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("locked", "lodo"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    deep_path = DEEP_ROOT / args.protocol / "predictions.csv"
    deep = pd.read_csv(deep_path)
    deep["patient_id"] = deep["patient_id"].astype(str)
    records = []
    merged_outputs = []
    for (architecture, dataset), part in deep.groupby(["architecture", "dataset"], sort=True):
        averaged = (
            part.groupby(["image", "patient_id", "target"], as_index=False)
            .agg(
                deep_probability=("probability", "mean"),
                deep_probability_min=("probability", "min"),
                deep_probability_max=("probability", "max"),
                seeds=("seed", "nunique"),
            )
        )
        handcrafted = load_handcrafted(args.protocol, dataset)
        merged = handcrafted.merge(
            averaged,
            on=["image", "patient_id", "target"],
            validate="one_to_one",
        ).sort_values(["patient_id", "image"]).reset_index(drop=True)
        if len(merged) != len(handcrafted):
            raise RuntimeError(f"{args.protocol} {architecture} {dataset}: incomplete merge")
        hprob = merged["handcrafted_probability"].to_numpy(float)
        dprob = merged["deep_probability"].to_numpy(float)
        handcrafted_auc = auc(merged, hprob)
        deep_auc = auc(merged, dprob)
        low, high, positive = paired_bootstrap(merged, hprob, dprob)
        records.append(
            {
                "protocol": args.protocol,
                "architecture": architecture,
                "target_dataset": dataset,
                "seeds_averaged": int(merged["seeds"].iloc[0]),
                "handcrafted_auc": handcrafted_auc,
                "deep_auc_seed_ensemble": deep_auc,
                "delta_handcrafted_minus_deep": handcrafted_auc - deep_auc,
                "delta_ci_low": low,
                "delta_ci_high": high,
                "probability_positive_delta": positive,
                "images": len(merged),
                "groups": merged["patient_id"].nunique(),
            }
        )
        merged["protocol"] = args.protocol
        merged["architecture"] = architecture
        merged["dataset"] = dataset
        merged_outputs.append(merged)
    result = pd.DataFrame(records)
    result.to_csv(OUT / f"{args.protocol}_paired_comparison.csv", index=False)
    pd.concat(merged_outputs, ignore_index=True).to_csv(
        OUT / f"{args.protocol}_paired_predictions.csv",
        index=False,
    )
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
