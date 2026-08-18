from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "v04_four_dataset_transfer"
OUTPUT = ROOT / "results" / "v06_lodo_paired_ci"
DATASETS = ("busuclm", "busi", "busbra", "breast")
ITERATIONS = 5000
SEED = 20260718


def group_weights(frame: pd.DataFrame) -> np.ndarray:
    return 1.0 / frame.groupby("patient_id")["patient_id"].transform("size").to_numpy(float)


def weighted_auc(frame: pd.DataFrame, probability: np.ndarray) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            probability,
            sample_weight=group_weights(frame),
        )
    )


def bootstrap(frame: pd.DataFrame) -> tuple[float, float, float]:
    frozen = frame["probability_frozen98"].to_numpy(float)
    advanced = frame["probability_advanced76"].to_numpy(float)
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
            roc_auc_score(labels, advanced[indices], sample_weight=weights)
            - roc_auc_score(labels, frozen[indices], sample_weight=weights)
        )
    values = np.asarray(deltas, dtype=float)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values > 0)),
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in DATASETS:
        frame = pd.read_csv(INPUT / f"lodo_{dataset}_predictions.csv")
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame = frame.sort_values(["patient_id", "image"]).reset_index(drop=True)
        frozen_auc = weighted_auc(frame, frame["probability_frozen98"].to_numpy(float))
        advanced_auc = weighted_auc(frame, frame["probability_advanced76"].to_numpy(float))
        low, high, positive = bootstrap(frame)
        records.append(
            {
                "held_out_dataset": dataset,
                "fre98_auc": frozen_auc,
                "gfwb76_auc": advanced_auc,
                "delta": advanced_auc - frozen_auc,
                "delta_ci_low": low,
                "delta_ci_high": high,
                "probability_positive_delta": positive,
                "images": len(frame),
                "groups": frame["patient_id"].nunique(),
                "bootstrap_iterations": ITERATIONS,
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT / "lodo_paired_ci.csv", index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
