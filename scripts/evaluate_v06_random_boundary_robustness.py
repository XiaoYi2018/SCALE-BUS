from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_v04_feature_domain_ablation import build_model, feature_blocks  # noqa: E402
from evaluate_v04_locked_external import external_frame  # noqa: E402
from run_busuclm_grouped_cv import patient_equal_weights  # noqa: E402
from screen_v04_embedding import prepare_metadata  # noqa: E402


INPUT_DIR = ROOT / "results" / "v06_random_boundary_perturbations"
OUTPUT_DIR = ROOT / "results" / "v06_random_boundary_robustness"
DATASETS = ("busbra", "breast")
AMPLITUDES = (25, 50)
SEEDS = (20260718, 20260719, 20260720, 20260721, 20260722)
METHODS = {"shape10": "Boundary shape (10)", "advanced76": "GFWB-76"}
ITERATIONS = 5000
BOOTSTRAP_SEED = 20260718


def auc(frame: pd.DataFrame, probability: np.ndarray) -> float:
    return float(
        roc_auc_score(
            frame["target"],
            probability,
            sample_weight=patient_equal_weights(frame),
        )
    )


def paired_bootstrap(
    metadata: pd.DataFrame,
    clean: np.ndarray,
    perturbed: np.ndarray,
) -> tuple[float, float, float]:
    units = {
        str(unit): part.index.to_numpy(dtype=int)
        for unit, part in metadata.groupby("patient_id", sort=True)
    }
    names = np.asarray(list(units), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(ITERATIONS):
        sampled = rng.choice(names, size=len(names), replace=True)
        selected = [units[name] for name in sampled]
        indices = np.concatenate(selected)
        weights = np.concatenate(
            [np.repeat(1.0 / len(group_indices), len(group_indices)) for group_indices in selected]
        )
        labels = metadata.loc[indices, "target"].to_numpy(dtype=int)
        if np.unique(labels).size < 2:
            continue
        values.append(
            roc_auc_score(labels, perturbed[indices], sample_weight=weights)
            - roc_auc_score(labels, clean[indices], sample_weight=weights)
        )
    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        float(np.mean(array >= 0)),
    )


def load_condition(dataset: str, amplitude: int, seed: int, clean: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(
        INPUT_DIR / f"features_{dataset}_jitter_{amplitude:03d}_seed_{seed}.csv"
    )
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["target"] = frame["label"].astype(str).str.lower().eq("malignant").astype(int)
    frame = frame.sort_values(["patient_id", "image"]).reset_index(drop=True)
    keys = ["image", "patient_id", "target"]
    if frame[keys].to_dict("records") != clean[keys].to_dict("records"):
        raise RuntimeError(f"{dataset} amplitude={amplitude} seed={seed}: order mismatch")
    return frame


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {"busuclm": prepare_metadata("busuclm"), "busi": prepare_metadata("busi")}
    clean_targets = {dataset: external_frame(dataset) for dataset in DATASETS}
    blocks = feature_blocks(sources["busuclm"])
    models = {}
    tuning = {}
    for method in METHODS:
        columns = blocks[method]
        for source_name, source in sources.items():
            model, best_c = build_model(source, columns, 4 if source_name == "busuclm" else 5)
            models[(method, source_name)] = model
            tuning[(method, source_name)] = best_c

    condition_records = []
    summary_records = []
    prediction_outputs = []
    for dataset in DATASETS:
        clean = clean_targets[dataset]
        for method in METHODS:
            columns = blocks[method]
            clean_probability = np.mean(
                [
                    models[(method, source)].predict_proba(clean[columns])[:, 1]
                    for source in ("busuclm", "busi")
                ],
                axis=0,
            )
            clean_auc = auc(clean, clean_probability)
            for amplitude in AMPLITUDES:
                seed_probabilities = []
                seed_aucs = []
                for seed in SEEDS:
                    perturbed = load_condition(dataset, amplitude, seed, clean)
                    probability = np.mean(
                        [
                            models[(method, source)].predict_proba(perturbed[columns])[:, 1]
                            for source in ("busuclm", "busi")
                        ],
                        axis=0,
                    )
                    value = auc(clean, probability)
                    seed_probabilities.append(probability)
                    seed_aucs.append(value)
                    condition_records.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "amplitude_equivalent_radius": amplitude / 1000.0,
                            "seed": seed,
                            "auc": value,
                            "delta_vs_clean": value - clean_auc,
                            "clean_auc": clean_auc,
                        }
                    )
                ensemble_probability = np.mean(seed_probabilities, axis=0)
                ensemble_auc = auc(clean, ensemble_probability)
                low, high, nonnegative = paired_bootstrap(
                    clean,
                    clean_probability,
                    ensemble_probability,
                )
                summary_records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "amplitude_equivalent_radius": amplitude / 1000.0,
                        "clean_auc": clean_auc,
                        "seed_auc_mean": float(np.mean(seed_aucs)),
                        "seed_auc_min": float(np.min(seed_aucs)),
                        "seed_auc_max": float(np.max(seed_aucs)),
                        "seed_delta_median": float(np.median(np.asarray(seed_aucs) - clean_auc)),
                        "seed_worst_delta": float(np.min(np.asarray(seed_aucs) - clean_auc)),
                        "ensemble_probability_auc": ensemble_auc,
                        "ensemble_delta_vs_clean": ensemble_auc - clean_auc,
                        "ensemble_delta_ci_low": low,
                        "ensemble_delta_ci_high": high,
                        "probability_nonnegative_delta": nonnegative,
                    }
                )
                output = clean[["image", "patient_id", "label", "target"]].copy()
                output["dataset"] = dataset
                output["method"] = method
                output["amplitude_equivalent_radius"] = amplitude / 1000.0
                output["clean_probability"] = clean_probability
                output["jitter_ensemble_probability"] = ensemble_probability
                prediction_outputs.append(output)

    pd.DataFrame(condition_records).to_csv(
        OUTPUT_DIR / "condition_metrics.csv",
        index=False,
    )
    summary = pd.DataFrame(summary_records)
    summary.to_csv(OUTPUT_DIR / "amplitude_summary.csv", index=False)
    pd.concat(prediction_outputs, ignore_index=True).to_csv(
        OUTPUT_DIR / "ensemble_predictions.csv",
        index=False,
    )
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "iterations": ITERATIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "jitter_seeds": SEEDS,
                "amplitudes_equivalent_radius": [value / 1000.0 for value in AMPLITUDES],
                "source_tuning_c": {f"{method}|{source}": value for (method, source), value in tuning.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
