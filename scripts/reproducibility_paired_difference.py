"""Paired patient bootstrap of inter-view score reproducibility statistics."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from metrology_paths import OUTPUT_ROOT

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
RESULTS = OUTPUT_ROOT
ANALYSIS = RESULTS / "v07_interview_reproducibility"
PAIRED_SCORES = ANALYSIS / "paired_locked_probabilities.csv"
PAIR_CONSISTENCY = ANALYSIS / "pair_consistency_all_811.csv"
BUS_BRA_PREDICTIONS = RESULTS / "v07_positive_control" / "busbra_predictions.csv"
OUTPUT = OUTPUT_ROOT / "metrology_statistics" / "reproducibility"
SEED = 20261018
BOOTSTRAP_ITERATIONS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logit_probability(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1.0 - values))


def icc_statistics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    pair_mean = (first + second) / 2.0
    pair_mean = pair_mean - pair_mean.mean()
    paired_variance = (first - second) ** 2 / 2.0
    between = 2.0 * np.var(pair_mean, ddof=1)
    within = paired_variance.mean()
    denominator = between + within
    icc = float((between - within) / denominator) if denominator > 0 else float("nan")
    sw = float(math.sqrt(max(within, 0.0)))
    return {"icc": icc, "sw": sw, "rdc": 2.77 * sw}


def contrast(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    return icc_statistics(logit_probability(first), logit_probability(second))


def bootstrap_rows(frame: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, float]]]:
    if not frame["Case"].is_unique:
        raise RuntimeError("Case identifiers must be unique for patient-level bootstrap")
    advanced_first = frame["advanced76_first"].to_numpy(dtype=float)
    advanced_second = frame["advanced76_second"].to_numpy(dtype=float)
    frozen_first = frame["frozen98_first"].to_numpy(dtype=float)
    frozen_second = frame["frozen98_second"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    indices_matrix = rng.integers(0, len(frame), size=(BOOTSTRAP_ITERATIONS, len(frame)))
    rows = []
    for indices in indices_matrix:
        advanced = contrast(advanced_first[indices], advanced_second[indices])
        frozen = contrast(frozen_first[indices], frozen_second[indices])
        rows.append(
            {
                "advanced_icc": advanced["icc"],
                "frozen_icc": frozen["icc"],
                "icc_difference": advanced["icc"] - frozen["icc"],
                "advanced_sw": advanced["sw"],
                "frozen_sw": frozen["sw"],
                "sw_ratio": advanced["sw"] / frozen["sw"],
                "advanced_rdc": advanced["rdc"],
                "frozen_rdc": frozen["rdc"],
                "rdc_difference": advanced["rdc"] - frozen["rdc"],
            }
        )
    return indices_matrix, rows


def all_811_estimates() -> tuple[dict[str, dict[str, float]], int]:
    consistency = pd.read_csv(PAIR_CONSISTENCY)
    predictions = pd.read_csv(BUS_BRA_PREDICTIONS)
    if len(consistency) != 811 or not predictions["image"].is_unique:
        raise RuntimeError("Unexpected 811-pair or BUS-BRA prediction inventory")
    prediction_lookup = predictions.set_index(predictions["image"].astype(str).map(lambda value: Path(value).stem))
    first_ids = consistency["ID_first"].astype(str)
    second_ids = consistency["ID_second"].astype(str)
    missing = sorted((set(first_ids) | set(second_ids)) - set(prediction_lookup.index))
    if missing:
        raise RuntimeError(f"Missing prediction identifiers: {missing[:5]}")
    advanced_first = prediction_lookup.loc[first_ids, "probability_advanced76_ensemble"].to_numpy(dtype=float)
    advanced_second = prediction_lookup.loc[second_ids, "probability_advanced76_ensemble"].to_numpy(dtype=float)
    frozen_first = prediction_lookup.loc[first_ids, "probability_frozen98_ensemble"].to_numpy(dtype=float)
    frozen_second = prediction_lookup.loc[second_ids, "probability_frozen98_ensemble"].to_numpy(dtype=float)
    return {
        "advanced76": contrast(advanced_first, advanced_second),
        "frozen98": contrast(frozen_first, frozen_second),
    }, len(consistency)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory already exists: {OUTPUT}")
    frame = pd.read_csv(PAIRED_SCORES)
    if len(frame) != 810:
        raise RuntimeError("Expected 810 included pairs")
    point_advanced = contrast(frame["advanced76_first"].to_numpy(), frame["advanced76_second"].to_numpy())
    point_frozen = contrast(frame["frozen98_first"].to_numpy(), frame["frozen98_second"].to_numpy())
    point = {
        "advanced76": point_advanced,
        "frozen98": point_frozen,
        "differences": {
            "icc": point_advanced["icc"] - point_frozen["icc"],
            "sw_ratio": point_advanced["sw"] / point_frozen["sw"],
            "rdc": point_advanced["rdc"] - point_frozen["rdc"],
        },
    }

    _, bootstrap = bootstrap_rows(frame)
    bootstrap_frame = pd.DataFrame(bootstrap)
    intervals = {}
    for column in bootstrap_frame.columns:
        values = bootstrap_frame[column].to_numpy(dtype=float)
        intervals[column] = {
            "mean": float(values.mean()),
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }

    sensitivity_811, sensitivity_n = all_811_estimates()
    reproduction = {
        "advanced76_icc_error": abs(point_advanced["icc"] - 0.825500),
        "frozen98_icc_error": abs(point_frozen["icc"] - 0.718392),
        "advanced76_rdc_error": abs(point_advanced["rdc"] - 1.979341),
        "frozen98_rdc_error": abs(point_frozen["rdc"] - 2.548396),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise RuntimeError(f"Output directory appeared during analysis: {OUTPUT}")
    OUTPUT.mkdir()
    bootstrap_frame.to_csv(OUTPUT / "bootstrap_distribution.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {"model": "SCALE-BUS", **point_advanced},
            {"model": "FRE", **point_frozen},
        ]
    ).to_csv(OUTPUT / "point_estimates.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [{"contrast": key, **value} for key, value in intervals.items()]
    ).to_csv(OUTPUT / "bootstrap_intervals.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {"pair_set": "included_810", "model": "SCALE-BUS", **point_advanced},
            {"pair_set": "included_810", "model": "FRE", **point_frozen},
            {"pair_set": "all_811", "model": "SCALE-BUS", **sensitivity_811["advanced76"]},
            {"pair_set": "all_811", "model": "FRE", **sensitivity_811["frozen98"]},
        ]
    ).to_csv(OUTPUT / "sample_sensitivity.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [{"check": key, "absolute_difference": value} for key, value in reproduction.items()]
    ).to_csv(OUTPUT / "reproduction_checks.csv", index=False, lineterminator="\n")

    output_hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    summary = {
        "analysis": "paired_reproducibility_difference",
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_unit": "lesion/pair; one lesion per patient in BUS-BRA",
        "input_hashes": {
            "paired_locked_probabilities.csv": sha256(PAIRED_SCORES),
            "pair_consistency_all_811.csv": sha256(PAIR_CONSISTENCY),
            "busbra_predictions.csv": sha256(BUS_BRA_PREDICTIONS),
        },
        "point_estimates": point,
        "bootstrap_intervals": intervals,
        "sample_size": {"included_pairs": len(frame), "all_pairs": sensitivity_n},
        "sensitivity_all_811": sensitivity_811,
        "reproduction_checks": reproduction,
        "output_hashes": output_hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "point": point, "intervals": intervals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
