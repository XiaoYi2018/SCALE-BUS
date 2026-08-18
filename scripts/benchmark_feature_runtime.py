from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fractal_extrema import (  # noqa: E402
    extract_feature_dict,
    extract_multizone_feature_dict,
)


SEED = 20260717
SAMPLES_PER_DATASET = 100
OUT_DIR = ROOT / "results" / "runtime_benchmark"
DATASETS = {
    "BUS-UCLM-clean": (
        ROOT
        / "results"
        / "busuclm_multizone_features"
        / "features_multizone.csv"
    ),
    "BUSI-valid": ROOT / "results" / "busi_zenodo_features_v2" / "features.csv",
}


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for dataset, path in DATASETS.items():
        frame = pd.read_csv(path)
        if dataset == "BUS-UCLM-clean":
            frame = frame.loc[parse_bool(frame["is_clean_primary"])].copy()
        selected = np.sort(
            rng.choice(
                len(frame),
                size=min(SAMPLES_PER_DATASET, len(frame)),
                replace=False,
            )
        )
        for position in selected:
            record = frame.iloc[int(position)]
            started = time.perf_counter()
            with Image.open(record["image_path"]) as handle:
                image = np.asarray(handle.convert("L"))
            with Image.open(record["mask_path"]) as handle:
                mask = np.asarray(handle.convert("L"))
            loaded = time.perf_counter()
            basic_fractal_extrema = extract_feature_dict(image, mask)
            base_finished = time.perf_counter()
            multizone = extract_multizone_feature_dict(image, mask)
            finished = time.perf_counter()
            rows.append(
                {
                    "dataset": dataset,
                    "image": record["image"],
                    "load_seconds": loaded - started,
                    "base_94_seconds": base_finished - loaded,
                    "multizone_63_seconds": finished - base_finished,
                    "total_157_seconds": finished - started,
                    "features": len(basic_fractal_extrema) + len(multizone),
                }
            )
        print(f"{dataset}: benchmarked {len(selected)} samples", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "runtime_per_sample.csv", index=False)
    summary_rows: list[dict[str, object]] = []
    for dataset, subset in frame.groupby("dataset"):
        row: dict[str, object] = {
            "dataset": dataset,
            "samples": int(len(subset)),
        }
        for column in (
            "load_seconds",
            "base_94_seconds",
            "multizone_63_seconds",
            "total_157_seconds",
        ):
            row[f"{column}_mean"] = float(subset[column].mean())
            row[f"{column}_median"] = float(subset[column].median())
            row[f"{column}_p95"] = float(subset[column].quantile(0.95))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "runtime_summary.csv",
        index=False,
    )
    result = {
        "seed": SEED,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu_used": False,
        "summary": summary_rows,
    }
    (OUT_DIR / "runtime_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
