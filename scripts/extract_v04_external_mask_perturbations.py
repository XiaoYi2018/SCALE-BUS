from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage.morphology import binary_dilation, binary_erosion, disk


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fractal_extrema import extract_advanced_feature_dict  # noqa: E402


INPUT_DIR = ROOT / "results" / "v04_external_features"
OUTPUT_DIR = ROOT / "results" / "v04_external_mask_perturbations"
DATASETS = ("busbra", "breast")
CONDITIONS = ("mask_erode_3", "mask_dilate_3")


def perturb_mask(mask: np.ndarray, condition: str) -> np.ndarray:
    binary = mask > 0
    if condition == "mask_erode_3":
        result = binary_erosion(binary, disk(3))
    elif condition == "mask_dilate_3":
        result = binary_dilation(binary, disk(3))
    else:
        raise KeyError(condition)
    if not result.any():
        raise RuntimeError(f"{condition} produced an empty mask")
    return result.astype(np.uint8)


def extract_one(
    record: dict[str, object],
    condition: str,
) -> tuple[dict[str, object] | None, dict[str, str] | None, float]:
    started = time.perf_counter()
    try:
        with Image.open(str(record["image_path"])) as handle:
            image = np.asarray(handle.convert("L"))
        with Image.open(str(record["mask_path"])) as handle:
            mask = np.asarray(handle.convert("L"))
        perturbed = perturb_mask(mask, condition)
        row = dict(record)
        row["condition"] = condition
        row["perturbed_mask_foreground_pixels"] = int(perturbed.sum())
        row.update(extract_advanced_feature_dict(image, perturbed))
        return row, None, time.perf_counter() - started
    except Exception as exc:
        return (
            None,
            {
                "image": str(record.get("image", "")),
                "patient_id": str(record.get("patient_id", "")),
                "condition": condition,
                "error": repr(exc),
            },
            time.perf_counter() - started,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_path = INPUT_DIR / f"accepted_manifest_{args.dataset}.csv"
    source = pd.read_csv(source_path).sort_values(
        ["patient_id", "image"]
    ).reset_index(drop=True)
    records = source.to_dict(orient="records")
    summary: dict[str, object] = {
        "dataset": args.dataset,
        "source": str(source_path.resolve()),
        "rows": len(source),
        "groups": source["patient_id"].astype(str).nunique(),
        "n_jobs": args.n_jobs,
        "conditions": {},
    }
    for condition in CONDITIONS:
        started = time.perf_counter()
        outputs = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(extract_one)(record, condition) for record in records
        )
        rows = [row for row, _, _ in outputs if row is not None]
        failures = [
            failure for _, failure, _ in outputs if failure is not None
        ]
        elapsed = [value for _, _, value in outputs]
        frame = pd.DataFrame(rows)
        advanced = [
            column for column in frame if column.startswith("advanced_")
        ]
        if len(advanced) != 76:
            raise RuntimeError(
                f"{args.dataset} {condition}: expected 76 advanced columns, "
                f"found {len(advanced)}"
            )
        if failures or len(frame) != len(source):
            summary["conditions"][condition] = {
                "rows": len(frame),
                "failures": failures,
            }
            (OUTPUT_DIR / f"{args.dataset}_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{args.dataset} {condition}: "
                f"{len(failures)} failures, {len(frame)}/{len(source)} rows"
            )
        values = frame[advanced].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"{args.dataset} {condition}: non-finite advanced features"
            )
        output_path = (
            OUTPUT_DIR / f"features_{args.dataset}_{condition}.csv"
        )
        frame.to_csv(output_path, index=False)
        summary["conditions"][condition] = {
            "output": str(output_path.resolve()),
            "rows": len(frame),
            "failures": failures,
            "wall_seconds": time.perf_counter() - started,
            "per_image_seconds_mean": float(np.mean(elapsed)),
            "per_image_seconds_median": float(np.median(elapsed)),
            "foreground_fraction_median": float(
                (
                    frame["perturbed_mask_foreground_pixels"]
                    / source["mask_foreground_pixels"]
                ).median()
            ),
        }
        print(
            f"{args.dataset} {condition}: {len(frame)} rows in "
            f"{summary['conditions'][condition]['wall_seconds']:.1f}s",
            flush=True,
        )
    (OUTPUT_DIR / f"{args.dataset}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
