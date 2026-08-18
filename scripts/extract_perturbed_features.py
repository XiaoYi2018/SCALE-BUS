from __future__ import annotations

import argparse
import hashlib
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

from fractal_extrema import (  # noqa: E402
    extract_feature_dict,
    extract_multizone_feature_dict,
)


CONDITIONS = (
    "downsample_50",
    "speckle_005",
    "gamma_08",
    "gamma_12",
    "mask_erode_3",
    "mask_dilate_3",
)
DATASETS = {
    "busi": ROOT / "results" / "busi_zenodo_features_v2" / "features.csv",
    "busuclm": (
        ROOT
        / "results"
        / "busuclm_multizone_features"
        / "features_multizone.csv"
    ),
}
METADATA_COLUMNS = (
    "image",
    "patient_id",
    "label",
    "image_path",
    "mask_path",
)


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def deterministic_seed(dataset: str, image_name: str, condition: str) -> int:
    payload = f"{dataset}|{image_name}|{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def perturb(
    image: np.ndarray,
    mask: np.ndarray,
    dataset: str,
    image_name: str,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    pil_image = Image.fromarray(image.astype(np.uint8))
    if condition == "downsample_50":
        small = pil_image.resize(
            (max(1, pil_image.width // 2), max(1, pil_image.height // 2)),
            Image.Resampling.BILINEAR,
        )
        return (
            np.asarray(
                small.resize(pil_image.size, Image.Resampling.BILINEAR)
            ),
            mask,
        )
    if condition == "speckle_005":
        rng = np.random.default_rng(
            deterministic_seed(dataset, image_name, condition)
        )
        values = np.asarray(pil_image, dtype=np.float64) / 255.0
        noisy = values + values * rng.normal(0.0, 0.05, values.shape)
        return np.clip(noisy * 255.0, 0.0, 255.0).astype(np.uint8), mask
    if condition in ("gamma_08", "gamma_12"):
        gamma = 0.8 if condition == "gamma_08" else 1.2
        values = np.asarray(pil_image, dtype=np.float64) / 255.0
        return (
            np.clip(np.power(values, gamma) * 255.0, 0.0, 255.0).astype(
                np.uint8
            ),
            mask,
        )
    binary = mask > 0
    if condition == "mask_erode_3":
        return image, binary_erosion(binary, disk(3)).astype(np.uint8)
    if condition == "mask_dilate_3":
        return image, binary_dilation(binary, disk(3)).astype(np.uint8)
    raise KeyError(f"unknown perturbation condition: {condition}")


def extract_one(
    record: dict[str, object],
    dataset: str,
    condition: str,
) -> tuple[dict[str, object] | None, dict[str, str] | None, float]:
    started = time.perf_counter()
    try:
        with Image.open(str(record["image_path"])) as handle:
            image = np.asarray(handle.convert("L"))
        with Image.open(str(record["mask_path"])) as handle:
            mask = np.asarray(handle.convert("L"))
        image, mask = perturb(
            image,
            mask,
            dataset,
            str(record["image"]),
            condition,
        )
        row = {column: record[column] for column in METADATA_COLUMNS}
        row["condition"] = condition
        row.update(extract_feature_dict(image, mask))
        row.update(extract_multizone_feature_dict(image, mask))
        return row, None, time.perf_counter() - started
    except Exception as exc:
        return (
            None,
            {
                "image": str(record.get("image", "")),
                "condition": condition,
                "error": repr(exc),
            },
            time.perf_counter() - started,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    source_path = DATASETS[args.dataset]
    source = pd.read_csv(source_path)
    if args.dataset == "busuclm":
        source = source.loc[parse_bool(source["is_clean_primary"])].copy()
    source = source.sort_values(["patient_id", "image"]).reset_index(drop=True)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else ROOT / "results" / "perturbed_features" / args.dataset
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    records = source.to_dict(orient="records")
    summary: dict[str, object] = {
        "dataset": args.dataset,
        "source_features": str(source_path),
        "rows": int(len(source)),
        "groups": int(source["patient_id"].nunique()),
        "n_jobs": args.n_jobs,
        "conditions": {},
    }

    for condition in CONDITIONS:
        condition_started = time.perf_counter()
        outputs = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(extract_one)(record, args.dataset, condition)
            for record in records
        )
        rows = [row for row, _, _ in outputs if row is not None]
        failures = [failure for _, failure, _ in outputs if failure is not None]
        times = [elapsed for _, _, elapsed in outputs]
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / f"features_{condition}.csv", index=False)
        condition_summary = {
            "rows": int(len(frame)),
            "failures": failures,
            "wall_seconds": float(time.perf_counter() - condition_started),
            "per_image_seconds_mean": float(np.mean(times)),
            "per_image_seconds_median": float(np.median(times)),
        }
        summary["conditions"][condition] = condition_summary
        print(
            f"{args.dataset} {condition}: rows={len(frame)} "
            f"failures={len(failures)} "
            f"wall={condition_summary['wall_seconds']:.2f}s",
            flush=True,
        )
        if failures or len(frame) != len(source):
            (out_dir / "extraction_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{args.dataset} {condition}: "
                f"{len(failures)} failures, {len(frame)}/{len(source)} rows"
            )

    (out_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
