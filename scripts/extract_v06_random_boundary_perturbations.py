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
from scipy.ndimage import distance_transform_edt
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import remove_small_holes


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fractal_extrema import extract_advanced_feature_dict  # noqa: E402


INPUT_DIR = ROOT / "results" / "v04_external_features"
OUTPUT_DIR = ROOT / "results" / "v06_random_boundary_perturbations"
DATASETS = ("busbra", "breast")
AMPLITUDES = (0.025, 0.050)
SEEDS = (20260718, 20260719, 20260720, 20260721, 20260722)


def deterministic_seed(seed: int, image: str) -> int:
    digest = hashlib.sha256(f"{seed}|{image}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def largest_component(binary: np.ndarray) -> np.ndarray:
    components = label(binary, connectivity=2)
    values, counts = np.unique(components[components > 0], return_counts=True)
    if not len(values):
        raise RuntimeError("perturbation produced an empty mask")
    selected = values[int(np.argmax(counts))]
    output = components == selected
    return remove_small_holes(output, area_threshold=max(16, int(0.002 * output.sum())))


def perturb_mask(mask: np.ndarray, amplitude: float, seed: int, image: str) -> np.ndarray:
    binary = mask > 0
    equivalent_radius = float(np.sqrt(binary.sum() / np.pi))
    if equivalent_radius < 4:
        raise RuntimeError("mask too small for scale-normalized perturbation")
    signed_distance = distance_transform_edt(binary) - distance_transform_edt(~binary)
    rng = np.random.default_rng(deterministic_seed(seed, image))
    random_field = rng.normal(size=binary.shape)
    correlation = max(2.0, 0.06 * equivalent_radius)
    random_field = gaussian(
        random_field,
        sigma=correlation,
        preserve_range=True,
        mode="reflect",
    )
    random_field = random_field - float(np.mean(random_field))
    scale = float(np.std(random_field))
    if scale <= 1e-8:
        raise RuntimeError("degenerate random field")
    random_field /= scale
    displacement = amplitude * equivalent_radius * random_field
    return largest_component(signed_distance + displacement > 0).astype(np.uint8)


def extract_one(
    record: dict[str, object],
    amplitude: float,
    seed: int,
) -> tuple[dict[str, object] | None, dict[str, str] | None, float]:
    started = time.perf_counter()
    condition = f"jitter_{int(round(amplitude * 1000)):03d}_seed_{seed}"
    try:
        with Image.open(str(record["image_path"])) as handle:
            image = np.asarray(handle.convert("L"))
        with Image.open(str(record["mask_path"])) as handle:
            mask = np.asarray(handle.convert("L"))
        perturbed = perturb_mask(mask, amplitude, seed, str(record["image"]))
        row = dict(record)
        row["condition"] = condition
        row["jitter_amplitude_equivalent_radius"] = amplitude
        row["jitter_seed"] = seed
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
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_path = INPUT_DIR / f"accepted_manifest_{args.dataset}.csv"
    source = pd.read_csv(source_path).sort_values(["patient_id", "image"]).reset_index(drop=True)
    records = source.to_dict(orient="records")
    summary: dict[str, object] = {
        "dataset": args.dataset,
        "source": str(source_path.resolve()),
        "rows": len(source),
        "groups": source["patient_id"].astype(str).nunique(),
        "n_jobs": args.n_jobs,
        "definition": (
            "signed-distance boundary displaced by a Gaussian-correlated random field; "
            "amplitude expressed as a fraction of equivalent lesion radius; largest "
            "connected component retained"
        ),
        "conditions": {},
    }
    for amplitude in AMPLITUDES:
        for seed in SEEDS:
            condition = f"jitter_{int(round(amplitude * 1000)):03d}_seed_{seed}"
            started = time.perf_counter()
            outputs = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                delayed(extract_one)(record, amplitude, seed) for record in records
            )
            rows = [row for row, _, _ in outputs if row is not None]
            failures = [failure for _, failure, _ in outputs if failure is not None]
            elapsed = [value for _, _, value in outputs]
            frame = pd.DataFrame(rows)
            advanced = [column for column in frame if column.startswith("advanced_")]
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
                (OUTPUT_DIR / f"{args.dataset}_summary_partial.json").write_text(
                    json.dumps(summary, indent=2),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"{args.dataset} {condition}: "
                    f"{len(failures)} failures, {len(frame)}/{len(source)} rows"
                )
            values = frame[advanced].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"{args.dataset} {condition}: non-finite features")
            output_path = OUTPUT_DIR / f"features_{args.dataset}_{condition}.csv"
            frame.to_csv(output_path, index=False)
            area_ratio = (
                frame["perturbed_mask_foreground_pixels"].to_numpy(float)
                / source["mask_foreground_pixels"].to_numpy(float)
            )
            summary["conditions"][condition] = {
                "output": str(output_path.resolve()),
                "rows": len(frame),
                "failures": failures,
                "wall_seconds": time.perf_counter() - started,
                "per_image_seconds_mean": float(np.mean(elapsed)),
                "foreground_fraction_median": float(np.median(area_ratio)),
                "foreground_fraction_q05": float(np.quantile(area_ratio, 0.05)),
                "foreground_fraction_q95": float(np.quantile(area_ratio, 0.95)),
            }
            (OUTPUT_DIR / f"{args.dataset}_summary_partial.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
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
