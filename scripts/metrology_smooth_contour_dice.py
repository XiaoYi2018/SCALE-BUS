"""Dice of the deterministic 3-pixel erosion/dilation masks against the supplied masks.

Uses the same native-resolution frame and the same morphological operators as
extract_v04_external_mask_perturbations.py, and the same Dice definition as
metrology_uncertainty_propagation.py. Read-only on all existing results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from metrology_paths import OUTPUT_ROOT

from extract_v04_external_mask_perturbations import (  # noqa: E402
    CONDITIONS,
    DATASETS,
    INPUT_DIR,
    perturb_mask,
)

OUTPUT_DIR = OUTPUT_ROOT / "v07_smooth_contour_dice"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    rows = []
    for dataset in DATASETS:
        source = pd.read_csv(INPUT_DIR / f"accepted_manifest_{dataset}.csv")
        for record in source.to_dict(orient="records"):
            with Image.open(str(record["mask_path"])) as handle:
                mask = np.asarray(handle.convert("L"))
            a = mask > 0
            radius = float(np.sqrt(a.sum() / np.pi))
            for condition in CONDITIONS:
                b = perturb_mask(mask, condition) > 0
                dice = 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())
                rows.append(
                    {
                        "dataset": dataset,
                        "image": record["image"],
                        "condition": condition,
                        "native_equivalent_radius": radius,
                        "Dice": float(dice),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "smooth_contour_dice.csv", index=False)
    summary = (
        frame.groupby(["dataset", "condition"])
        .agg(
            n=("Dice", "size"),
            median=("Dice", "median"),
            q25=("Dice", lambda x: x.quantile(0.25)),
            q75=("Dice", lambda x: x.quantile(0.75)),
            radius_median=("native_equivalent_radius", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "smooth_contour_dice_summary.csv", index=False)
    script = Path(__file__).read_bytes()
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "script_sha256": hashlib.sha256(script).hexdigest(),
                "frame": "native-resolution supplied masks, as in v04 erosion/dilation and v07 E3 Dice",
                "operators": "skimage binary_erosion/binary_dilation with disk(3)",
                "dice": "2|A∩B|/(|A|+|B|)",
                "summary": summary.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
