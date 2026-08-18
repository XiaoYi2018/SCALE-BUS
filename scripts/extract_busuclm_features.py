from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fractal_extrema import (  # noqa: E402
    COMPACT_EXTREMA_FEATURES,
    extract_feature_dict,
)


DEFAULT_MANIFEST = (
    ROOT / "results" / "busuclm_audit" / "manifest_lesions_valid.csv"
)
DEFAULT_OUT_DIR = ROOT / "results" / "busuclm_features"
METADATA_COLUMNS = (
    "image",
    "patient_id",
    "label",
    "doppler",
    "marks",
    "combined",
    "is_clean_primary",
    "foreground_fraction",
    "image_path",
    "mask_path",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for position, record in enumerate(manifest.to_dict(orient="records"), 1):
        try:
            with Image.open(record["image_path"]) as handle:
                image = np.asarray(handle.convert("L"))
            with Image.open(record["mask_path"]) as handle:
                mask = np.asarray(handle.convert("L"))
            extracted = extract_feature_dict(image, mask)
            row = {column: record[column] for column in METADATA_COLUMNS}
            row.update(extracted)
            rows.append(row)
            print(
                f"[{position:03d}/{len(manifest):03d}] "
                f"{record['image']} {record['label']}",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "image": str(record.get("image", "")),
                    "patient_id": str(record.get("patient_id", "")),
                    "error": repr(exc),
                }
            )
            print(
                f"[{position:03d}/{len(manifest):03d}] "
                f"FAILED {record.get('image')}: {exc!r}",
                flush=True,
            )

    features = pd.DataFrame(rows)
    features.to_csv(out_dir / "features_all_valid.csv", index=False)
    summary = {
        "manifest": str(manifest_path),
        "rows_manifest": int(len(manifest)),
        "rows_features": int(len(features)),
        "patients": int(features["patient_id"].nunique()) if len(features) else 0,
        "failures": failures,
        "feature_count": int(
            sum(
                column.startswith(("basic_", "fractal_", "extrema_"))
                for column in features.columns
            )
        ),
        "compact_extrema_count": len(COMPACT_EXTREMA_FEATURES),
        "compact_extrema_features": list(COMPACT_EXTREMA_FEATURES),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (out_dir / "feature_extraction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if failures:
        (out_dir / "feature_failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(failures)} feature extraction failures")
    if len(features) != len(manifest):
        raise RuntimeError("feature row count does not match manifest")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
