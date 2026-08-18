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

from fractal_extrema import extract_multizone_feature_dict  # noqa: E402


DEFAULT_FEATURES = ROOT / "results" / "busuclm_features" / "features_all_valid.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "busuclm_multizone_features"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    base = pd.read_csv(args.features.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for position, record in enumerate(base.to_dict(orient="records"), 1):
        try:
            with Image.open(record["image_path"]) as handle:
                image = np.asarray(handle.convert("L"))
            with Image.open(record["mask_path"]) as handle:
                mask = np.asarray(handle.convert("L"))
            row = dict(record)
            row.update(extract_multizone_feature_dict(image, mask))
            rows.append(row)
            print(
                f"[{position:03d}/{len(base):03d}] {record['image']}",
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
                f"[{position:03d}/{len(base):03d}] "
                f"FAILED {record.get('image')}: {exc!r}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "features_multizone.csv", index=False)
    zone_columns = [column for column in frame if column.startswith("zone_")]
    summary = {
        "base_features": str(args.features.resolve()),
        "rows_base": int(len(base)),
        "rows_features": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()) if len(frame) else 0,
        "multizone_feature_count": len(zone_columns),
        "multizone_features": zone_columns,
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (out_dir / "multizone_extraction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if failures:
        (out_dir / "multizone_failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(failures)} multizone feature failures")
    if len(frame) != len(base):
        raise RuntimeError("multizone feature row count does not match base")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
