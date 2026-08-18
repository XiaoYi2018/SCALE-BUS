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
    extract_feature_dict,
    extract_multizone_feature_dict,
)


INPUT_DIR = ROOT / "results" / "v04_external_features"
OUTPUT_DIR = ROOT / "results" / "v04_external_features"
DATASETS = {
    "busbra": "accepted_manifest_busbra.csv",
    "breast": "accepted_manifest_breast.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    source_path = INPUT_DIR / DATASETS[args.dataset]
    source_all = pd.read_csv(source_path)
    stop = (
        len(source_all)
        if args.limit <= 0
        else min(len(source_all), args.start + args.limit)
    )
    source = source_all.iloc[args.start:stop].copy().reset_index(drop=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for position, record in enumerate(source.to_dict(orient="records"), 1):
        try:
            with Image.open(record["image_path"]) as handle:
                image = np.asarray(handle.convert("L"))
            with Image.open(record["mask_path"]) as handle:
                mask = np.asarray(handle.convert("L"))
            row = dict(record)
            row.update(extract_feature_dict(image, mask))
            row.update(extract_multizone_feature_dict(image, mask))
            rows.append(row)
            if position == 1 or position % 25 == 0 or position == len(source):
                print(
                    f"[frozen] {args.dataset}: {position}/{len(source)}",
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
                f"[frozen] {args.dataset}: FAILED "
                f"{record.get('image')}: {exc!r}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frozen_columns = [
        column
        for column in frame
        if column.startswith(("basic_", "fractal_", "zone_"))
    ]
    if len(frozen_columns) != 98:
        raise RuntimeError(
            f"{args.dataset}: {len(frozen_columns)} frozen columns, "
            "expected 98"
        )
    if len(frame) != len(source):
        raise RuntimeError(
            f"{args.dataset}: {len(frame)} rows, expected {len(source)}"
        )
    values = frame[frozen_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{args.dataset}: non-finite frozen features")
    output = OUTPUT_DIR / (
        f"features_{args.dataset}_frozen_part_{args.start:04d}_"
        f"n{args.limit:04d}.csv"
    )
    frame.to_csv(output, index=False)
    result = {
        "protocol": 'V04_LOCKED_EXTERNAL_PROTOCOL_V1',
        "dataset": args.dataset,
        "source": str(source_path.resolve()),
        "output": str(output.resolve()),
        "source_rows_total": int(len(source_all)),
        "slice_start": int(args.start),
        "slice_stop": int(stop),
        "rows": int(len(frame)),
        "groups": int(frame["patient_id"].nunique()),
        "frozen_features": len(frozen_columns),
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (OUTPUT_DIR / (
        f"extraction_{args.dataset}_frozen_part_{args.start:04d}_"
        f"n{args.limit:04d}.json"
    )).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

