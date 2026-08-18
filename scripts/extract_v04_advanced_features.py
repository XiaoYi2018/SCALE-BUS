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

from fractal_extrema import extract_advanced_feature_dict  # noqa: E402


DEFAULT_INPUT_DIR = ROOT / "results" / "component_tree_comparator"
DEFAULT_OUT_DIR = ROOT / "results" / "v04_advanced_features"
DATASETS = {
    "BUS-UCLM-clean": "features_busuclm_clean.csv",
    "BUSI-valid": "features_busi_valid.csv",
}


def extract_dataset(
    dataset_name: str,
    source_path: Path,
    output_path: Path,
    start: int = 0,
    limit: int = 0,
) -> dict[str, object]:
    source_all = pd.read_csv(source_path)
    stop = len(source_all) if limit <= 0 else min(len(source_all), start + limit)
    source = source_all.iloc[start:stop].copy().reset_index(drop=True)
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
            row.update(extract_advanced_feature_dict(image, mask))
            rows.append(row)
            if position == 1 or position % 25 == 0 or position == len(source):
                print(
                    f"[advanced] {dataset_name}: {position}/{len(source)}",
                    flush=True,
                )
        except Exception as exc:
            failures.append(
                {
                    "image": str(record.get("image", "")),
                    "group": str(
                        record.get(
                            "cv_group_id",
                            record.get("patient_id", ""),
                        )
                    ),
                    "error": repr(exc),
                }
            )
            print(
                f"[advanced] {dataset_name}: FAILED "
                f"{record.get('image')}: {exc!r}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    advanced_columns = [
        column for column in frame if column.startswith("advanced_")
    ]
    if len(advanced_columns) != 76:
        raise RuntimeError(
            f"{dataset_name} advanced column count is "
            f"{len(advanced_columns)}, expected 76"
        )
    if len(frame) != len(source):
        raise RuntimeError(
            f"{dataset_name} row count is {len(frame)}, expected {len(source)}"
        )
    values = frame[advanced_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{dataset_name} has non-finite advanced features")

    frame.to_csv(output_path, index=False)
    group_column = (
        "cv_group_id" if dataset_name == "BUSI-valid" else "patient_id"
    )
    return {
        "dataset": dataset_name,
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "source_rows_total": int(len(source_all)),
        "slice_start": int(start),
        "slice_stop": int(stop),
        "rows": int(len(frame)),
        "groups": int(frame[group_column].nunique()),
        "advanced_features": len(advanced_columns),
        "feature_names": advanced_columns,
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dataset",
        choices=("all", "busuclm", "busi"),
        default="all",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.start < 0 or args.limit < 0:
        raise ValueError("--start and --limit must be non-negative")

    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    started = time.perf_counter()

    selected = {
        "all": set(DATASETS),
        "busuclm": {"BUS-UCLM-clean"},
        "busi": {"BUSI-valid"},
    }[args.dataset]
    for dataset_name, filename in DATASETS.items():
        if dataset_name not in selected:
            continue
        stem = (
            "features_busuclm_advanced.csv"
            if dataset_name == "BUS-UCLM-clean"
            else "features_busi_advanced.csv"
        )
        if args.start or args.limit:
            stem_path = Path(stem)
            suffix = f"_part_{args.start:04d}_n{args.limit:04d}"
            output_name = f"{stem_path.stem}{suffix}{stem_path.suffix}"
        else:
            output_name = stem
        summaries.append(
            extract_dataset(
                dataset_name,
                input_dir / filename,
                out_dir / output_name,
                start=args.start,
                limit=args.limit,
            )
        )

    summary = {
        "protocol": 'V04_DEVELOPMENT_PROTOCOL_V1',
        "amendment": 'V04_DEVELOPMENT_PROTOCOL_V1_AMENDMENT_20260718',
        "datasets": summaries,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    summary_name = (
        "extraction_summary.json"
        if not (args.start or args.limit)
        else (
            f"extraction_summary_{args.dataset}_"
            f"part_{args.start:04d}_n{args.limit:04d}.json"
        )
    )
    (out_dir / summary_name).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
