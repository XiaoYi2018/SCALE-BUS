from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "v04_external_features"
PART_PATTERN = re.compile(r"_part_(\d+)_n(\d+)\.csv$")
DATASETS = {
    "busbra": "accepted_manifest_busbra.csv",
    "breast": "accepted_manifest_breast.csv",
}


def part_start(path: Path) -> int:
    match = PART_PATTERN.search(path.name)
    if match is None:
        raise RuntimeError(f"invalid part filename: {path}")
    return int(match.group(1))


def merge_dataset(dataset: str, source_name: str) -> dict[str, object]:
    source = pd.read_csv(RESULT_DIR / source_name)
    parts = sorted(
        RESULT_DIR.glob(f"features_{dataset}_frozen_part_*.csv"),
        key=part_start,
    )
    if not parts:
        raise RuntimeError(f"{dataset}: no frozen feature parts")
    frame = pd.concat(
        [pd.read_csv(path) for path in parts],
        ignore_index=True,
    )
    if len(frame) != len(source):
        raise RuntimeError(
            f"{dataset}: {len(frame)} rows, expected {len(source)}"
        )
    if frame["image"].astype(str).tolist() != source["image"].astype(str).tolist():
        raise RuntimeError(f"{dataset}: row order does not match manifest")
    frozen = [
        column
        for column in frame
        if column.startswith(("basic_", "fractal_", "zone_"))
    ]
    if len(frozen) != 98:
        raise RuntimeError(
            f"{dataset}: {len(frozen)} frozen columns, expected 98"
        )
    values = frame[frozen].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{dataset}: non-finite frozen values")
    output = RESULT_DIR / f"features_{dataset}_frozen.csv"
    frame.to_csv(output, index=False)
    return {
        "dataset": dataset,
        "source": str((RESULT_DIR / source_name).resolve()),
        "parts": [str(path.resolve()) for path in parts],
        "output": str(output.resolve()),
        "rows": int(len(frame)),
        "groups": int(frame["patient_id"].nunique()),
        "frozen_features": len(frozen),
        "finite_values": int(values.size),
        "row_order_matches_manifest": True,
    }


def main() -> int:
    summaries = [
        merge_dataset(dataset, source_name)
        for dataset, source_name in DATASETS.items()
    ]
    result = {
        "protocol": 'V04_LOCKED_EXTERNAL_PROTOCOL_V1',
        "datasets": summaries,
        "failures": [],
    }
    (RESULT_DIR / "external_frozen_feature_audit.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
