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
    "busbra": {
        "source": "accepted_manifest_busbra.csv",
        "pattern": "features_busbra_advanced_part_*.csv",
        "output": "features_busbra_advanced.csv",
    },
    "breast": {
        "source": "accepted_manifest_breast.csv",
        "pattern": "features_breast_advanced_part_*.csv",
        "output": "features_breast_advanced.csv",
    },
}


def part_start(path: Path) -> int:
    match = PART_PATTERN.search(path.name)
    if match is None:
        raise RuntimeError(f"invalid part filename: {path}")
    return int(match.group(1))


def merge_dataset(
    dataset: str,
    specification: dict[str, str],
) -> dict[str, object]:
    source = pd.read_csv(RESULT_DIR / specification["source"])
    parts = sorted(
        RESULT_DIR.glob(specification["pattern"]),
        key=part_start,
    )
    if not parts:
        raise RuntimeError(f"{dataset}: no feature part files")
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
    advanced = [
        column for column in frame if column.startswith("advanced_")
    ]
    if len(advanced) != 76:
        raise RuntimeError(
            f"{dataset}: {len(advanced)} advanced columns, expected 76"
        )
    values = frame[advanced].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{dataset}: non-finite advanced values")
    output = RESULT_DIR / specification["output"]
    frame.to_csv(output, index=False)
    return {
        "dataset": dataset,
        "source": str((RESULT_DIR / specification["source"]).resolve()),
        "parts": [str(path.resolve()) for path in parts],
        "output": str(output.resolve()),
        "rows": int(len(frame)),
        "groups": int(frame["patient_id"].nunique()),
        "advanced_features": len(advanced),
        "finite_values": int(values.size),
        "row_order_matches_manifest": True,
    }


def main() -> int:
    summaries = [
        merge_dataset(dataset, specification)
        for dataset, specification in DATASETS.items()
    ]
    result = {
        "protocol": 'V04_LOCKED_EXTERNAL_PROTOCOL_V1',
        "datasets": summaries,
        "failures": [],
    }
    (RESULT_DIR / "external_advanced_feature_audit.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
