from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "v04_advanced_features"
SOURCE_DIR = ROOT / "results" / "component_tree_comparator"


def audit_frame(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    dataset: str,
    group_column: str,
) -> dict[str, object]:
    advanced = [
        column for column in frame if column.startswith("advanced_")
    ]
    if len(frame) != len(source):
        raise RuntimeError(
            f"{dataset}: {len(frame)} rows, expected {len(source)}"
        )
    if frame["image"].astype(str).tolist() != source["image"].astype(str).tolist():
        raise RuntimeError(f"{dataset}: row order does not match source")
    if len(advanced) != 76:
        raise RuntimeError(
            f"{dataset}: {len(advanced)} advanced columns, expected 76"
        )
    values = frame[advanced].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{dataset}: non-finite advanced values")
    return {
        "dataset": dataset,
        "rows": int(len(frame)),
        "groups": int(frame[group_column].nunique()),
        "advanced_features": len(advanced),
        "finite_values": int(values.size),
        "row_order_matches_source": True,
    }


def main() -> int:
    parts = sorted(RESULT_DIR.glob("features_busi_advanced_part_*.csv"))
    if not parts:
        raise RuntimeError("no BUSI advanced-feature part files found")
    busi = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
    busi_source = pd.read_csv(SOURCE_DIR / "features_busi_valid.csv")
    busi_audit = audit_frame(
        busi,
        busi_source,
        "BUSI-valid",
        "cv_group_id",
    )
    busi_output = RESULT_DIR / "features_busi_advanced.csv"
    busi.to_csv(busi_output, index=False)

    busuclm_path = RESULT_DIR / "features_busuclm_advanced.csv"
    busuclm = pd.read_csv(busuclm_path)
    busuclm_source = pd.read_csv(
        SOURCE_DIR / "features_busuclm_clean.csv"
    )
    busuclm_audit = audit_frame(
        busuclm,
        busuclm_source,
        "BUS-UCLM-clean",
        "patient_id",
    )

    summary = {
        "protocol": 'V04_DEVELOPMENT_PROTOCOL_V1',
        "amendment": 'V04_DEVELOPMENT_PROTOCOL_V1_AMENDMENT_20260718',
        "busi_parts": [str(path.resolve()) for path in parts],
        "datasets": [busuclm_audit, busi_audit],
        "total_rows": int(len(busuclm) + len(busi)),
        "failures": [],
    }
    (RESULT_DIR / "advanced_feature_audit.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

