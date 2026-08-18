from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTEGRITY_DIR = ROOT / "results" / "v04_external_integrity"
OUTPUT_DIR = ROOT / "results" / "v04_external_features"
DATASETS = {
    "busbra": INTEGRITY_DIR / "manifest_bus_bra.csv",
    "breast": INTEGRITY_DIR / "manifest_breast.csv",
}


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset, path in DATASETS.items():
        frame = pd.read_csv(path)
        accepted = frame.loc[parse_bool(frame["accepted"])].copy()
        accepted = accepted.sort_values("manifest_order").reset_index(drop=True)
        output = OUTPUT_DIR / f"accepted_manifest_{dataset}.csv"
        accepted.to_csv(output, index=False)
        summaries.append(
            {
                "dataset": dataset,
                "source": str(path.resolve()),
                "output": str(output.resolve()),
                "rows": int(len(accepted)),
                "groups": int(accepted["patient_id"].nunique()),
                "benign": int((accepted["label"] == "benign").sum()),
                "malignant": int((accepted["label"] == "malignant").sum()),
            }
        )
    result = {
        "protocol": 'V04_LOCKED_EXTERNAL_PROTOCOL_V1',
        "datasets": summaries,
    }
    (OUTPUT_DIR / "accepted_manifest_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

