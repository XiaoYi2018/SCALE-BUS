from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_v04_advanced_features import extract_dataset  # noqa: E402


INPUT_DIR = ROOT / "results" / "v04_external_features"
OUTPUT_DIR = ROOT / "results" / "v04_external_features"
DATASETS = {
    "busbra": {
        "name": "BUS-BRA-locked",
        "source": "accepted_manifest_busbra.csv",
        "stem": "features_busbra_advanced",
    },
    "breast": {
        "name": "BrEaST-locked",
        "source": "accepted_manifest_breast.csv",
        "stem": "features_breast_advanced",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.start < 0 or args.limit < 0:
        raise ValueError("--start and --limit must be non-negative")
    specification = DATASETS[args.dataset]
    if args.start or args.limit:
        output_name = (
            f"{specification['stem']}_part_{args.start:04d}_"
            f"n{args.limit:04d}.csv"
        )
    else:
        output_name = f"{specification['stem']}.csv"
    summary = extract_dataset(
        specification["name"],
        INPUT_DIR / specification["source"],
        OUTPUT_DIR / output_name,
        start=args.start,
        limit=args.limit,
    )
    summary["protocol"] = 'V04_LOCKED_EXTERNAL_PROTOCOL_V1'
    summary_path = OUTPUT_DIR / (
        f"extraction_{args.dataset}_part_{args.start:04d}_"
        f"n{args.limit:04d}.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

