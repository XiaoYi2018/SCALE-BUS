from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "v04_external_integrity"
BUSBRA_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "BUS-BRA_zenodo_8231412"
    / "BUSBRA"
    / "BUSBRA"
)
BREAST_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "BrEaST_TCIA_9WKK-Q141"
    / "BrEaST-Lesions_USG-images_and_masks"
)
BREAST_CLINICAL = (
    ROOT
    / "data"
    / "raw"
    / "BrEaST_TCIA_9WKK-Q141"
    / "BrEaST-Lesions-USG-clinical-data.xlsx"
)
DEVELOPMENT_DIR = ROOT / "results" / "v04_advanced_features"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_audit(path: Path) -> dict[str, object]:
    with Image.open(path) as handle:
        gray = handle.convert("L")
        array = np.asarray(gray)
        resized = np.asarray(
            gray.resize((9, 8), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    pixel_digest = hashlib.sha256()
    pixel_digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    pixel_digest.update(array.tobytes())
    bits = resized[:, 1:] > resized[:, :-1]
    dhash_value = 0
    for bit in bits.ravel():
        dhash_value = (dhash_value << 1) | int(bit)
    return {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "file_sha256": file_sha256(path),
        "pixel_sha256": pixel_digest.hexdigest(),
        "dhash64": f"{dhash_value:016x}",
    }


def mask_audit(path: Path) -> dict[str, object]:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("L"))
    foreground = array > 0
    return {
        "mask_width": int(array.shape[1]),
        "mask_height": int(array.shape[0]),
        "mask_foreground_pixels": int(foreground.sum()),
        "mask_foreground_fraction": float(foreground.mean()),
        "mask_file_sha256": file_sha256(path),
    }


def development_manifest(dataset: str, path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    patient_column = "cv_group_id" if dataset == "BUSI" else "patient_id"
    rows: list[dict[str, object]] = []
    for number, record in enumerate(source.to_dict(orient="records"), 1):
        image_path = Path(record["image_path"])
        mask_path = Path(record["mask_path"])
        image = image_audit(image_path)
        mask = mask_audit(mask_path)
        valid = (
            image_path.is_file()
            and mask_path.is_file()
            and image["width"] == mask["mask_width"]
            and image["height"] == mask["mask_height"]
            and mask["mask_foreground_pixels"] > 0
            and str(record["label"]) in {"benign", "malignant"}
        )
        rows.append(
            {
                "dataset": dataset,
                "manifest_order": number,
                "image": str(record["image"]),
                "patient_id": str(record[patient_column]),
                "label": str(record["label"]),
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "accepted": bool(valid),
                "exclusion_reason": "" if valid else "development_row_invalid",
                "birads": "",
                "device": "",
                "histology": "",
                "verification": "",
                "diagnosis": "",
                **image,
                **mask,
            }
        )
    return pd.DataFrame(rows)


def busbra_manifest() -> pd.DataFrame:
    metadata = pd.read_csv(BUSBRA_ROOT / "bus_data.csv")
    image_dir = BUSBRA_ROOT / "Images"
    mask_dir = BUSBRA_ROOT / "Masks"
    rows: list[dict[str, object]] = []
    for number, record in enumerate(metadata.to_dict(orient="records"), 1):
        image_name = f"{record['ID']}.png"
        suffix = str(record["ID"]).removeprefix("bus_")
        mask_name = f"mask_{suffix}.png"
        image_path = image_dir / image_name
        mask_path = mask_dir / mask_name
        reason = ""
        if not image_path.is_file():
            reason = "missing_image"
        elif not mask_path.is_file():
            reason = "missing_mask"
        elif str(record["Pathology"]) not in {"benign", "malignant"}:
            reason = "unsupported_label"
        if reason:
            rows.append(
                {
                    "dataset": "BUS-BRA",
                    "manifest_order": number,
                    "image": image_name,
                    "patient_id": str(record["Case"]),
                    "label": str(record["Pathology"]),
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "accepted": False,
                    "exclusion_reason": reason,
                    "birads": str(record["BIRADS"]),
                    "device": str(record["Device"]),
                    "histology": str(record["Histology"]),
                    "verification": "biopsy",
                    "diagnosis": str(record["Histology"]),
                }
            )
            continue
        image = image_audit(image_path)
        mask = mask_audit(mask_path)
        if (
            image["width"] != mask["mask_width"]
            or image["height"] != mask["mask_height"]
        ):
            reason = "image_mask_geometry_mismatch"
        elif mask["mask_foreground_pixels"] <= 0:
            reason = "empty_mask"
        rows.append(
            {
                "dataset": "BUS-BRA",
                "manifest_order": number,
                "image": image_name,
                "patient_id": str(record["Case"]),
                "label": str(record["Pathology"]),
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "accepted": not bool(reason),
                "exclusion_reason": reason,
                "birads": str(record["BIRADS"]),
                "device": str(record["Device"]),
                "histology": str(record["Histology"]),
                "verification": "biopsy",
                "diagnosis": str(record["Histology"]),
                **image,
                **mask,
            }
        )
    return pd.DataFrame(rows)


def breast_manifest() -> pd.DataFrame:
    metadata = pd.read_excel(
        BREAST_CLINICAL,
        sheet_name="BrEaST-Lesions-USG clinical dat",
    )
    rows: list[dict[str, object]] = []
    for number, record in enumerate(metadata.to_dict(orient="records"), 1):
        image_name = str(record["Image_filename"])
        mask_name = str(record["Mask_tumor_filename"])
        image_path = BREAST_ROOT / image_name
        mask_path = BREAST_ROOT / mask_name
        label = str(record["Classification"]).strip().lower()
        reason = ""
        if not image_path.is_file():
            reason = "missing_image"
        elif not mask_path.is_file():
            reason = "missing_mask"
        elif label not in {"benign", "malignant"}:
            reason = "unsupported_label"
        if reason:
            rows.append(
                {
                    "dataset": "BrEaST",
                    "manifest_order": number,
                    "image": image_name,
                    "patient_id": str(record["CaseID"]),
                    "label": label,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "accepted": False,
                    "exclusion_reason": reason,
                    "birads": str(record["BIRADS"]),
                    "device": "",
                    "histology": "",
                    "verification": str(record["Verification"]),
                    "diagnosis": str(record["Diagnosis"]),
                }
            )
            continue
        image = image_audit(image_path)
        mask = mask_audit(mask_path)
        if (
            image["width"] != mask["mask_width"]
            or image["height"] != mask["mask_height"]
        ):
            reason = "image_mask_geometry_mismatch"
        elif mask["mask_foreground_pixels"] <= 0:
            reason = "empty_mask"
        rows.append(
            {
                "dataset": "BrEaST",
                "manifest_order": number,
                "image": image_name,
                "patient_id": str(record["CaseID"]),
                "label": label,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "accepted": not bool(reason),
                "exclusion_reason": reason,
                "birads": str(record["BIRADS"]),
                "device": "",
                "histology": "",
                "verification": str(record["Verification"]),
                "diagnosis": str(record["Diagnosis"]),
                **image,
                **mask,
            }
        )
    return pd.DataFrame(rows)


def exact_cross_dataset_overlaps(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    accepted = frame.loc[frame["accepted"]].copy()
    for hash_column in ("file_sha256", "pixel_sha256"):
        for value, group in accepted.groupby(hash_column):
            datasets = sorted(group["dataset"].unique())
            if len(datasets) < 2:
                continue
            rows.append(
                {
                    "hash_type": hash_column,
                    "hash": value,
                    "datasets": "|".join(datasets),
                    "rows": int(len(group)),
                    "images": "|".join(
                        f"{row.dataset}:{row.image}"
                        for row in group.itertuples(index=False)
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["hash_type", "hash", "datasets", "rows", "images"],
    )


def perceptual_candidates(
    frame: pd.DataFrame,
    threshold: int = 4,
) -> pd.DataFrame:
    accepted = frame.loc[frame["accepted"]].reset_index(drop=True)
    by_dataset = {
        dataset: group.reset_index(drop=True)
        for dataset, group in accepted.groupby("dataset")
    }
    datasets = sorted(by_dataset)
    rows: list[dict[str, object]] = []
    for left_index, left_name in enumerate(datasets):
        left = by_dataset[left_name]
        left_hash = [int(value, 16) for value in left["dhash64"]]
        for right_name in datasets[left_index + 1 :]:
            right = by_dataset[right_name]
            right_hash = [int(value, 16) for value in right["dhash64"]]
            for i, hash_left in enumerate(left_hash):
                for j, hash_right in enumerate(right_hash):
                    distance = (hash_left ^ hash_right).bit_count()
                    if distance <= threshold:
                        rows.append(
                            {
                                "dataset_left": left_name,
                                "image_left": left.iloc[i]["image"],
                                "dataset_right": right_name,
                                "image_right": right.iloc[j]["image"],
                                "dhash_hamming": distance,
                                "pixel_exact": (
                                    left.iloc[i]["pixel_sha256"]
                                    == right.iloc[j]["pixel_sha256"]
                                ),
                            }
                        )
    return pd.DataFrame(
        rows,
        columns=[
            "dataset_left",
            "image_left",
            "dataset_right",
            "image_right",
            "dhash_hamming",
            "pixel_exact",
        ],
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {
        "BUS-UCLM": development_manifest(
            "BUS-UCLM",
            DEVELOPMENT_DIR / "features_busuclm_advanced.csv",
        ),
        "BUSI": development_manifest(
            "BUSI",
            DEVELOPMENT_DIR / "features_busi_advanced.csv",
        ),
        "BUS-BRA": busbra_manifest(),
        "BrEaST": breast_manifest(),
    }
    for dataset, frame in manifests.items():
        frame.to_csv(
            OUTPUT_DIR / f"manifest_{dataset.lower().replace('-', '_')}.csv",
            index=False,
        )
    combined = pd.concat(manifests.values(), ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "manifest_all.csv", index=False)
    exclusions = combined.loc[~combined["accepted"]].copy()
    exclusions.to_csv(OUTPUT_DIR / "exclusions.csv", index=False)
    exact = exact_cross_dataset_overlaps(combined)
    exact.to_csv(OUTPUT_DIR / "cross_dataset_exact_overlaps.csv", index=False)
    perceptual = perceptual_candidates(combined)
    perceptual.to_csv(
        OUTPUT_DIR / "cross_dataset_perceptual_candidates_dhash_le4.csv",
        index=False,
    )

    summaries = []
    for dataset, frame in manifests.items():
        accepted = frame.loc[frame["accepted"]]
        summaries.append(
            {
                "dataset": dataset,
                "rows_total": int(len(frame)),
                "rows_accepted": int(len(accepted)),
                "rows_excluded": int((~frame["accepted"]).sum()),
                "groups_accepted": int(accepted["patient_id"].nunique()),
                "benign": int((accepted["label"] == "benign").sum()),
                "malignant": int(
                    (accepted["label"] == "malignant").sum()
                ),
                "devices": sorted(
                    value
                    for value in accepted["device"].astype(str).unique()
                    if value and value != "nan"
                ),
            }
        )
    result = {
        "protocol": 'V04_LOCKED_EXTERNAL_PROTOCOL_V1',
        "datasets": summaries,
        "cross_dataset_exact_overlap_groups": int(len(exact)),
        "cross_dataset_dhash_le4_candidates": int(len(perceptual)),
        "duplicate_birads_copy_tree_excluded_by_design": {
            "path": str(
                (
                    BUSBRA_ROOT.parents[1]
                    / "busbra_birads"
                    / "busbra_birads"
                ).resolve()
            ),
            "files": 1875,
        },
        "files": {
            "combined_manifest": str(
                (OUTPUT_DIR / "manifest_all.csv").resolve()
            ),
            "exclusions": str((OUTPUT_DIR / "exclusions.csv").resolve()),
            "exact_overlaps": str(
                (
                    OUTPUT_DIR / "cross_dataset_exact_overlaps.csv"
                ).resolve()
            ),
            "perceptual_candidates": str(
                (
                    OUTPUT_DIR
                    / "cross_dataset_perceptual_candidates_dhash_le4.csv"
                ).resolve()
            ),
        },
    }
    (OUTPUT_DIR / "integrity_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
