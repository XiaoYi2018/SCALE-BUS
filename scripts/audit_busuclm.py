from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "BUS-UCLM_v3"
OUT_DIR = ROOT / "results" / "busuclm_audit"
EXPECTED_COLUMNS = ["Image", "Resolution", "Label", "Doppler", "Marks", "Combined"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yes(value: object) -> bool:
    return str(value).strip().lower() == "yes"


def main() -> int:
    info_files = list(RAW_ROOT.rglob("INFO.csv"))
    if len(info_files) != 1:
        raise RuntimeError(f"expected one INFO.csv, found {len(info_files)}")
    info_path = info_files[0]
    dataset_root = info_path.parent
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    frame = pd.read_csv(info_path, sep=";")
    failures: list[str] = []
    warnings: list[str] = []
    if list(frame.columns) != EXPECTED_COLUMNS:
        failures.append(f"unexpected columns: {list(frame.columns)}")

    image_paths = {path.name: path for path in image_dir.glob("*.png")}
    mask_paths = {path.name: path for path in mask_dir.glob("*.png")}
    metadata_names = set(frame["Image"].astype(str))
    if len(frame) != 683:
        failures.append(f"metadata rows={len(frame)}, expected 683")
    if len(image_paths) != 683:
        failures.append(f"image files={len(image_paths)}, expected 683")
    if len(mask_paths) != 683:
        failures.append(f"mask files={len(mask_paths)}, expected 683")
    if metadata_names != set(image_paths):
        failures.append("metadata/image filename sets differ")
    if metadata_names != set(mask_paths):
        failures.append("metadata/mask filename sets differ")

    records: list[dict[str, object]] = []
    patient_labels: dict[str, Counter[str]] = defaultdict(Counter)
    image_hashes: dict[str, list[str]] = defaultdict(list)
    lesion_mask_hashes: dict[str, list[str]] = defaultdict(list)
    shape_mismatches = 0
    resolution_mismatches = 0
    shape_mismatch_records: list[dict[str, object]] = []
    resolution_mismatch_records: list[dict[str, object]] = []
    empty_lesion_masks = 0
    nonempty_normal_masks = 0
    color_mismatches = 0

    for row in frame.itertuples(index=False):
        filename = str(row.Image)
        image_path = image_paths.get(filename)
        mask_path = mask_paths.get(filename)
        if image_path is None or mask_path is None:
            continue
        label = str(row.Label).strip().lower()
        patient_id = Path(filename).stem.split("_", 1)[0]
        patient_labels[patient_id][label] += 1

        with Image.open(image_path) as image_obj:
            image = np.asarray(image_obj)
            image_size = image_obj.size
            image_mode = image_obj.mode
        with Image.open(mask_path) as mask_obj:
            mask_rgb = np.asarray(mask_obj.convert("RGB"))
            mask_size = mask_obj.size
            mask_mode = mask_obj.mode

        if image_size != mask_size:
            shape_mismatches += 1
            shape_mismatch_records.append(
                {
                    "image": filename,
                    "patient_id": patient_id,
                    "label": label,
                    "image_size": list(image_size),
                    "mask_size": list(mask_size),
                }
            )
        expected_resolution = tuple(
            int(value) for value in str(row.Resolution).lower().split("x")
        )
        if image_size != expected_resolution:
            resolution_mismatches += 1
            resolution_mismatch_records.append(
                {
                    "image": filename,
                    "patient_id": patient_id,
                    "label": label,
                    "metadata_resolution": list(expected_resolution),
                    "image_size": list(image_size),
                }
            )

        foreground = np.any(mask_rgb > 0, axis=2)
        foreground_fraction = float(foreground.mean())
        red_pixels = int(
            (
                (mask_rgb[..., 0] > 200)
                & (mask_rgb[..., 1] < 50)
                & (mask_rgb[..., 2] < 50)
            ).sum()
        )
        green_pixels = int(
            (
                (mask_rgb[..., 1] > 200)
                & (mask_rgb[..., 0] < 50)
                & (mask_rgb[..., 2] < 50)
            ).sum()
        )
        color_consistent = True
        if label == "benign":
            color_consistent = green_pixels > 0 and red_pixels == 0
        elif label == "malignant":
            color_consistent = red_pixels > 0 and green_pixels == 0
        elif label == "normal":
            color_consistent = not foreground.any()
        else:
            failures.append(f"unknown label {row.Label!r} for {filename}")
            color_consistent = False

        if label in {"benign", "malignant"} and not foreground.any():
            empty_lesion_masks += 1
        if label == "normal" and foreground.any():
            nonempty_normal_masks += 1
        if not color_consistent:
            color_mismatches += 1

        image_digest = sha256(image_path)
        mask_digest = sha256(mask_path)
        image_hashes[image_digest].append(filename)
        if label in {"benign", "malignant"}:
            lesion_mask_hashes[mask_digest].append(filename)

        geometry_consistent = (
            image_size == mask_size and image_size == expected_resolution
        )
        is_clean = (
            label in {"benign", "malignant"}
            and not yes(row.Doppler)
            and not yes(row.Marks)
            and not yes(row.Combined)
            and foreground.any()
            and color_consistent
            and geometry_consistent
        )
        records.append(
            {
                "image": filename,
                "patient_id": patient_id,
                "label": label,
                "doppler": yes(row.Doppler),
                "marks": yes(row.Marks),
                "combined": yes(row.Combined),
                "is_lesion": label in {"benign", "malignant"},
                "is_clean_primary": bool(is_clean),
                "image_width": int(image_size[0]),
                "image_height": int(image_size[1]),
                "image_mode": image_mode,
                "mask_mode": mask_mode,
                "foreground_fraction": foreground_fraction,
                "red_pixels": red_pixels,
                "green_pixels": green_pixels,
                "color_consistent": bool(color_consistent),
                "geometry_consistent": bool(geometry_consistent),
                "image_sha256": image_digest,
                "mask_sha256": mask_digest,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    manifest = pd.DataFrame(records)
    duplicate_images = {
        digest: names for digest, names in image_hashes.items() if len(names) > 1
    }
    duplicate_lesion_masks = {
        digest: names
        for digest, names in lesion_mask_hashes.items()
        if len(names) > 1
    }
    if duplicate_images:
        warnings.append(f"exact duplicate image groups={len(duplicate_images)}")
    if duplicate_lesion_masks:
        warnings.append(
            f"exact duplicate lesion-mask groups={len(duplicate_lesion_masks)}"
        )
    mismatch_patients = sorted(
        {
            str(row["patient_id"])
            for row in shape_mismatch_records + resolution_mismatch_records
        }
    )
    if shape_mismatches or resolution_mismatches:
        warnings.append(
            "geometry mismatch rows excluded: "
            f"shape={shape_mismatches}, resolution={resolution_mismatches}, "
            f"patients={mismatch_patients}"
        )
    if empty_lesion_masks:
        failures.append(f"empty lesion masks={empty_lesion_masks}")
    if nonempty_normal_masks:
        failures.append(f"non-empty normal masks={nonempty_normal_masks}")
    if color_mismatches:
        failures.append(f"mask color/label mismatches={color_mismatches}")

    patient_rows: list[dict[str, object]] = []
    for patient_id, counts in sorted(patient_labels.items()):
        patient_rows.append(
            {
                "patient_id": patient_id,
                "images": int(sum(counts.values())),
                "normal": int(counts["normal"]),
                "benign": int(counts["benign"]),
                "malignant": int(counts["malignant"]),
                "has_benign": bool(counts["benign"]),
                "has_malignant": bool(counts["malignant"]),
                "mixed_lesion_labels": bool(counts["benign"] and counts["malignant"]),
            }
        )
    patient_frame = pd.DataFrame(patient_rows)
    if len(patient_frame) != 38:
        failures.append(f"patient IDs={len(patient_frame)}, expected 38")

    lesion = manifest[manifest["is_lesion"]].copy()
    lesion_valid = lesion[lesion["geometry_consistent"]].copy()
    clean = manifest[manifest["is_clean_primary"]].copy()
    report = {
        "source_zip_sha256": (
            "7CB15270F1A2C920B4CD89122ACC5DD399EAE0FE545893A44596676225AC313E"
        ),
        "info_path": str(info_path),
        "rows": int(len(manifest)),
        "patients": int(manifest["patient_id"].nunique()),
        "class_counts": {
            str(key): int(value)
            for key, value in manifest["label"].value_counts().to_dict().items()
        },
        "lesion_rows": int(len(lesion)),
        "lesion_patients": int(lesion["patient_id"].nunique()),
        "lesion_class_counts": {
            str(key): int(value)
            for key, value in lesion["label"].value_counts().to_dict().items()
        },
        "valid_lesion_rows": int(len(lesion_valid)),
        "valid_lesion_patients": int(lesion_valid["patient_id"].nunique()),
        "valid_lesion_class_counts": {
            str(key): int(value)
            for key, value in lesion_valid["label"].value_counts().to_dict().items()
        },
        "clean_primary_rows": int(len(clean)),
        "clean_primary_patients": int(clean["patient_id"].nunique()),
        "clean_primary_class_counts": {
            str(key): int(value)
            for key, value in clean["label"].value_counts().to_dict().items()
        },
        "metadata_flags": {
            "doppler_yes": int(manifest["doppler"].sum()),
            "marks_yes": int(manifest["marks"].sum()),
            "combined_yes": int(manifest["combined"].sum()),
        },
        "patient_label_structure": {
            "benign_patients": int(patient_frame["has_benign"].sum()),
            "malignant_patients": int(patient_frame["has_malignant"].sum()),
            "mixed_lesion_label_patients": int(
                patient_frame["mixed_lesion_labels"].sum()
            ),
        },
        "shape_mismatches": shape_mismatches,
        "shape_mismatch_records": shape_mismatch_records,
        "resolution_mismatches": resolution_mismatches,
        "resolution_mismatch_records": resolution_mismatch_records,
        "geometry_excluded_patients": mismatch_patients,
        "empty_lesion_masks": empty_lesion_masks,
        "nonempty_normal_masks": nonempty_normal_masks,
        "color_mismatches": color_mismatches,
        "duplicate_image_groups": duplicate_images,
        "duplicate_lesion_mask_groups": duplicate_lesion_masks,
        "failures": failures,
        "warnings": warnings,
        "pass": not failures,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_DIR / "manifest_all.csv", index=False)
    lesion.to_csv(OUT_DIR / "manifest_lesions.csv", index=False)
    lesion_valid.to_csv(OUT_DIR / "manifest_lesions_valid.csv", index=False)
    clean.to_csv(OUT_DIR / "manifest_clean_primary.csv", index=False)
    patient_frame.to_csv(OUT_DIR / "patient_summary.csv", index=False)
    (OUT_DIR / "dataset_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
