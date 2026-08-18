from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "BUSI_zenodo_21128640" / "BUSI"
DEFAULT_OUT_DIR = ROOT / "results" / "busi_zenodo_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_groups(mapping: dict[str, list[str]]) -> list[list[str]]:
    return [sorted(names) for names in mapping.values() if len(names) > 1]


def perceptual_hash(gray: Image.Image) -> np.ndarray:
    small = np.asarray(gray.resize((32, 32)), dtype=np.float64)
    coefficients = dctn(small, norm="ortho")[:8, :8].ravel()[1:]
    return coefficients > np.median(coefficients)


def connected_components(
    names: list[str],
    edges: list[tuple[str, str, int, float, float]],
) -> list[list[str]]:
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(first: str, second: str) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first, second, *_ in edges:
        union(first, second)
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[find(name)].append(name)
    return sorted((sorted(group) for group in groups.values()), key=lambda x: x[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    image_dir = dataset / "images"
    mask_dir = dataset / "labels"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {path.name: path for path in image_dir.glob("*.png")}
    mask_paths = {path.name: path for path in mask_dir.glob("*.png")}
    paired_names = sorted(image_paths.keys() & mask_paths.keys())
    missing_masks = sorted(image_paths.keys() - mask_paths.keys())
    missing_images = sorted(mask_paths.keys() - image_paths.keys())

    rows: list[dict[str, object]] = []
    image_hashes: dict[str, list[str]] = defaultdict(list)
    mask_hashes: dict[str, list[str]] = defaultdict(list)
    mask_value_patterns: Counter[tuple[int, ...]] = Counter()
    failures: list[dict[str, str]] = []
    perceptual_hashes: dict[str, np.ndarray] = {}
    resized_gray: dict[str, np.ndarray] = {}

    for position, name in enumerate(paired_names, 1):
        image_path = image_paths[name]
        mask_path = mask_paths[name]
        try:
            label = (
                "benign"
                if name.startswith("benign_")
                else "malignant"
                if name.startswith("malignant_")
                else "unknown"
            )
            with Image.open(image_path) as handle:
                image_mode = handle.mode
                image = np.asarray(handle)
                gray = handle.convert("L")
                perceptual_hashes[name] = perceptual_hash(gray)
                resized_gray[name] = np.asarray(
                    gray.resize((256, 256)),
                    dtype=np.uint8,
                )
            with Image.open(mask_path) as handle:
                mask_mode = handle.mode
                mask = np.asarray(handle)
            image_shape = image.shape[:2]
            mask_shape = mask.shape[:2]
            values = tuple(int(value) for value in np.unique(mask))
            mask_value_patterns[values] += 1
            foreground = mask > 0
            image_sha = sha256(image_path)
            mask_sha = sha256(mask_path)
            image_hashes[image_sha].append(name)
            mask_hashes[mask_sha].append(name)
            rows.append(
                {
                    "image": name,
                    "label": label,
                    "image_width": int(image_shape[1]),
                    "image_height": int(image_shape[0]),
                    "mask_width": int(mask_shape[1]),
                    "mask_height": int(mask_shape[0]),
                    "image_mode": image_mode,
                    "mask_mode": mask_mode,
                    "geometry_consistent": image_shape == mask_shape,
                    "mask_values": "|".join(map(str, values)),
                    "foreground_pixels": int(foreground.sum()),
                    "foreground_fraction": float(foreground.mean()),
                    "image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )
            print(f"[{position:03d}/{len(paired_names):03d}] {name}", flush=True)
        except Exception as exc:
            failures.append({"image": name, "error": repr(exc)})
            print(f"FAILED {name}: {exc!r}", flush=True)

    near_duplicate_edges: list[tuple[str, str, int, float, float]] = []
    names_with_hash = sorted(perceptual_hashes)
    for first_idx, first in enumerate(names_with_hash):
        for second in names_with_hash[first_idx + 1 :]:
            hamming = int(
                np.count_nonzero(
                    perceptual_hashes[first] != perceptual_hashes[second]
                )
            )
            if hamming > 5:
                continue
            first_image = resized_gray[first].astype(np.float64) / 255.0
            second_image = resized_gray[second].astype(np.float64) / 255.0
            similarity = float(
                structural_similarity(
                    first_image,
                    second_image,
                    data_range=1.0,
                )
            )
            if similarity < 0.75:
                continue
            correlation = float(
                np.corrcoef(first_image.ravel(), second_image.ravel())[0, 1]
            )
            near_duplicate_edges.append(
                (first, second, hamming, similarity, correlation)
            )

    components = connected_components(names_with_hash, near_duplicate_edges)
    nontrivial_components = [group for group in components if len(group) > 1]
    label_by_name = {str(row["image"]): str(row["label"]) for row in rows}
    cross_label_groups = [
        group
        for group in nontrivial_components
        if len({label_by_name[name] for name in group}) > 1
    ]
    conflicted_names = {name for group in cross_label_groups for name in group}
    component_by_name: dict[str, str] = {}
    for index, group in enumerate(components, 1):
        group_id = f"visual_group_{index:04d}"
        for name in group:
            component_by_name[name] = group_id

    manifest = pd.DataFrame(rows)
    manifest["cv_group_id"] = manifest["image"].map(component_by_name)
    manifest["near_duplicate_group_size"] = manifest.groupby("cv_group_id")[
        "image"
    ].transform("size")
    manifest["cross_label_conflict"] = manifest["image"].isin(conflicted_names)
    manifest["is_external_valid"] = ~manifest["cross_label_conflict"]
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    manifest.loc[manifest["is_external_valid"]].to_csv(
        out_dir / "manifest_valid.csv",
        index=False,
    )
    pd.DataFrame(
        near_duplicate_edges,
        columns=(
            "first_image",
            "second_image",
            "phash_hamming",
            "ssim_256",
            "correlation_256",
        ),
    ).to_csv(out_dir / "near_duplicate_edges.csv", index=False)
    (out_dir / "near_duplicate_groups.json").write_text(
        json.dumps(
            {
                "thresholds": {
                    "phash_hamming_max": 5,
                    "ssim_256_min": 0.75,
                },
                "groups": nontrivial_components,
                "cross_label_groups": cross_label_groups,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    geometry_mismatch = (
        manifest.loc[~manifest["geometry_consistent"], "image"].tolist()
        if len(manifest)
        else []
    )
    empty_masks = (
        manifest.loc[manifest["foreground_pixels"] == 0, "image"].tolist()
        if len(manifest)
        else []
    )
    unknown_labels = (
        manifest.loc[manifest["label"] == "unknown", "image"].tolist()
        if len(manifest)
        else []
    )
    image_duplicates = duplicate_groups(image_hashes)
    mask_duplicates = duplicate_groups(mask_hashes)
    passed = not any(
        (
            missing_masks,
            missing_images,
            failures,
            geometry_mismatch,
            empty_masks,
            unknown_labels,
        )
    )
    summary = {
        "dataset": str(dataset),
        "images": len(image_paths),
        "masks": len(mask_paths),
        "paired": len(paired_names),
        "rows_audited": int(len(manifest)),
        "labels": {
            key: int(value)
            for key, value in manifest["label"].value_counts().items()
        },
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "failures": failures,
        "geometry_mismatch": geometry_mismatch,
        "empty_masks": empty_masks,
        "unknown_labels": unknown_labels,
        "mask_value_patterns": {
            "|".join(map(str, key)): int(value)
            for key, value in mask_value_patterns.items()
        },
        "exact_duplicate_image_groups": image_duplicates,
        "exact_duplicate_mask_groups": mask_duplicates,
        "near_duplicate_thresholds": {
            "phash_hamming_max": 5,
            "ssim_256_min": 0.75,
        },
        "near_duplicate_edges": len(near_duplicate_edges),
        "near_duplicate_groups": nontrivial_components,
        "cross_label_near_duplicate_groups": cross_label_groups,
        "cross_label_conflict_images": sorted(conflicted_names),
        "valid_after_conflict_exclusion": int(
            manifest["is_external_valid"].sum()
        ),
        "valid_labels": {
            key: int(value)
            for key, value in manifest.loc[
                manifest["is_external_valid"],
                "label",
            ].value_counts().items()
        },
        "cv_groups_after_exclusion": int(
            manifest.loc[
                manifest["is_external_valid"],
                "cv_group_id",
            ].nunique()
        ),
        "foreground_fraction": {
            "min": float(manifest["foreground_fraction"].min()),
            "median": float(manifest["foreground_fraction"].median()),
            "max": float(manifest["foreground_fraction"].max()),
        },
        "warnings": [
            "cross-label exact/near-duplicate groups are excluded from modeling",
            "same-label near-duplicate groups are kept in one CV group",
            "BUSI does not provide patient identifiers",
        ],
        "pass": passed,
    }
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
