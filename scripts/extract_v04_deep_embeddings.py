from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.morphology import binary_dilation, binary_erosion, disk
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "tools" / "torch_cache"))
INPUT_DIR = ROOT / "results" / "v04_advanced_features"
OUTPUT_DIR = ROOT / "results" / "v04_deep_embeddings"
DATASETS = {
    "busuclm": INPUT_DIR / "features_busuclm_advanced.csv",
    "busi": INPUT_DIR / "features_busi_advanced.csv",
    "busbra": (
        ROOT
        / "results"
        / "v04_external_features"
        / "features_busbra_advanced.csv"
    ),
    "breast": (
        ROOT
        / "results"
        / "v04_external_features"
        / "features_breast_advanced.csv"
    ),
}
ENCODER_DIMENSIONS = {
    "resnet18": 512,
    "efficientnet_b0": 1280,
    "convnext_tiny": 768,
}
VIEWS = ("roi_context", "lesion_only", "inner_only", "lesion_outer")
SEED = 20260717


def build_encoder(
    name: str,
) -> tuple[nn.Module, object, int, str]:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Identity()
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Identity()
    else:
        raise ValueError(f"unsupported encoder: {name}")
    model.eval()
    return (
        model,
        weights.transforms(),
        ENCODER_DIMENSIONS[name],
        weights.__class__.__name__ + "." + weights.name,
    )


def _crop_to_support(
    image: np.ndarray,
    support: np.ndarray,
    margin: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.where(support)
    if rows.size == 0:
        raise ValueError("empty support")
    top = max(0, int(rows.min()) - margin)
    bottom = min(image.shape[0], int(rows.max()) + margin + 1)
    left = max(0, int(columns.min()) - margin)
    right = min(image.shape[1], int(columns.max()) + margin + 1)
    return image[top:bottom, left:right], support[top:bottom, left:right]


def build_view(
    image: np.ndarray,
    mask: np.ndarray,
    view: str,
) -> Image.Image:
    grayscale = np.asarray(image, dtype=np.uint8)
    lesion = np.asarray(mask) > 0
    if grayscale.ndim != 2 or lesion.ndim != 2:
        raise ValueError("image and mask must be two-dimensional")
    if grayscale.shape != lesion.shape:
        raise ValueError("image-mask shape mismatch")
    if lesion.sum() < 32:
        raise ValueError("lesion mask is too small")

    outer_support = binary_dilation(lesion, disk(12))
    crop_image, crop_outer = _crop_to_support(
        grayscale,
        outer_support,
    )
    crop_lesion, _ = _crop_to_support(lesion, outer_support)
    crop_lesion = np.asarray(crop_lesion, dtype=bool)
    crop_inner = binary_erosion(crop_lesion, disk(2))
    if crop_inner.sum() < 32:
        crop_inner = crop_lesion.copy()

    if view == "roi_context":
        selected = crop_image.copy()
    elif view == "lesion_only":
        selected = crop_image.copy()
        selected[~crop_lesion] = int(np.median(selected[crop_lesion]))
    elif view == "inner_only":
        selected = crop_image.copy()
        selected[~crop_inner] = int(np.median(selected[crop_inner]))
    elif view == "lesion_outer":
        selected = crop_image.copy()
        selected[~crop_outer] = int(np.median(selected[crop_outer]))
    else:
        raise ValueError(f"unsupported view: {view}")

    rgb = np.repeat(selected[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


class UltrasoundViewDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        view: str,
        transform: object,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.view = view
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.frame.iloc[index]
        with Image.open(record["image_path"]) as handle:
            image = np.asarray(handle.convert("L"))
        with Image.open(record["mask_path"]) as handle:
            mask = np.asarray(handle.convert("L"))
        view_image = build_view(image, mask, self.view)
        tensor = self.transform(view_image)
        return tensor, index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument(
        "--encoder",
        choices=tuple(ENCODER_DIMENSIONS),
        required=True,
    )
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected environment")
    device = torch.device("cuda")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_path = DATASETS[args.dataset]
    frame = pd.read_csv(source_path)
    model, transform, expected_dimension, weights_name = build_encoder(
        args.encoder
    )
    model = model.to(device)
    dataset = UltrasoundViewDataset(frame, args.view, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    started = time.perf_counter()
    batches: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for position, (images, batch_indices) in enumerate(loader, 1):
            output = model(images.to(device, non_blocking=True))
            output = output.reshape(output.shape[0], -1)
            batches.append(output.detach().cpu().numpy().astype(np.float32))
            indices.append(batch_indices.numpy())
            if position == 1 or position % 5 == 0 or position == len(loader):
                print(
                    f"[embedding] {args.dataset} {args.encoder} "
                    f"{args.view}: batch {position}/{len(loader)}",
                    flush=True,
                )

    embedding = np.concatenate(batches, axis=0)
    observed_indices = np.concatenate(indices)
    if not np.array_equal(observed_indices, np.arange(len(frame))):
        raise RuntimeError("embedding row order changed")
    if embedding.shape != (len(frame), expected_dimension):
        raise RuntimeError(
            f"embedding shape {embedding.shape}, expected "
            f"{(len(frame), expected_dimension)}"
        )
    if not np.isfinite(embedding).all():
        raise RuntimeError("embedding contains non-finite values")

    stem = f"{args.dataset}_{args.encoder}_{args.view}"
    npz_path = OUTPUT_DIR / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        embedding=embedding,
        image=np.asarray(frame["image"].astype(str).tolist(), dtype="U"),
    )
    metadata_columns = [
        column
        for column in (
            "image",
            "label",
            "patient_id",
            "cv_group_id",
            "image_path",
            "mask_path",
        )
        if column in frame
    ]
    frame[metadata_columns].to_csv(
        OUTPUT_DIR / f"{stem}_manifest.csv",
        index=False,
    )
    protocol_name = (
        "V04_LOCKED_EXTERNAL_PROTOCOL_V1.md"
        if args.dataset in {"busbra", "breast"}
        else "V04_DEVELOPMENT_PROTOCOL_V1.md"
    )
    summary = {
        "protocol": protocol_name.removesuffix(".md"),
        "dataset": args.dataset,
        "source": str(source_path.resolve()),
        "encoder": args.encoder,
        "weights": weights_name,
        "view": args.view,
        "rows": int(len(frame)),
        "dimension": int(embedding.shape[1]),
        "batch_size": args.batch_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "finite_values": int(embedding.size),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output": str(npz_path.resolve()),
    }
    (OUTPUT_DIR / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
