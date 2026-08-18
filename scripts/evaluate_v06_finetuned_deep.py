from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "tools" / "torch_cache"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


TABLES = {
    "BUS-UCLM": ROOT / "results" / "v04_advanced_features" / "features_busuclm_advanced.csv",
    "BUSI": ROOT / "results" / "v04_advanced_features" / "features_busi_advanced.csv",
    "BUS-BRA": ROOT / "results" / "v04_external_features" / "features_busbra_advanced.csv",
    "BrEaST": ROOT / "results" / "v04_external_features" / "features_breast_advanced.csv",
}
OUT = ROOT / "results" / "v06_finetuned_deep"
SEEDS = (20260718, 20260719, 20260720)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    if "accepted" in frame.columns and frame["accepted"].notna().any():
        frame = frame[frame["accepted"].astype(str).str.lower() == "true"]
    if "is_clean_primary" in frame.columns and frame["is_clean_primary"].notna().any():
        frame = frame[frame["is_clean_primary"].astype(str).str.lower() == "true"]
    return frame.copy()


def load_manifest() -> pd.DataFrame:
    frames = []
    for dataset, path in TABLES.items():
        frame = eligible(pd.read_csv(path))
        frame = frame[["image", "patient_id", "label", "image_path", "mask_path"]].copy()
        frame["dataset"] = dataset
        frame["target"] = frame["label"].astype(str).str.lower().eq("malignant").astype(int)
        frame["group_id"] = dataset + "::" + frame["patient_id"].astype(str)
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output[["image_path", "mask_path"]].isna().any().any():
        raise RuntimeError("missing image or mask path")
    return output


def roi_crop(image_path: str, mask_path: str, expansion: float = 0.12) -> np.ndarray:
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.uint8)
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert("L")) > 0
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError(f"empty mask: {mask_path}")
    h = int(rows.max() - rows.min() + 1)
    w = int(columns.max() - columns.min() + 1)
    pad_y = max(2, int(round(expansion * h)))
    pad_x = max(2, int(round(expansion * w)))
    y0 = max(0, int(rows.min()) - pad_y)
    y1 = min(image.shape[0], int(rows.max()) + pad_y + 1)
    x0 = max(0, int(columns.min()) - pad_x)
    x1 = min(image.shape[1], int(columns.max()) + pad_x + 1)
    crop = Image.fromarray(image[y0:y1, x0:x1]).resize((224, 224), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.uint8)


def cache_images(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    cache: dict[int, np.ndarray] = {}
    started = time.time()
    for index, row in frame.iterrows():
        cache[int(index)] = roi_crop(row["image_path"], row["mask_path"])
        if len(cache) % 500 == 0:
            print(f"[cache] {len(cache)}/{len(frame)} images, {time.time() - started:.1f}s", flush=True)
    return cache


class UltrasoundDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        cache: dict[int, np.ndarray],
        transform: transforms.Compose,
    ) -> None:
        self.frame = frame.copy()
        self.indices = self.frame.index.to_numpy(dtype=int)
        self.targets = self.frame["target"].to_numpy(dtype=np.int64)
        self.cache = cache
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[position])
        image = self.cache[index]
        tensor = self.transform(image)
        target = torch.tensor(int(self.targets[position]), dtype=torch.long)
        return tensor, target, index


def make_transforms(training: bool) -> transforms.Compose:
    operations: list[object] = [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3)]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                    fill=0,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


def group_weights(frame: pd.DataFrame) -> np.ndarray:
    strata = frame["dataset"].astype(str) + "::" + frame["target"].astype(str)
    group_sizes = frame.groupby("group_id")["group_id"].transform("size").to_numpy(float)
    groups_per_stratum = (
        frame.assign(_stratum=strata)
        .groupby("_stratum")["group_id"]
        .transform("nunique")
        .to_numpy(float)
    )
    weights = 1.0 / np.maximum(group_sizes * groups_per_stratum, 1.0)
    return weights / weights.mean()


def evaluation_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("group_id")["group_id"].transform("size").to_numpy(float)
    return 1.0 / np.maximum(counts, 1.0)


def valid_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set((row.dataset, int(row.target)) for row in frame.itertuples())
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + offset)
        train_idx, val_idx = next(
            splitter.split(frame, frame["target"], groups=frame["group_id"])
        )
        train = frame.iloc[train_idx].copy()
        val = frame.iloc[val_idx].copy()
        train_cells = set((row.dataset, int(row.target)) for row in train.itertuples())
        val_cells = set((row.dataset, int(row.target)) for row in val.itertuples())
        if required.issubset(train_cells) and required.issubset(val_cells):
            return train, val
    raise RuntimeError("unable to form a validation split containing every source dataset/class cell")


def build_model(name: str) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(name)
    return model


@torch.inference_mode()
def predict(
    model: nn.Module,
    frame: pd.DataFrame,
    cache: dict[int, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = UltrasoundDataset(frame, cache, make_transforms(False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    targets = []
    indices = []
    model.eval()
    for images, labels, batch_indices in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        targets.append(labels.numpy())
        indices.append(batch_indices.numpy())
    return np.concatenate(probabilities), np.concatenate(targets), np.concatenate(indices)


def balanced_auc(frame: pd.DataFrame, targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        roc_auc_score(
            targets,
            probabilities,
            sample_weight=evaluation_weights(frame),
        )
    )


def train_one(
    architecture: str,
    source: pd.DataFrame,
    cache: dict[int, np.ndarray],
    seed: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, dict[str, object]]:
    set_seed(seed)
    train_frame, val_frame = valid_split(source, seed)
    train_dataset = UltrasoundDataset(train_frame, cache, make_transforms(True))
    train_sample_weights = group_weights(train_frame)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
        num_samples=len(train_sample_weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probability, val_target, val_indices = predict(
            model, val_frame, cache, device, batch_size
        )
        ordered_val = val_frame.loc[val_indices]
        val_auc = balanced_auc(ordered_val, val_target, val_probability)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_auc": val_auc,
            }
        )
        print(
            f"[train] {architecture} seed={seed} epoch={epoch} "
            f"loss={np.mean(losses):.4f} val_auc={val_auc:.4f}",
            flush=True,
        )
        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no model state selected")
    model.load_state_dict(best_state)
    metadata = {
        "architecture": architecture,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "training_seconds": time.time() - started,
        "train_images": len(train_frame),
        "validation_images": len(val_frame),
        "train_groups": int(train_frame["group_id"].nunique()),
        "validation_groups": int(val_frame["group_id"].nunique()),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }
    return model, metadata


def evaluate_run(
    architecture: str,
    seed: int,
    protocol: str,
    source_names: list[str],
    target_names: list[str],
    manifest: pd.DataFrame,
    cache: dict[int, np.ndarray],
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], dict[str, object]]:
    source = manifest[manifest["dataset"].isin(source_names)].copy()
    model, metadata = train_one(
        architecture,
        source,
        cache,
        seed,
        device,
        batch_size,
        max_epochs,
        patience,
    )
    metric_rows = []
    prediction_frames = []
    for target_name in target_names:
        target_frame = manifest[manifest["dataset"] == target_name].copy()
        started = time.time()
        probability, target, indices = predict(model, target_frame, cache, device, batch_size)
        ordered = target_frame.loc[indices].copy()
        elapsed = time.time() - started
        weights = evaluation_weights(ordered)
        auc = float(roc_auc_score(target, probability, sample_weight=weights))
        ap = float(average_precision_score(target, probability, sample_weight=weights))
        metric_rows.append(
            {
                "protocol": protocol,
                "architecture": architecture,
                "seed": seed,
                "sources": "+".join(source_names),
                "target_dataset": target_name,
                "roc_auc_group_balanced": auc,
                "average_precision_group_balanced": ap,
                "target_images": len(ordered),
                "target_groups": int(ordered["group_id"].nunique()),
                "inference_seconds": elapsed,
                "milliseconds_per_image": 1000.0 * elapsed / len(ordered),
                "best_epoch": metadata["best_epoch"],
                "validation_auc": metadata["best_validation_auc"],
                "parameters": metadata["parameters"],
            }
        )
        ordered["protocol"] = protocol
        ordered["architecture"] = architecture
        ordered["seed"] = seed
        ordered["sources"] = "+".join(source_names)
        ordered["probability"] = probability
        prediction_frames.append(ordered)
        print(
            f"[test] {protocol} {architecture} seed={seed} target={target_name} "
            f"AUC={auc:.4f} AP={ap:.4f}",
            flush=True,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_rows, prediction_frames, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("locked", "lodo", "all"), default="all")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("resnet18", "efficientnet_b0"),
        default=("resnet18", "efficientnet_b0"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_out = OUT / args.protocol
    run_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "protocol": args.protocol,
                "architectures": args.architectures,
                "seeds": args.seeds,
            },
            indent=2,
        ),
        flush=True,
    )
    manifest = load_manifest()
    manifest.to_csv(run_out / "curated_manifest.csv", index=False)
    cache = cache_images(manifest)
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    run_metadata: list[dict[str, object]] = []
    protocols = []
    if args.protocol in ("locked", "all"):
        protocols.append(
            (
                "combined-development-to-locked-external",
                ["BUS-UCLM", "BUSI"],
                ["BUS-BRA", "BrEaST"],
            )
        )
    if args.protocol in ("lodo", "all"):
        for target in TABLES:
            protocols.append(
                (
                    "lodo",
                    [dataset for dataset in TABLES if dataset != target],
                    [target],
                )
            )
    for protocol, sources, targets in protocols:
        for architecture in args.architectures:
            for seed in args.seeds:
                rows, predictions, metadata = evaluate_run(
                    architecture,
                    seed,
                    protocol,
                    sources,
                    targets,
                    manifest,
                    cache,
                    device,
                    args.batch_size,
                    args.max_epochs,
                    args.patience,
                )
                metadata.update(
                    {
                        "protocol": protocol,
                        "sources": sources,
                        "targets": targets,
                    }
                )
                all_metrics.extend(rows)
                all_predictions.extend(predictions)
                run_metadata.append(metadata)
                pd.DataFrame(all_metrics).to_csv(run_out / "metrics_partial.csv", index=False)
                pd.concat(all_predictions, ignore_index=True).to_csv(
                    run_out / "predictions_partial.csv",
                    index=False,
                )
                (run_out / "training_metadata_partial.json").write_text(
                    json.dumps(run_metadata, indent=2),
                    encoding="utf-8",
                )
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(run_out / "metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(run_out / "predictions.csv", index=False)
    (run_out / "training_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    summary = (
        metrics.groupby(["protocol", "architecture", "target_dataset"])
        .agg(
            roc_auc_mean=("roc_auc_group_balanced", "mean"),
            roc_auc_min=("roc_auc_group_balanced", "min"),
            roc_auc_max=("roc_auc_group_balanced", "max"),
            average_precision_mean=("average_precision_group_balanced", "mean"),
            seeds=("seed", "nunique"),
            parameters=("parameters", "first"),
            milliseconds_per_image=("milliseconds_per_image", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(run_out / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
