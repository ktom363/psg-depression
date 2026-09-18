#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision Transformer training on PSG epoch images.

Consumes the output of ``edf_to_images.py`` directly:

    <image-root>/<patient_id>_image/<patient_id>_T900.png
    <image-root>/<patient_id>_image/<patient_id>_T930.png
    ...

Usage
-----
    python train_vit.py --image-root /path/to/images \
                        --labels labels.csv \
                        --output-dir runs/vit

``labels.csv`` must contain the columns ``patient_id`` and ``label``, where the
label is 1 for clinically significant depressive symptoms and 0 otherwise. 

Outputs
-------
    <output-dir>/best_vit.pth              best-validation-loss checkpoint
    <output-dir>/training_history.csv      per-epoch loss and accuracy
    <output-dir>/split.json                participant identifiers per subset
    <output-dir>/test_predictions.csv      participant-level test probabilities
    <output-dir>/run_config.json           resolved settings for reproducibility

Requirements
------------
    torch >= 2.4, torchvision, timm >= 0.9, numpy, scikit-learn, pillow, tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import timm

LOGGER = logging.getLogger("train_vit")

#: Two-class softmax output, as reported in Table S5.
NUM_CLASSES = 2
#: Index of the positive class (clinically significant depressive symptoms).
POSITIVE_CLASS = 1
#: A participant is classified as positive when the mean probability reaches
#: this value, i.e. the decision rule is ``>= 0.5``.
DECISION_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Fix the random state of every library used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give every dataloader worker a deterministic, distinct random state."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_state_dict_file(path: Path, map_location) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        return torch.load(path, map_location=map_location)


# --------------------------------------------------------------------------- #
# Data discovery and splitting                                                 #
# --------------------------------------------------------------------------- #

_EPOCH_PATTERN = re.compile(r"_T(\d+)$")
_DIGIT_RUN = re.compile(r"(\d+)")


def epoch_sort_key(path: Path):
    match = _EPOCH_PATTERN.search(path.stem)
    if match is not None:
        return (0, int(match.group(1)), path.stem)
    natural = [
        int(part) if part.isdigit() else part.lower()
        for part in _DIGIT_RUN.split(path.stem)
    ]
    return (1, natural, path.stem)


def load_labels(path: Path) -> Dict[str, int]:
    """Read participant labels from a CSV with columns patient_id,label."""
    labels: Dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("labels file must contain columns: patient_id,label")
        for line_number, row in enumerate(reader, start=2):
            patient_id = (row.get("patient_id") or "").strip()
            raw_label = (row.get("label") or "").strip()
            if not patient_id:
                raise ValueError(f"empty patient_id at line {line_number}")
            if patient_id in labels:
                raise ValueError(f"duplicate patient_id: {patient_id}")
            if raw_label not in {"0", "1"}:
                raise ValueError(
                    f"label for {patient_id} must be 0 or 1, found {raw_label!r}"
                )
            labels[patient_id] = int(raw_label)
    if not labels:
        raise ValueError("labels file contains no rows")
    return labels


def discover_participants(
    image_root: Path,
    labels: Dict[str, int],
    min_images: int,
) -> Dict[str, Dict[str, object]]:
    """Collect the epoch images of every labelled participant found on disk."""
    participants: Dict[str, Dict[str, object]] = {}
    for patient_id, label in labels.items():
        folder = image_root / f"{patient_id}_image"
        if not folder.is_dir():
            LOGGER.warning("[%s] image folder not found, skipped", patient_id)
            continue
        images = sorted(folder.glob("*.png"), key=epoch_sort_key)
        if len(images) < min_images:
            LOGGER.warning("[%s] %d image(s) < --min-images %d, skipped",
                           patient_id, len(images), min_images)
            continue
        participants[patient_id] = {"images": images, "label": label}
    if not participants:
        raise ValueError(f"no usable participant folder under {image_root}")
    return participants


def split_participants(
    participants: Dict[str, Dict[str, object]],
    test_size: float,
    val_size: float,
    random_state: int,
) -> Dict[str, List[str]]:
    ordered = (
        sorted(p for p, v in participants.items() if v["label"] == 1)
        + sorted(p for p, v in participants.items() if v["label"] == 0)
    )
    labels = [int(participants[p]["label"]) for p in ordered]

    outer = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(outer.split(ordered, labels))
    train_val = [ordered[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]

    inner = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size / (1.0 - test_size),
        random_state=random_state,
    )
    train_idx, val_idx = next(inner.split(train_val, train_val_labels))
    return {
        "train": [train_val[i] for i in train_idx],
        "val": [train_val[i] for i in val_idx],
        "test": [ordered[i] for i in test_idx],
    }


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

def build_items(
    participants: Dict[str, Dict[str, object]],
    patient_ids: Sequence[str],
    max_images: Optional[int] = None,
) -> List[Tuple[Path, int, str]]:
    """Flatten a subset of participants into one entry per epoch image."""
    items: List[Tuple[Path, int, str]] = []
    for patient_id in sorted(patient_ids):
        info = participants[patient_id]
        paths: Sequence[Path] = info["images"]
        if max_images is not None:
            paths = paths[:max_images]
        label = int(info["label"])
        items.extend((path, label, patient_id) for path in paths)
    if not items:
        raise ValueError("subset contains no images")
    return items


class PSGImageDataset(Dataset):

    def __init__(
        self,
        items: Sequence[Tuple[Path, int, str]],
        transform: transforms.Compose,
    ) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        path, label, patient_id = self.items[index]
        with Image.open(path) as handle:
            image = self.transform(handle.convert("RGB"))
        return image, label, patient_id


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #

class ViTClassifier(nn.Module):

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        checkpoint_path: Optional[Path] = None,
        pretrained: bool = True,
        train_head_only: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained and checkpoint_path is None,
            num_classes=NUM_CLASSES,
            drop_rate=dropout,
        )
        if checkpoint_path is not None:
            state = load_state_dict_file(checkpoint_path, map_location="cpu")
            state = state.get("state_dict", state)
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            LOGGER.info("loaded %s (missing=%d, unexpected=%d)",
                        checkpoint_path, len(missing), len(unexpected))

        if train_head_only:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.get_classifier().parameters():
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def resolve_normalization(model: nn.Module, mode: str) -> Dict[str, object]:

    if mode == "half":
        return {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "source": "fixed 0.5/0.5",
        }
    try:
        config = timm.data.resolve_model_data_config(model.backbone)
    except AttributeError:  # timm < 0.9
        config = timm.data.resolve_data_config({}, model=model.backbone)
    return {
        "mean": [float(v) for v in config["mean"]],
        "std": [float(v) for v in config["std"]],
        "source": "timm pretrained data config",
    }


# --------------------------------------------------------------------------- #
# Forward pass, evaluation, and aggregation                                    #
# --------------------------------------------------------------------------- #

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    description: str = "",
    collect: bool = False,
) -> Dict[str, object]:

    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    ids: List[str] = []
    y_true: List[int] = []
    y_prob: List[float] = []

    for images, labels, patient_ids in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits.float(), labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(logits.float().detach(), dim=1)[:, POSITIVE_CLASS]
        predictions = (probabilities >= DECISION_THRESHOLD).long()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((predictions == labels).sum().item())
        count += batch_size
        if collect:
            ids.extend(patient_ids)
            y_true.extend(int(v) for v in labels.detach().cpu().tolist())
            y_prob.extend(float(v) for v in probabilities.cpu().tolist())

    return {
        "loss": total_loss / max(count, 1),
        "accuracy": 100.0 * correct / max(count, 1),
        "images": count,
        "ids": ids,
        "y_true": np.asarray(y_true, dtype=int),
        "y_prob": np.asarray(y_prob, dtype=float),
    }


def aggregate_participants(
    ids: Sequence[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:

    order: List[str] = []
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    truth: Dict[str, int] = {}
    for patient_id, label, probability in zip(ids, y_true, y_prob):
        if patient_id not in totals:
            order.append(patient_id)
            totals[patient_id] = 0.0
            counts[patient_id] = 0
            truth[patient_id] = int(label)
        elif truth[patient_id] != int(label):
            raise ValueError(f"inconsistent labels for participant {patient_id}")
        totals[patient_id] += float(probability)
        counts[patient_id] += 1
    probabilities = np.asarray([totals[p] / counts[p] for p in order], dtype=float)
    labels = np.asarray([truth[p] for p in order], dtype=int)
    n_images = np.asarray([counts[p] for p in order], dtype=int)
    return order, labels, probabilities, n_images


def participant_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    if labels.size == 0:
        return 0.0
    return 100.0 * float((predictions == labels).mean())


def safe_auroc(labels: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probabilities))


# --------------------------------------------------------------------------- #
# Artefacts                                                                    #
# --------------------------------------------------------------------------- #

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_participant_acc",
    "val_participant_auroc",
    "lr",
]


def write_history(path: Path, history: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def write_predictions(
    path: Path,
    ids: Sequence[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_images: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["patient_id", "label", "probability", "prediction", "n_images"]
        )
        for patient_id, true, probability, images in zip(
            ids, y_true, y_prob, n_images
        ):
            writer.writerow([
                patient_id,
                int(true),
                f"{probability:.6f}",
                int(probability >= DECISION_THRESHOLD),
                int(images),
            ])


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Vision Transformer on PSG epoch images produced "
                    "by edf_to_images.py, with participant-level evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument("--image-root", required=True, type=Path,
                      help="directory containing <patient_id>_image folders")
    data.add_argument("--labels", required=True, type=Path,
                      help="CSV with columns patient_id,label")
    data.add_argument("--output-dir", required=True, type=Path,
                      help="directory for checkpoints, history, and predictions")
    data.add_argument("--min-images", type=positive_int, default=1,
                      help="skip participants with fewer epoch images than this")
    data.add_argument("--max-images", type=positive_int, default=None,
                      help="use only the first n images per participant, in "
                           "recording order")
    data.add_argument("--test-size", type=float, default=0.2)
    data.add_argument("--val-size", type=float, default=0.2)
    data.add_argument("--split-seed", type=int, default=1112,
                      help="random state of the stratified participant split; "
                           "independent of --seed, so that runs with different "
                           "training seeds share one partition")

    model = parser.add_argument_group("model")
    model.add_argument("--model-name", default="vit_base_patch16_224",
                       help="timm model identifier")
    model.add_argument("--checkpoint", type=Path, default=None,
                       help="local pretrained state dict; omit to let timm "
                            "download the pretrained weights")
    model.add_argument("--no-pretrained", dest="pretrained",
                       action="store_false",
                       help="initialise the backbone randomly")
    model.add_argument("--head-only", dest="train_head_only",
                       action="store_true",
                       help="freeze the pretrained encoder and update the "
                            "classification head only; by default the whole "
                            "backbone is fine-tuned")
    model.add_argument("--dropout", type=float, default=0.2,
                       help="dropout applied to the pooled representation "
                            "before the classifier")
    model.add_argument("--normalization", choices=("model", "half"),
                       default="model",
                       help="channel statistics for input normalisation; "
                            "'model' uses the statistics of the pretrained "
                            "weights of the resolved backbone, 'half' forces "
                            "0.5/0.5")

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--epochs", type=positive_int, default=100)
    optimisation.add_argument("--lr", type=float, default=1e-4)
    optimisation.add_argument("--weight-decay", type=float, default=1e-4)
    optimisation.add_argument("--batch-size", type=positive_int, default=32,
                              help="images per optimiser step")
    optimisation.add_argument("--eval-batch-size", type=positive_int, default=64,
                              help="images per forward pass at validation and "
                                   "test time")
    optimisation.add_argument("--seed", type=int, default=42)
    optimisation.add_argument("--no-amp", dest="amp", action="store_false",
                              help="disable mixed-precision training")
    optimisation.add_argument("--num-workers", type=int, default=0)
    optimisation.add_argument("--device", default=None,
                              help="cuda, cpu, or omit for automatic selection")
    optimisation.add_argument("--log-level", default="INFO",
                              choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = args.amp and device.type == "cuda"
    LOGGER.info("device: %s | mixed precision: %s", device, use_amp)

    labels = load_labels(args.labels)
    participants = discover_participants(args.image_root, labels, args.min_images)
    split = split_participants(
        participants, args.test_size, args.val_size, args.split_seed
    )
    for name in ("train", "val", "test"):
        ids = split[name]
        positives = sum(int(participants[p]["label"]) for p in ids)
        LOGGER.info("%-5s: %3d participants (%d positive)", name, len(ids), positives)
    (args.output_dir / "split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )

    model = ViTClassifier(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        pretrained=args.pretrained,
        train_head_only=args.train_head_only,
        dropout=args.dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("parameters: %s trainable of %s total", f"{trainable:,}", f"{total:,}")

    # Input normalisation follows the pretrained weights of the resolved
    # backbone unless --normalization half is given.
    normalization = resolve_normalization(model, args.normalization)
    LOGGER.info("normalisation (%s): mean=%s std=%s",
                normalization["source"], normalization["mean"], normalization["std"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalization["mean"], normalization["std"]),
    ])

    def make_loader(subset: str, train: bool) -> DataLoader:
        dataset = PSGImageDataset(
            build_items(participants, split[subset], args.max_images),
            transform=transform,
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        return DataLoader(
            dataset,
            batch_size=args.batch_size if train else args.eval_batch_size,
            shuffle=train,
            generator=generator if train else None,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    train_loader = make_loader("train", train=True)
    val_loader = make_loader("val", train=False)
    test_loader = make_loader("test", train=False)
    LOGGER.info("images: %d train | %d val | %d test",
                len(train_loader.dataset), len(val_loader.dataset),
                len(test_loader.dataset))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()}
    config.update({
        "device": str(device),
        "loss": "CrossEntropyLoss",
        "num_classes": NUM_CLASSES,
        "decision_threshold": DECISION_THRESHOLD,
        "training_unit": "image",
        "evaluation_unit": "participant (mean of per-image probabilities)",
        "images_per_optimiser_step": args.batch_size,
        "lr_scheduler": "CosineAnnealingLR",
        "normalization": normalization,
        "mixed_precision": use_amp,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "n_images": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
    })
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    checkpoint_path = args.output_dir / "best_vit.pth"
    best_val_loss = float("inf")
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model, train_loader, criterion, device, use_amp,
            optimizer=optimizer, scaler=scaler,
            description=f"epoch {epoch}/{args.epochs} [train]",
        )
        val_stats = run_epoch(
            model, val_loader, criterion, device, use_amp,
            description=f"epoch {epoch}/{args.epochs} [val]",
            collect=True,
        )
        _, val_labels, val_probs, _ = aggregate_participants(
            val_stats["ids"], val_stats["y_true"], val_stats["y_prob"]
        )
        val_participant_acc = participant_accuracy(val_labels, val_probs)
        val_participant_auroc = safe_auroc(val_labels, val_probs)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "train_acc": round(train_stats["accuracy"], 4),
            "val_loss": round(val_stats["loss"], 6),
            "val_acc": round(val_stats["accuracy"], 4),
            "val_participant_acc": round(val_participant_acc, 4),
            "val_participant_auroc": (
                "" if val_participant_auroc is None
                else round(val_participant_auroc, 4)
            ),
            "lr": current_lr,
        })
        write_history(args.output_dir / "training_history.csv", history)
        LOGGER.info(
            "epoch %3d | train %.4f / %.2f%% | val %.4f / %.2f%% | "
            "val participant %.2f%%",
            epoch, train_stats["loss"], train_stats["accuracy"],
            val_stats["loss"], val_stats["accuracy"], val_participant_acc,
        )

        # Model selection uses the image-level validation loss, which is the
        # quantity the network is optimised on; participant-level validation
        # metrics are logged for monitoring only.
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), checkpoint_path)
            LOGGER.info("epoch %3d | checkpoint saved (val loss %.4f)",
                        epoch, best_val_loss)

    if not checkpoint_path.exists():
        LOGGER.error("no checkpoint was written")
        return 1

    LOGGER.info("evaluating the best checkpoint on the internal test set")
    model.load_state_dict(
        load_state_dict_file(checkpoint_path, map_location=device)
    )
    test_stats = run_epoch(
        model, test_loader, criterion, device, use_amp,
        description="test", collect=True,
    )
    ids, y_true, y_prob, n_images = aggregate_participants(
        test_stats["ids"], test_stats["y_true"], test_stats["y_prob"]
    )
    write_predictions(
        args.output_dir / "test_predictions.csv", ids, y_true, y_prob, n_images
    )

    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("confusion matrix (rows: true 0/1, columns: predicted 0/1)")
    print(confusion_matrix(y_true, y_pred))
    auroc = safe_auroc(y_true, y_prob)
    if auroc is None:
        LOGGER.warning("test set contains a single class; AUROC not defined")
    else:
        LOGGER.info("test AUROC %.4f | AUPRC %.4f",
                    auroc, average_precision_score(y_true, y_prob))

    LOGGER.info("artefacts written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
