#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate a multiclass segmentation model built with segmentation_models.pytorch.

Features
--------
1. Inference + post-processing: softmax -> argmax
2. Metrics with SMP (mode='multiclass'):
   - per-class precision / recall / IoU / Dice / F1 / F2
   - micro-average precision / recall / IoU / Dice / F1 / F2
3. HD95:
   - per-class HD95 over validation set
   - "micro" HD95 defined here as the mean HD95 over all valid (image, class) pairs
     Note: HD95 does not have a canonical pixel-level micro reduction like confusion-matrix metrics.
4. PR curves:
   - per-class one-vs-rest PR curve
   - micro-average PR curve over all classes
5. Save plots + CSV + JSON

Usage example
-------------
python validate_smp_multiclass.py \
    --checkpoint path/to/model.pt \
    --num-classes 9 \
    --batch-size 4 \
    --output-dir ./val_results

You need to adapt:
- build_model()
- create_val_dataloader()

Requirements
------------
pip install segmentation-models-pytorch torch torchvision scikit-learn scipy matplotlib pandas numpy
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import precision_recall_curve, average_precision_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# =========================
# 1. Dataset placeholder
# =========================
class DummySegDataset(Dataset):
    """
    Replace this dataset with your own validation dataset.
    It should return:
        image: FloatTensor [C, H, W]
        mask : LongTensor  [H, W], values in [0, num_classes-1] or ignore_index
    """

    def __init__(self, length: int = 8, num_classes: int = 9, image_size: Tuple[int, int] = (256, 256)):
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h, w = self.image_size
        image = torch.randn(1, h, w).float()  # e.g. ultrasound single-channel
        mask = torch.randint(0, self.num_classes, (h, w), dtype=torch.long)
        return image, mask


def create_val_dataloader(batch_size: int, num_classes: int, num_workers: int) -> DataLoader:
    """
    Replace this function with your real validation dataloader.
    """
    dataset = DummySegDataset(length=16, num_classes=num_classes, image_size=(256, 256))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# =========================
# 2. Model placeholder
# =========================
def build_model(num_classes: int, in_channels: int = 1) -> torch.nn.Module:
    """
    Replace with your actual model definition.
    Example uses SMP Unet.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Optional: strip "module." if saved by DDP/DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    return model


# =========================
# 3. HD95 utilities
# =========================
def _mask_to_surface(mask: np.ndarray) -> np.ndarray:
    """
    Get 1-pixel surface/boundary from a binary mask.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    surface = mask ^ eroded
    return surface


def hd95_binary(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: Optional[Tuple[float, float]] = None) -> float:
    """
    Compute symmetric 95th percentile Hausdorff Distance for a binary mask pair.

    Returns
    -------
    float
        np.nan if both pred and gt are empty
        np.inf if only one of them is empty
        otherwise HD95
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if not pred_mask.any() and not gt_mask.any():
        return np.nan
    if pred_mask.any() != gt_mask.any():
        return np.inf

    pred_surface = _mask_to_surface(pred_mask)
    gt_surface = _mask_to_surface(gt_mask)

    if not pred_surface.any() and not gt_surface.any():
        return 0.0
    if pred_surface.any() != gt_surface.any():
        return np.inf

    # Distance transform of complement of surface
    # distance_transform_edt gives distance to nearest zero
    # so we pass ~surface to get distance to surface.
    if spacing is None:
        spacing = (1.0, 1.0)

    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    distances_pred_to_gt = dt_gt[pred_surface]
    distances_gt_to_pred = dt_pred[gt_surface]

    all_surface_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred], axis=0)
    if all_surface_distances.size == 0:
        return 0.0

    return float(np.percentile(all_surface_distances, 95))


# =========================
# 4. PR curve helpers
# =========================
def one_hot_labels_from_index(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    mask: [N, H, W] long
    returns: [N, C, H, W] float
    """
    oh = F.one_hot(mask.long(), num_classes=num_classes)  # [N, H, W, C]
    oh = oh.permute(0, 3, 1, 2).float()
    return oh


def flatten_valid_probs_and_targets(
    probs: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    probs  : [N, C, H, W] float
    target : [N, H, W] long

    Returns
    -------
    probs_flat  : [M, C]
    target_flat : [M]
    """
    probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)  # [N*H*W, C]
    target = target.reshape(-1)

    if ignore_index is not None:
        valid = target != ignore_index
        probs = probs[valid]
        target = target[valid]

    return probs.detach().cpu().numpy(), target.detach().cpu().numpy()


# =========================
# 5. Validation
# =========================
@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    output_dir: str,
    ignore_index: Optional[int] = None,
    class_names: Optional[List[str]] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    assert len(class_names) == num_classes

    model.eval()

    # Accumulate tp/fp/fn/tn batch-wise
    tp_all, fp_all, fn_all, tn_all = [], [], [], []

    # For PR curves
    all_probs_flat = []
    all_targets_flat = []

    # For per-class HD95
    hd95_values_per_class: Dict[int, List[float]] = {c: [] for c in range(num_classes)}

    for batch_idx, batch in enumerate(val_loader):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()  # [N, H, W]

        logits = model(images)  # [N, C, H, W]

        # ---------- post-processing ----------
        probs = torch.softmax(logits, dim=1)         # [N, C, H, W]
        pred = torch.argmax(probs, dim=1).long()     # [N, H, W]

        # ---------- smp multiclass metrics ----------
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred,
            target,
            mode="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        tp_all.append(tp.cpu())
        fp_all.append(fp.cpu())
        fn_all.append(fn.cpu())
        tn_all.append(tn.cpu())

        # ---------- accumulate for PR curves ----------
        probs_flat, target_flat = flatten_valid_probs_and_targets(
            probs, target, num_classes=num_classes, ignore_index=ignore_index
        )
        all_probs_flat.append(probs_flat)
        all_targets_flat.append(target_flat)

        # ---------- per-image per-class HD95 ----------
        pred_np = pred.detach().cpu().numpy()      # [N, H, W]
        target_np = target.detach().cpu().numpy()  # [N, H, W]

        for i in range(pred_np.shape[0]):
            for c in range(num_classes):
                if ignore_index is not None:
                    valid = target_np[i] != ignore_index
                    pred_bin = (pred_np[i] == c) & valid
                    gt_bin = (target_np[i] == c) & valid
                else:
                    pred_bin = pred_np[i] == c
                    gt_bin = target_np[i] == c

                hd = hd95_binary(pred_bin, gt_bin)
                if not np.isnan(hd):
                    hd95_values_per_class[c].append(hd)

    # Concatenate stats across batches: [num_images, C]
    tp_all = torch.cat(tp_all, dim=0)
    fp_all = torch.cat(fp_all, dim=0)
    fn_all = torch.cat(fn_all, dim=0)
    tn_all = torch.cat(tn_all, dim=0)

    # =========================
    # 6. Per-class metrics
    # =========================
    # SMP docs: with reduction="none" metrics keep image x class granularity;
    # we sum across images manually to get per-class dataset-level metrics.
    tp_sum_c = tp_all.sum(dim=0, keepdim=True)  # [1, C]
    fp_sum_c = fp_all.sum(dim=0, keepdim=True)
    fn_sum_c = fn_all.sum(dim=0, keepdim=True)
    tn_sum_c = tn_all.sum(dim=0, keepdim=True)

    per_class_precision = smp.metrics.precision(tp_sum_c, fp_sum_c, fn_sum_c, tn_sum_c, reduction=None).squeeze(0)
    per_class_recall = smp.metrics.recall(tp_sum_c, fp_sum_c, fn_sum_c, tn_sum_c, reduction=None).squeeze(0)
    per_class_iou = smp.metrics.iou_score(tp_sum_c, fp_sum_c, fn_sum_c, tn_sum_c, reduction=None).squeeze(0)
    per_class_dice = smp.metrics.f1_score(tp_sum_c, fp_sum_c, fn_sum_c, tn_sum_c, reduction=None).squeeze(0)
    per_class_f1 = smp.metrics.f1_score(tp_sum_c, fp_sum_c, fn_sum_c, tn_sum_c, reduction=None).squeeze(0)
    per_class_f2 = smp.metrics.fbeta_score(tp_sum_c, fp_sum_c, fn_sum_c, tn_sum_c, beta=2.0, reduction=None).squeeze(0)

    # HD95 per class
    per_class_hd95 = []
    for c in range(num_classes):
        vals = hd95_values_per_class[c]
        finite_vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 0:
            per_class_hd95.append(np.nan)
        elif len(finite_vals) == 0:
            per_class_hd95.append(np.inf)
        else:
            per_class_hd95.append(float(np.mean(finite_vals)))

    # =========================
    # 7. Micro-average metrics
    # =========================
    micro_precision = smp.metrics.precision(tp_all, fp_all, fn_all, tn_all, reduction="micro").item()
    micro_recall = smp.metrics.recall(tp_all, fp_all, fn_all, tn_all, reduction="micro").item()
    micro_iou = smp.metrics.iou_score(tp_all, fp_all, fn_all, tn_all, reduction="micro").item()
    micro_dice = smp.metrics.f1_score(tp_all, fp_all, fn_all, tn_all, reduction="micro").item()
    micro_f1 = smp.metrics.f1_score(tp_all, fp_all, fn_all, tn_all, reduction="micro").item()
    micro_f2 = smp.metrics.fbeta_score(tp_all, fp_all, fn_all, tn_all, beta=2.0, reduction="micro").item()

    # HD95 "micro" here is defined as mean over all valid (image, class) pairs
    all_hd95_vals = []
    for c in range(num_classes):
        all_hd95_vals.extend([v for v in hd95_values_per_class[c] if np.isfinite(v)])
    micro_hd95 = float(np.mean(all_hd95_vals)) if len(all_hd95_vals) > 0 else np.nan

    # =========================
    # 8. Save metrics table
    # =========================
    per_class_df = pd.DataFrame({
        "class_id": list(range(num_classes)),
        "class_name": class_names,
        "precision": per_class_precision.cpu().numpy(),
        "recall": per_class_recall.cpu().numpy(),
        "iou": per_class_iou.cpu().numpy(),
        "dice": per_class_dice.cpu().numpy(),
        "hd95": np.array(per_class_hd95, dtype=np.float64),
        "f1score": per_class_f1.cpu().numpy(),
        "f2score": per_class_f2.cpu().numpy(),
    })
    per_class_csv = os.path.join(output_dir, "per_class_metrics.csv")
    per_class_df.to_csv(per_class_csv, index=False)

    summary = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_iou": micro_iou,
        "micro_dice": micro_dice,
        "micro_hd95": micro_hd95,
        "micro_f1score": micro_f1,
        "micro_f2score": micro_f2,
    }
    summary_json = os.path.join(output_dir, "micro_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # =========================
    # 9. PR curves
    # =========================
    all_probs_flat = np.concatenate(all_probs_flat, axis=0)   # [M, C]
    all_targets_flat = np.concatenate(all_targets_flat, axis=0)  # [M]

    # one-hot GT for PR
    gt_onehot = np.eye(num_classes, dtype=np.uint8)[all_targets_flat]  # [M, C]

    pr_curve_dir = os.path.join(output_dir, "pr_curves")
    os.makedirs(pr_curve_dir, exist_ok=True)

    pr_ap_rows = []

    # Per-class PR curves
    for c in range(num_classes):
        y_true = gt_onehot[:, c]
        y_score = all_probs_flat[:, c]

        # If a class is absent in the validation set, sklearn PR/AP is undefined
        if np.unique(y_true).size < 2:
            print(f"[WARN] class {c} ({class_names[c]}) absent or constant in GT, skip PR curve.")
            continue

        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        plt.figure(figsize=(6, 5))
        plt.plot(recall_arr, precision_arr, lw=2, label=f"AP={ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - Class {c}: {class_names[c]}")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(pr_curve_dir, f"pr_curve_class_{c}.png"), dpi=200)
        plt.close()

        curve_df = pd.DataFrame({"recall": recall_arr, "precision": precision_arr})
        curve_df.to_csv(os.path.join(pr_curve_dir, f"pr_curve_class_{c}.csv"), index=False)

        pr_ap_rows.append({
            "class_id": c,
            "class_name": class_names[c],
            "average_precision": ap,
        })

    # Micro-average PR curve
    y_true_micro = gt_onehot.reshape(-1)
    y_score_micro = all_probs_flat.reshape(-1)

    precision_micro, recall_micro, _ = precision_recall_curve(y_true_micro, y_score_micro)
    ap_micro = average_precision_score(y_true_micro, y_score_micro, average="micro")

    plt.figure(figsize=(6, 5))
    plt.plot(recall_micro, precision_micro, lw=2, label=f"micro-AP={ap_micro:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-average PR Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(pr_curve_dir, "pr_curve_micro.png"), dpi=200)
    plt.close()

    pd.DataFrame({"recall": recall_micro, "precision": precision_micro}).to_csv(
        os.path.join(pr_curve_dir, "pr_curve_micro.csv"), index=False
    )

    pd.DataFrame(pr_ap_rows).to_csv(os.path.join(output_dir, "pr_average_precision.csv"), index=False)

    # =========================
    # 10. Console print
    # =========================
    print("\n========== Per-class metrics ==========")
    print(per_class_df.to_string(index=False))

    print("\n========== Micro-average metrics ==========")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}" if isinstance(v, (float, int)) and not math.isinf(v) else f"{k}: {v}")

    print(f"\nSaved per-class metrics to: {per_class_csv}")
    print(f"Saved micro summary to:     {summary_json}")
    print(f"Saved PR curves to:         {pr_curve_dir}")

    return per_class_df, summary


# =========================
# 6. Main
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of segmentation classes")
    parser.add_argument("--in-channels", type=int, default=1, help="Input channels, e.g. ultrasound usually 1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ignore-index", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./val_results")
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional class names, e.g. --class-names bg muscle1 muscle2 ...",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.class_names is not None and len(args.class_names) != args.num_classes:
        raise ValueError("Length of --class-names must equal --num-classes")

    model = build_model(num_classes=args.num_classes, in_channels=args.in_channels)
    model = load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)

    val_loader = create_val_dataloader(
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
    )

    validate(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        ignore_index=args.ignore_index,
        class_names=args.class_names,
    )


if __name__ == "__main__":
    main()
