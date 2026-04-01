#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多类超声语义分割训练脚本
默认方案：AdamW + CosineAnnealingLR + Linear Warmup + 编码器/解码器分组学习率

支持模型：
- Unet
- UnetPlusPlus

支持编码器：
- resnet50
- mobilenet_v2
- timm-mobilenetv3_small_100
- timm-mobilenetv3_large_100

支持调度方案：
- adamw_cosine          (默认，推荐)
- adamw_onecycle
- adamw_plateau
- radam_plateau
- sgd_step
- sgd_cawr
- sgd_poly

数据组织示例：
dataset/
  train/
    images/
      0001.png
      0002.png
    masks/
      0001.png
      0002.png
  val/
    images/
    masks/

mask 要求：
- 单通道图，像素值为类别 id，范围 [0, num_classes-1]
"""

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


# =========================
# Utils
# =========================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_image_files(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(name)
    return sorted(files)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# =========================
# Dataset
# =========================

class SegDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        image_size: int = 512,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size

        image_files = list_image_files(image_dir)
        mask_files = set(list_image_files(mask_dir))

        self.samples = []
        for fname in image_files:
            if fname in mask_files:
                self.samples.append(fname)

        if len(self.samples) == 0:
            raise RuntimeError(f"No matched image-mask pairs found in:\n{image_dir}\n{mask_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname = self.samples[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # 如果 mask 是 3 通道，转单通道
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()
        else:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


def build_transforms(image_size: int = 512):
    train_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# =========================
# Metrics
# =========================

def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    pred: [N,H,W]
    target: [N,H,W]
    """
    with torch.no_grad():
        mask = (target >= 0) & (target < num_classes)
        label = num_classes * target[mask].to(torch.int64) + pred[mask]
        count = torch.bincount(label, minlength=num_classes ** 2)
        conf = count.reshape(num_classes, num_classes)
    return conf


def dice_from_confmat(conf: torch.Tensor, eps: float = 1e-7) -> Tuple[float, List[float]]:
    """
    返回 mean dice 和每类 dice
    """
    tp = torch.diag(conf).float()
    fp = conf.sum(dim=0).float() - tp
    fn = conf.sum(dim=1).float() - tp
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice.mean().item(), dice.cpu().tolist()


def iou_from_confmat(conf: torch.Tensor, eps: float = 1e-7) -> Tuple[float, List[float]]:
    tp = torch.diag(conf).float()
    fp = conf.sum(dim=0).float() - tp
    fn = conf.sum(dim=1).float() - tp
    iou = (tp + eps) / (tp + fp + fn + eps)
    return iou.mean().item(), iou.cpu().tolist()


# =========================
# Loss
# =========================

class DiceLossMultiClass(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [N,C,H,W]
        target: [N,H,W]
        """
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        cardinality = torch.sum(probs + target_oh, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class CombinedSegLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLossMultiClass(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


# =========================
# Warmup
# =========================

@dataclass
class WarmupConfig:
    warmup_steps: int
    warmup_type: str = "linear"    # linear | cosine | exponential
    start_factor: float = 0.05
    exp_k: float = 5.0


class WarmupFn:
    def __init__(self, cfg: WarmupConfig):
        self.warmup_steps = int(cfg.warmup_steps)
        self.warmup_type = str(cfg.warmup_type)
        self.start_factor = float(cfg.start_factor)
        self.exp_k = float(cfg.exp_k)

    def __call__(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        if step >= self.warmup_steps:
            return 1.0

        t = step / float(self.warmup_steps)

        if self.warmup_type == "linear":
            return self.start_factor + (1.0 - self.start_factor) * t

        if self.warmup_type == "cosine":
            return self.start_factor + (1.0 - self.start_factor) * (0.5 - 0.5 * math.cos(math.pi * t))

        if self.warmup_type == "exponential":
            num = math.exp(self.exp_k * t) - 1.0
            den = math.exp(self.exp_k) - 1.0
            return self.start_factor + (1.0 - self.start_factor) * (num / max(den, 1e-12))

        raise ValueError(f"Unknown warmup_type: {self.warmup_type}")


class WarmupThenScheduler:
    """
    warmup 按 iteration 调度
    warmup 结束后切换到主调度器
    """

    def __init__(self, warmup_scheduler: LambdaLR, main_scheduler, warmup_steps: int):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_steps = int(warmup_steps)
        self.global_step = 0

    def step(self, metric: Optional[float] = None):
        if self.global_step < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            if isinstance(self.main_scheduler, ReduceLROnPlateau):
                if metric is not None:
                    self.main_scheduler.step(metric)
            else:
                self.main_scheduler.step()
        self.global_step += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "warmup_steps": self.warmup_steps,
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            "main_scheduler": self.main_scheduler.state_dict() if hasattr(self.main_scheduler, "state_dict") else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.global_step = int(state["global_step"])
        self.warmup_steps = int(state["warmup_steps"])
        self.warmup_scheduler.load_state_dict(state["warmup_scheduler"])
        if state.get("main_scheduler") is not None:
            self.main_scheduler.load_state_dict(state["main_scheduler"])

    def get_last_lr(self) -> List[float]:
        if self.global_step <= self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        if hasattr(self.main_scheduler, "get_last_lr"):
            return self.main_scheduler.get_last_lr()
        return [g["lr"] for g in self.warmup_scheduler.optimizer.param_groups]


class PolyLRScheduler:
    """
    poly: lr = base_lr * (1 - iter / max_iter)^power
    """

    def __init__(self, optimizer: Optimizer, max_iters: int, power: float = 0.9, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.max_iters = max(1, int(max_iters))
        self.power = power
        self.min_lr = min_lr
        self.last_iter = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.last_iter += 1
        factor = (1.0 - min(self.last_iter, self.max_iters) / self.max_iters) ** self.power
        for i, group in enumerate(self.optimizer.param_groups):
            new_lr = max(self.base_lrs[i] * factor, self.min_lr)
            group["lr"] = new_lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_iters": self.max_iters,
            "power": self.power,
            "min_lr": self.min_lr,
            "last_iter": self.last_iter,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.max_iters = state["max_iters"]
        self.power = state["power"]
        self.min_lr = state["min_lr"]
        self.last_iter = state["last_iter"]
        self.base_lrs = state["base_lrs"]

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


# =========================
# Model / Optimizer groups
# =========================

def build_model(
    arch: str,
    encoder_name: str,
    num_classes: int,
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    arch = arch.lower()
    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    elif arch == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def split_encoder_decoder_named_params(model: nn.Module):
    """
    尽量兼容 segmentation_models_pytorch 的命名：
    encoder.* -> 编码器
    其他 -> 解码器/seg head
    """
    encoder_decay, encoder_no_decay = [], []
    decoder_decay, decoder_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_encoder = name.startswith("encoder.")
        is_no_decay = (p.ndim == 1) or name.endswith(".bias")

        if is_encoder:
            if is_no_decay:
                encoder_no_decay.append(p)
            else:
                encoder_decay.append(p)
        else:
            if is_no_decay:
                decoder_no_decay.append(p)
            else:
                decoder_decay.append(p)

    return encoder_decay, encoder_no_decay, decoder_decay, decoder_no_decay


def make_param_groups(model: nn.Module, lr_encoder: float, lr_decoder: float, weight_decay: float):
    enc_d, enc_nd, dec_d, dec_nd = split_encoder_decoder_named_params(model)
    groups = []
    if enc_d:
        groups.append({"params": enc_d, "lr": lr_encoder, "weight_decay": weight_decay})
    if enc_nd:
        groups.append({"params": enc_nd, "lr": lr_encoder, "weight_decay": 0.0})
    if dec_d:
        groups.append({"params": dec_d, "lr": lr_decoder, "weight_decay": weight_decay})
    if dec_nd:
        groups.append({"params": dec_nd, "lr": lr_decoder, "weight_decay": 0.0})
    return groups


# =========================
# Schedulers / Schemes
# =========================

def build_optimizer_and_scheduler(model: nn.Module, args, steps_per_epoch: int):
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    lr_encoder = args.lr * args.encoder_lr_ratio
    lr_decoder = args.lr
    param_groups = make_param_groups(model, lr_encoder, lr_decoder, args.weight_decay)

    scheme = args.scheme.lower()

    if scheme == "adamw_cosine":
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        main = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=args.min_lr,
        )
        warm_cfg = WarmupConfig(warmup_steps=warmup_steps,
                                warmup_type=args.warmup_type,
                                start_factor=args.warmup_start_factor,
                                exp_k=args.warmup_exp_k)
        warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warm_cfg))
        scheduler = WarmupThenScheduler(warm, main, warmup_steps)
        scheduler_mode = "iter"
        return optimizer, scheduler, scheduler_mode

    if scheme == "adamw_onecycle":
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        max_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler = OneCycleLR(optimizer,
                               max_lr=max_lrs,
                               epochs=args.epochs,
                               steps_per_epoch=steps_per_epoch,
                               pct_start=max(0.01, min(0.5, warmup_steps / max(1, total_steps))),
                               anneal_strategy="cos",
                               div_factor=args.onecycle_div_factor,
                               final_div_factor=args.onecycle_final_div_factor)
        scheduler_mode = "iter"
        return optimizer, scheduler, scheduler_mode

    if scheme == "adamw_plateau":
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        warm_cfg = WarmupConfig(
            warmup_steps=warmup_steps,
            warmup_type=args.warmup_type,
            start_factor=args.warmup_start_factor,
            exp_k=args.warmup_exp_k,
        )
        warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warm_cfg))
        main = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=1e-4,
            min_lr=args.min_lr,
        )
        scheduler = WarmupThenScheduler(warm, main, warmup_steps)
        scheduler_mode = "plateau"
        return optimizer, scheduler, scheduler_mode

    if scheme == "radam_plateau":
        optimizer = torch.optim.RAdam(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        warm_cfg = WarmupConfig(
            warmup_steps=warmup_steps,
            warmup_type=args.warmup_type,
            start_factor=args.warmup_start_factor,
            exp_k=args.warmup_exp_k,
        )
        warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warm_cfg))
        main = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=1e-4,
            min_lr=args.min_lr,
        )
        scheduler = WarmupThenScheduler(warm, main, warmup_steps)
        scheduler_mode = "plateau"
        return optimizer, scheduler, scheduler_mode

    if scheme == "sgd_step":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
        main = StepLR(
            optimizer,
            step_size=args.step_drop_epochs * steps_per_epoch,
            gamma=args.step_gamma,
        )
        warm_cfg = WarmupConfig(
            warmup_steps=warmup_steps,
            warmup_type=args.warmup_type,
            start_factor=args.warmup_start_factor,
            exp_k=args.warmup_exp_k,
        )
        warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warm_cfg))
        scheduler = WarmupThenScheduler(warm, main, warmup_steps)
        scheduler_mode = "iter"
        return optimizer, scheduler, scheduler_mode

    if scheme == "sgd_cawr":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
        main = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cawr_t0_epochs * steps_per_epoch,
            T_mult=args.cawr_tmult,
            eta_min=args.min_lr,
        )
        warm_cfg = WarmupConfig(
            warmup_steps=warmup_steps,
            warmup_type=args.warmup_type,
            start_factor=args.warmup_start_factor,
            exp_k=args.warmup_exp_k,
        )
        warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warm_cfg))
        scheduler = WarmupThenScheduler(warm, main, warmup_steps)
        scheduler_mode = "iter"
        return optimizer, scheduler, scheduler_mode

    if scheme == "sgd_poly":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
        main = PolyLRScheduler(
            optimizer=optimizer,
            max_iters=max(1, total_steps - warmup_steps),
            power=args.poly_power,
            min_lr=args.min_lr,
        )
        warm_cfg = WarmupConfig(
            warmup_steps=warmup_steps,
            warmup_type=args.warmup_type,
            start_factor=args.warmup_start_factor,
            exp_k=args.warmup_exp_k,
        )
        warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warm_cfg))
        scheduler = WarmupThenScheduler(warm, main, warmup_steps)
        scheduler_mode = "iter"
        return optimizer, scheduler, scheduler_mode

    raise ValueError(f"Unsupported scheme: {args.scheme}")


# =========================
# Train / Validate
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler,
    scheduler_mode: str,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_classes: int,
    max_grad_norm: Optional[float] = 1.0,
):
    model.train()

    running_loss = 0.0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler_mode == "iter":
            scheduler.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        confmat += compute_confusion_matrix(preds, masks, num_classes)

    epoch_loss = running_loss / len(loader.dataset)
    mean_dice, per_class_dice = dice_from_confmat(confmat)
    mean_iou, per_class_iou = iou_from_confmat(confmat)

    return {
        "loss": epoch_loss,
        "dice": mean_dice,
        "iou": mean_iou,
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):
    model.eval()

    running_loss = 0.0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        confmat += compute_confusion_matrix(preds, masks, num_classes)

    epoch_loss = running_loss / len(loader.dataset)
    mean_dice, per_class_dice = dice_from_confmat(confmat)
    mean_iou, per_class_iou = iou_from_confmat(confmat)

    return {
        "loss": epoch_loss,
        "dice": mean_dice,
        "iou": mean_iou,
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
    }


# =========================
# Checkpoint
# =========================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    map_location: str = "cpu",
):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_score = ckpt.get("best_score", -1.0)
    return start_epoch, best_score


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train-image-dir", type=str, required=True)
    parser.add_argument("--train-mask-dir", type=str, required=True)
    parser.add_argument("--val-image-dir", type=str, required=True)
    parser.add_argument("--val-mask-dir", type=str, required=True)

    # model
    parser.add_argument("--arch", type=str, default="Unet", choices=["Unet", "UnetPlusPlus"])
    parser.add_argument("--encoder-name", type=str, default="resnet50")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=512)

    # training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # optimization
    parser.add_argument("--scheme", type=str, default="adamw_cosine", choices=[
        "adamw_cosine",
        "adamw_onecycle",
        "adamw_plateau",
        "radam_plateau",
        "sgd_step",
        "sgd_cawr",
        "sgd_poly",
    ])
    parser.add_argument("--lr", type=float, default=3e-4, help="decoder/head base lr")
    parser.add_argument("--encoder-lr-ratio", type=float, default=0.1, help="encoder lr = lr * ratio")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # warmup
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--warmup-type", type=str, default="linear", choices=["linear", "cosine", "exponential"])
    parser.add_argument("--warmup-start-factor", type=float, default=0.05)
    parser.add_argument("--warmup-exp-k", type=float, default=5.0)

    # step scheduler
    parser.add_argument("--step-drop-epochs", type=int, default=50)
    parser.add_argument("--step-gamma", type=float, default=0.1)

    # cawr
    parser.add_argument("--cawr-t0-epochs", type=int, default=20)
    parser.add_argument("--cawr-tmult", type=int, default=2)

    # poly
    parser.add_argument("--poly-power", type=float, default=0.9)

    # plateau
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=10)

    # onecycle
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=1e4)

    # loss
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)

    # regularization / misc
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-dir", type=str, default="./runs/default")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-best-only", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.save_dir)
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_tf, val_tf = build_transforms(args.image_size)

    train_set = SegDataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        transform=train_tf,
        image_size=args.image_size,
    )
    val_set = SegDataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        transform=val_tf,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    model = build_model(
        arch=args.arch,
        encoder_name=args.encoder_name,
        num_classes=args.num_classes,
        encoder_weights=args.encoder_weights,
    ).to(device)

    criterion = CombinedSegLoss(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
    )

    optimizer, scheduler, scheduler_mode = build_optimizer_and_scheduler(model=model,
                                                                         args=args,
                                                                         steps_per_epoch=len(train_loader),
                                                                         )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    best_score = -1.0

    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_score = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        print(f"[Resume] start_epoch={start_epoch}, best_score={best_score:.6f}")

    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("========== Training Config ==========")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("=====================================")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_mode=scheduler_mode,
            criterion=criterion,
            scaler=scaler,
            device=device,
            num_classes=args.num_classes,
            max_grad_norm=args.max_grad_norm,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
        )

        if scheduler_mode == "plateau":
            scheduler.step(val_metrics["dice"])

        lr_list = [f"{x:.3e}" for x in (
            scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr")
            else [g["lr"] for g in optimizer.param_groups]
        )]

        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1:03d}/{args.epochs:03d}] "
            f"time={elapsed:.1f}s | "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} | "
            f"lr={lr_list}"
        )

        # 保存 latest
        latest_path = os.path.join(args.save_dir, "latest.pt")
        save_checkpoint(path=latest_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_score=best_score,
                        args=args,
                        )

        # 保存 best
        score = val_metrics["dice"]
        if score > best_score:
            best_score = score
            best_path = os.path.join(args.save_dir, "best.pt")
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                args=args,
            )
            print(f"[Best] saved to {best_path}, val_dice={best_score:.6f}")

        if not args.save_best_only:
            epoch_path = os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pt")
            save_checkpoint(
                path=epoch_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                args=args,
            )

    print(f"Training finished. Best val_dice = {best_score:.6f}")


if __name__ == "__main__":
    main()
