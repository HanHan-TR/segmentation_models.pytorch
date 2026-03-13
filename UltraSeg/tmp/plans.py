import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Iterable, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau


@dataclass
class WarmupConfig:
    warmup_steps: int
    warmup_type: str = "linear"   # "linear" | "cosine" | "exponential"
    start_factor: float = 0.05    # LR starts at start_factor * base_lr
    exp_k: float = 5.0            # only used for exponential warmup


class WarmupFn:
    """
    Callable object (NOT a lambda) so that LambdaLR can serialize its config in state_dict
    (PyTorch stores callable-object __dict__; lambdas/functions are not saved).
    """

    def __init__(self, cfg: WarmupConfig):
        self.warmup_steps = int(cfg.warmup_steps)
        self.warmup_type = str(cfg.warmup_type)
        self.start_factor = float(cfg.start_factor)
        self.exp_k = float(cfg.exp_k)

    def __call__(self, step: int) -> float:
        # step is LambdaLR.last_epoch, starting at 0.
        if self.warmup_steps <= 0:
            return 1.0
        if step >= self.warmup_steps:
            return 1.0
        t = step / float(self.warmup_steps)

        if self.warmup_type == "linear":
            return self.start_factor + (1.0 - self.start_factor) * t

        if self.warmup_type == "cosine":
            # smooth ramp using cosine from start_factor to 1
            return self.start_factor + (1.0 - self.start_factor) * (0.5 - 0.5 * math.cos(math.pi * t))

        if self.warmup_type == "exponential":
            # normalized exp curve to [0, 1], then map to [start_factor, 1]
            num = math.exp(self.exp_k * t) - 1.0
            den = math.exp(self.exp_k) - 1.0
            return self.start_factor + (1.0 - self.start_factor) * (num / max(den, 1e-12))

        raise ValueError(f"Unknown warmup_type: {self.warmup_type}")


class WarmupThen:
    """
    Lightweight wrapper: during warmup, step warmup_scheduler each iteration;
    after warmup, step main_scheduler.
    定义了一个轻量级包装器，在预热阶段使用 warmup_scheduler，预热后使用 main_scheduler。
    """

    def __init__(self,
                 warmup_scheduler: LambdaLR,  # 预热阶段使用的学习率调度器，类型为 LambdaLR
                 main_scheduler,  # 预热后使用的学习率调度器
                 warmup_steps: int):  # 预热步数
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_steps = int(warmup_steps)
        self.global_step = 0  # 初始化全局步数计数器为 0

    def step(self, metric: Optional[float] = None):
        """
        根据当前训练步数决定使用哪个方法来更新学习率。
        如果当前步数小于预热步数，调用预热学习率调度器；否则，调用主学习率调度器。
        如果主学习率调度器是 ReduceLROnPlateau 类型，且提供了指标，
        则使用该指标来更新学习率；否则，直接调用 step() 方法。
        """
        if self.global_step < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            if isinstance(self.main_scheduler, ReduceLROnPlateau):
                # Plateau must be stepped with a metric (typically per-epoch).
                if metric is not None:
                    self.main_scheduler.step(metric)
            else:
                self.main_scheduler.step()
        self.global_step += 1

    def get_last_lr(self) -> List[float]:
        """获取当前学习率，返回一个列表，包含所有参数组的学习率。

        - 如果仍在预热阶段，返回 warmup_scheduler 的学习率
        - 否则，如果主调度器有 get_last_lr 方法，使用该方法获取
        - 最后，通过从优化器参数组中直接获取学习率作为备选方案
        """
        # best-effort
        if self.global_step <= self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        if hasattr(self.main_scheduler, "get_last_lr"):
            return self.main_scheduler.get_last_lr()

        return [g["lr"] for g in self.warmup_scheduler.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """功能：保存调度器状态，用于模型 checkpoint
            - 返回值：包含状态信息的字典
            - 保存内容：
                - 当前全局步数
                - 预热步数
                - 预热调度器状态
                - 主调度器状态（如果有 state_dict 方法）
        """
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


def split_encoder_decoder_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Heuristic split: parameters under 'encoder.' go to encoder group, others to decoder/head group.
    Adapt this to your own model implementation (smp, monai, custom unet, nnunet, etc.).
    """
    enc, dec = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            enc.append(p)
        else:
            dec.append(p)
    return enc, dec


def make_optimizer_param_groups(model: nn.Module, lr_encoder: float, lr_decoder: float, weight_decay: float,
                                no_decay_bias_norm: bool = True) -> List[Dict[str, Any]]:
    """
    Optional: exclude bias / norm params from weight_decay (common heuristic). Keep if you want.
    """
    enc_params, dec_params = split_encoder_decoder_params(model)

    def group_params(params: List[nn.Parameter], lr: float):
        if not no_decay_bias_norm:
            return [{"params": params, "lr": lr, "weight_decay": weight_decay}]

        decay, no_decay = [], []
        for p in params:
            # We cannot reliably check "is norm" without names; this is minimal.
            if p.ndim == 1:  # often bias or norm scale
                no_decay.append(p)
            else:
                decay.append(p)

        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    return group_params(enc_params, lr_encoder) + group_params(dec_params, lr_decoder)


@torch.no_grad()
def validate(model: nn.Module, val_loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    # TODO: compute your metrics: e.g., mean Dice, val loss
    # return {"val_loss": ..., "val_dice": ...}
    return {"val_loss": 0.0, "val_dice": 0.0}


def train_one_epoch(model: nn.Module, train_loader, optimizer: Optimizer, lr_ctl,
                    device: torch.device, scaler: torch.amp.GradScaler,
                    criterion: nn.Module,
                    max_grad_norm: Optional[float] = 1.0) -> int:
    model.train()
    global_steps = 0

    for batch in train_loader:
        # --- unpack your batch ---
        # images, masks = batch
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()

        # Gradient clipping (optional but often stabilizes segmentation training)
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # IMPORTANT: scheduler step AFTER optimizer step
        lr_ctl.step()

        global_steps += 1

    return global_steps


def save_checkpoint(path: str, model: nn.Module, optimizer: Optimizer, lr_ctl,
                    scaler: torch.amp.GradScaler, epoch: int, extra: Optional[Dict[str, Any]] = None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_ctl": lr_ctl.state_dict() if lr_ctl is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optimizer, lr_ctl,
                    scaler: torch.amp.GradScaler, map_location="cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    # Safer pattern (per PyTorch optimizer docs warning): make sure lr scheduler is initialized before optimizer.load_state_dict
    # Then load scheduler state, then optimizer state.
    if lr_ctl is not None and ckpt.get("lr_ctl") is not None:
        lr_ctl.load_state_dict(ckpt["lr_ctl"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def build_scheme_A(model, train_loader_len: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 300
    steps_per_epoch = train_loader_len
    warmup_epochs = 10
    warmup_steps = warmup_epochs * steps_per_epoch

    lr_decoder = 0.02
    lr_encoder = 0.002  # 0.1x (heuristic; see DeepLab for head-LR>backbone-LR precedent)

    weight_decay = 1e-4

    param_groups = make_optimizer_param_groups(model, lr_encoder, lr_decoder, weight_decay, no_decay_bias_norm=True)

    optimizer = torch.optim.SGD(
        param_groups,
        momentum=0.9,
        nesterov=True,
    )

    # Main scheduler should be created BEFORE warmup so it captures base_lrs correctly.
    drop_every_epochs = 50
    step_size = drop_every_epochs * steps_per_epoch
    main = StepLR(optimizer, step_size=step_size, gamma=0.1)

    warmup_cfg = WarmupConfig(warmup_steps=warmup_steps, warmup_type="linear", start_factor=0.01)
    warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warmup_cfg))

    lr_ctl = WarmupThen(warm, main, warmup_steps=warmup_steps)
    return optimizer, lr_ctl, epochs


def build_scheme_B(model, train_loader_len: int):
    epochs = 300
    steps_per_epoch = train_loader_len
    warmup_steps = 10 * steps_per_epoch

    lr_decoder = 0.02
    lr_encoder = 0.002
    weight_decay = 1e-4

    param_groups = make_optimizer_param_groups(model, lr_encoder, lr_decoder, weight_decay, no_decay_bias_norm=True)
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)

    # SGDR-like restarts
    T0_epochs = 20
    main = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T0_epochs * steps_per_epoch,
        T_mult=2,
        eta_min=1e-5,
    )

    warmup_cfg = WarmupConfig(warmup_steps=warmup_steps, warmup_type="cosine", start_factor=0.01)
    warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warmup_cfg))

    lr_ctl = WarmupThen(warm, main, warmup_steps=warmup_steps)
    return optimizer, lr_ctl, epochs


def build_scheme_C(model, train_loader_len: int):
    epochs = 300
    steps_per_epoch = train_loader_len
    total_steps = epochs * steps_per_epoch

    warmup_steps = 10 * steps_per_epoch

    lr_decoder = 3e-4
    lr_encoder = 1e-4 * 0.3  # e.g., 0.3x (you can try 0.1x~0.3x)
    weight_decay = 1e-3

    param_groups = make_optimizer_param_groups(model, lr_encoder, lr_decoder, weight_decay, no_decay_bias_norm=True)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # main cosine decay after warmup
    main = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=1e-6,
    )

    warmup_cfg = WarmupConfig(warmup_steps=warmup_steps, warmup_type="linear", start_factor=0.05)
    warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warmup_cfg))

    lr_ctl = WarmupThen(warm, main, warmup_steps=warmup_steps)
    return optimizer, lr_ctl, epochs


def build_scheme_D(model, train_loader_len: int):
    # OneCycle often works with fewer epochs than cosine/step schedules (try 150~250)
    epochs = 200
    steps_per_epoch = train_loader_len
    total_steps = epochs * steps_per_epoch

    # Explicit warmup steps (converted to pct_start)
    warmup_epochs = 10
    warmup_steps = warmup_epochs * steps_per_epoch
    pct_start = min(0.5, max(0.01, warmup_steps / max(1, total_steps)))

    max_lr_decoder = 1e-3
    max_lr_encoder = 3e-4
    weight_decay = 1e-3

    # optimizer initial lr can be small; OneCycleLR will manage the schedule.
    param_groups = make_optimizer_param_groups(model, lr_encoder=max_lr_encoder, lr_decoder=max_lr_decoder,
                                               weight_decay=weight_decay, no_decay_bias_norm=True)
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[max_lr_encoder, max_lr_decoder],   # matches param_groups order if you keep 2 groups; otherwise pass a scalar
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # OneCycleLR is not chainable; use it directly (no extra warmup wrapper).
    return optimizer, scheduler, epochs


def build_scheme_E(model, train_loader_len: int):
    epochs = 400
    steps_per_epoch = train_loader_len
    warmup_steps = 10 * steps_per_epoch

    use_radam = True

    lr_decoder = 1e-3 if use_radam else 3e-4
    lr_encoder = lr_decoder * 0.3
    weight_decay = 1e-3

    param_groups = make_optimizer_param_groups(model, lr_encoder, lr_decoder, weight_decay, no_decay_bias_norm=True)

    if use_radam:
        optimizer = torch.optim.RAdam(param_groups, lr=lr_decoder, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # Warmup scheduler (iteration-based)
    warmup_cfg = WarmupConfig(warmup_steps=warmup_steps, warmup_type="exponential", start_factor=0.05, exp_k=5.0)
    warm = LambdaLR(optimizer, lr_lambda=WarmupFn(warmup_cfg))

    # Plateau scheduler (epoch-based)
    plateau = ReduceLROnPlateau(
        optimizer,
        mode="max",          # monitor val_dice
        factor=0.5,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-6,
        eps=1e-8,
    )

    return optimizer, warm, plateau, epochs, warmup_steps


def train_with_plateau(model, train_loader, val_loader, optimizer, warmup_sched, plateau_sched,
                       epochs: int, warmup_steps: int, device, scaler, criterion):
    global_step = 0

    best_dice = -1.0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # warmup step (per-iteration) only in the first warmup_steps
            if global_step < warmup_steps:
                warmup_sched.step()

            global_step += 1

        metrics = validate(model, val_loader, device)
        val_dice = metrics["val_dice"]

        # plateau step (per-epoch, needs metric)
        plateau_sched.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint("best.pt", model, optimizer, plateau_sched, scaler, epoch, extra={"best_dice": best_dice})
