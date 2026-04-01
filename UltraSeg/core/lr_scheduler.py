import math
import torch.nn as nn
from typing import Optional, List, Any, Dict, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (LambdaLR,
                                      StepLR,
                                      LinearLR,
                                      CosineAnnealingLR,
                                      OneCycleLR,
                                      ReduceLROnPlateau)


class Warmup:
    def __init__(self,
                 warmup_epochs: int = 5,
                 start_factor: float = 0.01,
                 exp_k: float = 2.0,
                 warmup_type: str = 'linear',
                 enabled: bool = True):
        """学习率预热

        Args:
            warmup_steps (int, optional): 预热步数. Defaults to 1000.
            start_factor (float, optional): 预热起始因子, 初始学习率为 start_factor * base_lr.
                Defaults to 0.001.
            exp_k (float, optional): 指数因子, 仅在 warmup_type 为 'exponential' 时生效.
                Defaults to 2.0.
            warmup_type (str, optional): 预热类型, 可选 'linear'、'cosine' 或 'exponential'.
                Defaults to 'linear'.
            enabled (bool, optional): 是否启用预热.
                Defaults to True.
        """
        self.enabled = enabled
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_type = warmup_type
        self.start_factor = start_factor
        self.exp_k = exp_k

    def __call__(self, step: int) -> float:
        """计算当前学习率调节因子

        Args:
            step (int): 当前步数
            base_lr (float): 基础学习率

        Returns:
            float: 计算后的学习率调节因子，范围为 [start_factor, 1.0]
        """
        if not self.enabled or step > self.warmup_epochs:
            return 1.0

        if self.warmup_epochs <= 0:
            return 1.0

        t = step / self.warmup_epochs

        if self.warmup_type == 'linear':
            lr_factor = self.start_factor + (1.0 - self.start_factor) * t
        elif self.warmup_type == 'cosine':
            lr_factor = self.start_factor + (1.0 - self.start_factor) * (1.0 - math.cos(math.pi * t)) / 2.0
        elif self.warmup_type == 'exponential':
            num = math.exp(self.exp_k * t) - 1.0
            den = math.exp(self.exp_k) - 1.0
            lr_factor = self.start_factor + (1.0 - self.start_factor) * (num / max(den, 1e-12))
        else:
            raise ValueError(f'Invalid warmup_type: {self.warmup_type}, only support `linear`, `cosine`, `exponential`.')

        return lr_factor


class Scheduler:
    def __init__(self,
                 warmup_scheduler: LambdaLR = None,
                 main_scheduler: str = 'cosine',
                 warmup_epochs: int = 5,
                 warmup_enabled: bool = True):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_enabled = warmup_enabled
        self.global_step = 0

    def step(self,
             metric: Optional[float] = None):
        if self.global_step < self.warmup_epochs and self.warmup_enabled:
            self.warmup_scheduler.step()
        else:
            if isinstance(self.main_scheduler, ReduceLROnPlateau):
                if metric is not None:
                    self.main_scheduler.step(metric)
                else:
                    raise ValueError('ReduceLROnPlateau scheduler requires metric to update learning rate.')
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
        if self.global_step <= self.warmup_epochs:
            return self.warmup_scheduler.get_last_lr()
        if hasattr(self.main_scheduler, "get_last_lr"):
            return self.main_scheduler.get_last_lr()

    def state_dict(self):
        return {
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'main_scheduler': self.main_scheduler.state_dict() if hasattr(self.main_scheduler, 'state_dict') else None,
            'warmup_epochs': self.warmup_epochs,
            'global_step': self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.warmup_epochs = int(state_dict['warmup_epochs'])
        self.global_step = int(state_dict['global_step'])
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        if state_dict.get('main_scheduler') is not None:
            self.main_scheduler.load_state_dict(state_dict['main_scheduler'])


def get_lr_scheduler(optimizer: Optimizer,
                     warmup_epochs: int = -1,
                     total_epochs: int = 100,
                     warmup_type: str = 'cosine',
                     warmup_start_factor: float = 0.001,
                     main_lr_type: str = 'cosine',
                     gamma: float = 0.1,
                     drop_steps: int = 10,
                     warmup_enabled: bool = True) -> Scheduler:
    """根据学习率调度器类型创建学习率调度器.

    Args:
        warmup_epochs (int): 预热轮数.
        main_lr_type (str): 主学习率调度器类型.
        gamma (float): 学习率衰减因子.
        drop_steps (int): 学习率衰减步长.
        optimizer (optim.Optimizer): 优化器实例.

    Returns:
        lr_scheduler (Scheduler): 学习率调度器实例.
    """
    warmup_fn = Warmup(warmup_epochs=warmup_epochs,
                       start_factor=warmup_start_factor,
                       warmup_type=warmup_type,
                       enabled=warmup_enabled)
    warmup_lr = LambdaLR(optimizer=optimizer, lr_lambda=warmup_fn)
    if main_lr_type == 'cosine':
        main_lr = CosineAnnealingLR(optimizer=optimizer,
                                    T_max=total_epochs,
                                    eta_min=0.0,
                                    last_epoch=warmup_epochs)
    elif main_lr_type == 'step':
        main_lr = StepLR(optimizer=optimizer,
                         step_size=drop_steps,
                         gamma=gamma)
    elif main_lr_type == 'linear':
        main_lr = LinearLR(optimizer=optimizer,
                           last_epoch=warmup_epochs,
                           total_steps=total_epochs)
    elif main_lr_type == 'reduce':
        main_lr = ReduceLROnPlateau(optimizer=optimizer,
                                    factor='min',
                                    patience=drop_steps,
                                    verbose=True)
    else:
        raise ValueError(f'Invalid main_lr_type: {main_lr_type}, only support `cosine`, `step`, `reduce`.')

    lr_scheduler = Scheduler(warmup_scheduler=warmup_lr,
                             main_scheduler=main_lr,
                             warmup_epochs=warmup_epochs,
                             warmup_enabled=warmup_enabled)

    return lr_scheduler
