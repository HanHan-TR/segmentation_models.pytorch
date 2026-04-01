import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any


def split_encoder_decoder_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """将模型参数分为编码器参数组和解码器/头部参数组, 该函数采用启发式方法，根据参数名称前缀将参数分组：
    - 名称以 'encoder.' 开头的参数被分配到编码器组
    - 其他参数被分配到解码器/头部组.

    Args:
        model (nn.Module): PyTorch 模型实例

    Returns:
        enc (List[nn.Parameter]): 编码器参数列表.

        dec (List[nn.Parameter]): 解码器/头部参数列表.
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


def make_optimizer_param_groups(model: nn.Module,
                                lr_encoder: float,
                                lr_decoder: float,
                                weight_decay: float,
                                no_decay_bias_norm: bool = True) -> List[Dict[str, Any]]:
    """为模型创建优化器参数组，支持为编码器和解码器设置不同的学习率和权重衰减策略.

    Args:
        model (nn.Module): 待优化的神经网络模型.
        lr_encoder (float): 编码器部分的学习率.
        lr_decoder (float): 解码器部分的学习率.
        weight_decay (float): 权重衰减系数.
        no_decay_bias_norm (bool, optional): 是否不对偏置和归一化参数应用权重衰减.
            默认为 True, 这是一种常见的优化策略.

    Returns:
        param_groups (List[Dict[str, Any]]): 优化器参数组列表，每个字典包含参数及其对应的学习率和权重衰减设置.

        - 当 no_decay_bias_norm 为 True 时，返回的参数列表为：[encoder_decay, encoder_no_decay, decoder_decay, decoder_no_decay]
        - 当 no_decay_bias_norm 为 False 时，返回的参数列表为：[encoder, decoder]

    Example:
        >>> param_groups = make_optimizer_param_groups(
        ...     model=model,
        ...     lr_encoder=1e-4,
        ...     lr_decoder=1e-3,
        ...     weight_decay=1e-4,
        ...     no_decay_bias_norm=True
        ... )
        >>> optimizer = torch.optim.AdamW(param_groups)
    """
    weight_decay = float(weight_decay)
    enc_params, dec_params = split_encoder_decoder_params(model)

    def group_params(params: List[nn.Parameter], lr: float):
        if not no_decay_bias_norm:
            # 当 no_decay_bias_norm 为 False 时，即不对偏置项与归一化参数应用权重衰减，
            # 所有参数作为一组，应用相同的学习率和权重衰减
            return [{"params": params, "lr": lr, "weight_decay": weight_decay}]

        # 当 no_decay_bias_norm 为 True 时，将参数分为两组：
        # - decay: 包含维度大于1的参数，通常是卷积核、全连接层的权重等
        # - no_decay: 包含维度等于1的参数，通常是偏置项、归一化参数等
        decay, no_decay = [], []
        for p in params:
            # We cannot reliably check "is norm" without names; this is minimal.
            if p.ndim == 1:  # or "norm" in p.name or "bias" in p.name:  # often bias or norm scale
                no_decay.append(p)
            else:
                decay.append(p)

        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    # 在 Python 中，+ 运算符的行为是由对象的 __add__ 方法定义的。
    # 对于列表类型，__add__ 方法被实现为连接操作，而不是元素级相加
    param_groups = group_params(enc_params, lr_encoder) + group_params(dec_params, lr_decoder)
    return param_groups


def get_optimizer(optimizer_type: str,
                  model: nn.Module,
                  decoder_lr: float,
                  encoder_lr_factor: float,
                  weight_decay: float,
                  momentum: float) -> torch.optim.Optimizer:

    param_groups = make_optimizer_param_groups(model,
                                               lr_decoder=decoder_lr,
                                               lr_encoder=encoder_lr_factor * decoder_lr,
                                               weight_decay=weight_decay)

    if optimizer_type == "SGD":
        return torch.optim.SGD(param_groups, momentum=momentum)
    elif optimizer_type == "AdamW":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer type {optimizer_type} not supported")
