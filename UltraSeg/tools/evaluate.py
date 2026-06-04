import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List
from thop import profile, clever_format
from torchinfo import summary
import time
import torch
import torch.nn as nn

from pathlib import PosixPath, Path

from UltraSeg.tools.plots import visualize_predictions
from UltraSeg.logger.logger import TQDM_BAR_FORMAT, log_write


def compute_class_weights_from_loader(dataloader,
                                      num_classes,
                                      method="sqrt",
                                      eps=1e-6,
                                      normalize=True):
    pixel_count = np.zeros(num_classes)
    pbar = tqdm(dataloader, desc="Computing Class Weights", bar_format=TQDM_BAR_FORMAT)
    for _, masks in pbar:
        # masks: (B, H, W)

        masks = masks.numpy()

        for c in range(num_classes):
            pixel_count[c] += np.sum(masks == c)

    total_pixels = pixel_count.sum()
    freq = pixel_count / total_pixels
    freq = np.clip(freq, eps, None)

    if method == "sqrt":
        class_weights = 1.0 / np.sqrt(freq)
    elif method == "log":
        class_weights = 1.0 / np.log(freq + eps + 1.02)
    else:
        raise ValueError

    if normalize:
        class_weights = class_weights / class_weights.sum()

    return class_weights


def evaluate_model(model: nn.Module,
                   val_loader: DataLoader,
                   mean: List[float] = None,
                   std: List[float] = None,
                   color_map: List[List[int]] = None,
                   save_path: PosixPath = None,
                   max_batch: int = 8,
                   device: torch.device = torch.device('cuda'),
                   model_type: str = 'ori',
                   use_roi: bool = False):

    model.eval().to(device)

    pbar = tqdm(val_loader, desc="Model inference on Validation Set", bar_format=TQDM_BAR_FORMAT)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(device), targets.to(device)

            logits = model(images)

            # postprocess
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).long()  # 预测值

            if batch_idx < max_batch and save_path is not None:
                # 可视化预测结果
                filename = f"{model_type}-batch_{batch_idx}-preds-masks.png"
                visualize_predictions(images=images,
                                      targets=targets,
                                      pred=pred,
                                      mean=mean,
                                      std=std,
                                      color_map=color_map,
                                      save_path=save_path / filename,
                                      use_roi=use_roi)

            pbar.set_postfix({"Save preds and masks": f"{filename}"})


def model_summary(model: torch.nn.Module,
                  logfile: str,
                  input_size: List[int] = [448, 448],
                  device: torch.device = torch.device('cpu')):
    input_tensor = torch.randn(1, 3, input_size[0], input_size[1]).to(device)  # 创建一个随机输入张量
    model.to(device).eval()  # 将模型设置为评估模式

    # 1. 使用 thop 计算总参数量和总FLOPs
    flops_total, params_total = profile(model, inputs=(input_tensor,))
    flops_total, params_total = clever_format([flops_total, params_total], "%.3f")  # 格式化输出

    log_write(logfile, "\n\n============================== model summary ==============================")
    # 2. 使用 torchinfo 获取逐层信息（包括计算量和参数量）
    # log_write(logfile, str(summary(model, input_size=input_tensor.shape, verbose=2, depth=0)))  # depth控制显示深度

    # 3. 计算模型大小（MB）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # 参数数量 * 每个参数的字节数
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2  # 转换为MB

    # 4. 测量推理时间（使用CPU）
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # 精确测量
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) * 1000 / 100  # 计算平均时间并转换为毫秒

    # 打印总览
    log_write(logfile, f"\nParameters total: {params_total}")
    log_write(logfile, f"\nComputational Complexity (FLOPs): {flops_total}")
    log_write(logfile, f"\nModel size: {size_all_mb:.2f} MB")
    log_write(logfile, f"\nAverage Inference Time (CPU): {avg_inference_time:.2f} ms (CPU)")
