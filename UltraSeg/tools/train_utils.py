import torch
import wandb
from torch import nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from UltraSeg.core.ema import EMA
from UltraSeg.logger.logger import TQDM_BAR_FORMAT
from UltraSeg.tools.plots import visualize_predictions


def train_one_epoch(epoch: int,
                    model: nn.Module,
                    train_loader: DataLoader,
                    loss_fn: nn.Module,
                    optimizer: Optimizer,
                    scaler: GradScaler,
                    color_map: list,
                    ema: EMA = None,
                    max_grad_norm: float = 1.0,
                    device: torch.device = torch.device('cuda'),
                    epochs: int = 100,
                    mean: list = None,
                    std: list = None,
                    use_roi: bool = False,
                    save_path: Path = None):
    model.train().to(device)
    if ema is not None:
        ema.to(device)

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs-1}", bar_format=TQDM_BAR_FORMAT)
    train_loss = []
    for idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = loss_fn(logits, masks)

        if idx < 3 and epoch == 0:  # 记录前3个batch的训练损失到wandb
            # postprocess
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).long()

            # 可视化预测结果
            if save_path is not None:
                filename = f"training-epoch{epoch}-batch_{idx}-preds-masks.png"
                visualize_predictions(images=images,
                                      targets=masks,
                                      pred=pred,
                                      mean=mean,
                                      std=std,
                                      color_map=color_map,
                                      save_path=save_path / filename,
                                      use_roi=use_roi)

        train_loss.append(loss.item())
        # loss.backward()
        scaler.scale(loss).backward()  # 计算梯度

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)  # 取消缩放，以便梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪

        scaler.step(optimizer)  # 更新参数
        scaler.update()  # 更新缩放因子

        if ema is not None:
            ema.update(model)  # 更新 EMA 模型

        postfix = {"train loss": f"{sum(train_loss) / len(train_loss):.4f}"}
        pbar.set_postfix(postfix)

    train_loss = sum(train_loss) / len(train_loss)
    return train_loss
