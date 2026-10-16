import math
import argparse
import json
import platform
import torch
import torch.nn as nn
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Optional, List, Dict
from pathlib import PosixPath, Path
import sys

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from UltraSeg.logger.logger import TQDM_BAR_FORMAT, log_write
import segmentation_models_pytorch as smp
from UltraSeg.core.lr_scheduler import Scheduler
from UltraSeg.core.fileio import yaml_load
from UltraSeg.core.dataset import create_dataset
from UltraSeg.inference import load_model

IS_WINDOWS = platform.system() == 'Windows'


class OpenVINOModel:
    """Wrapper for OpenVINO IR model that mimics PyTorch nn.Module inference interface."""

    def __init__(self, xml_path: Path, device: str = "CPU"):
        if not HAS_OPENVINO:
            raise ImportError("openvino is not installed. Run: pip install openvino")
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"OpenVINO IR file not found: {xml_path}")
        self.core = ov.Core()
        self.ov_model = self.core.read_model(str(self.xml_path))
        self.compiled_model = self.core.compile_model(self.ov_model, device)

    def eval(self):
        """No-op: OpenVINO models are always in inference mode (no training mode)."""
        return self

    def to(self, device):
        """No-op: Device is set at compile time and cannot be changed."""
        return self

    def __call__(self, input_tensor):
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.cpu().numpy()
        else:
            input_np = input_tensor
        if input_np.ndim == 4 and input_np.shape[0] > 1:
            results = [self.compiled_model(input_np[i:i + 1])[0] for i in range(input_np.shape[0])]
            result = np.concatenate(results, axis=0)
        else:
            result = self.compiled_model(input_np)[0]
        return torch.from_numpy(result)


def load_ov_model_meta(xml_path: Path):
    """Extract metadata (input_size, mean, std, num_classes) from an OpenVINO IR model.

    The model's input/output shapes are read directly from the IR.
    Mean/std are read from a companion .yaml or .json file (same stem as the .xml),
    falling back to ImageNet defaults if not found.
    """
    if not HAS_OPENVINO:
        raise ImportError("openvino is not installed. Run: pip install openvino")
    core = ov.Core()
    ov_model = core.read_model(str(xml_path))

    def _to_int(val):
        import numpy as np
        if isinstance(val, (list, tuple, np.ndarray)):
            val = val[0] if len(val) > 0 else 512
        return int(val)

    def _flatten_input_size(val):
        import numpy as np
        result = []

        def _extract(x):
            if isinstance(x, (int, float, np.integer, np.floating)):
                result.append(int(x))
            elif isinstance(x, (list, tuple, np.ndarray)):
                for item in x:
                    _extract(item)
            else:
                try:
                    result.append(int(x))
                except Exception:
                    result.append(512)
        _extract(val)
        while len(result) < 2:
            result.append(512)
        return [result[0], result[1]]

    # ---- input size from model input shape (expects [1, 3, H, W] or [1, 3, H, W]) ----
    input_shape = ov_model.input(0).get_shape()
    log_write(f"DEBUG: OpenVINO input_shape={input_shape}, type={type(input_shape)}\n")
    log_write(f"DEBUG: input_shape[2]={input_shape[2]}, type={type(input_shape[2])}\n" if len(input_shape) > 2 else "DEBUG: input_shape length < 3\n")

    if len(input_shape) == 4:
        h, w = _to_int(input_shape[2]), _to_int(input_shape[3])
        input_size = [h, w]
    elif len(input_shape) == 3:
        h, w = _to_int(input_shape[1]), _to_int(input_shape[2])
        input_size = [h, w]
    else:
        input_size = [512, 512]

    log_write(f"DEBUG: raw input_size={input_size}, types={[type(x) for x in input_size]}\n")

    input_size = _flatten_input_size(input_size)
    log_write(f"DEBUG: flattened input_size={input_size}, types={[type(x) for x in input_size]}\n")

    # ---- num_classes from model output shape ([1, C, H, W]) ----
    output_shape = ov_model.output(0).get_shape()
    num_classes = _to_int(output_shape[1]) if len(output_shape) >= 2 else None

    # ---- mean / std from companion metadata file ----
    mean, std = None, None
    xml_path = xml_path if isinstance(xml_path, Path) else Path(xml_path)
    for suffix in ['.yaml', '.yml', '.json']:
        meta_path = xml_path.with_suffix(suffix)
        if meta_path.exists():
            if suffix == '.json':
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            else:
                import yaml
                with open(meta_path, 'r') as f:
                    meta = yaml.safe_load(f)
            if isinstance(meta, dict):
                mean = meta.get('mean', mean)
                std = meta.get('std', std)
            if mean is not None and std is not None:
                break

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    mean = [float(m) for m in mean]
    std = [float(s) for s in std]

    return input_size, mean, std, num_classes


def calculate_metrics_per_class(tp_all, fp_all, fn_all, tn_all, empty_classes=None):
    # 计算每个类别的 tp/fp/fn/tn
    tp_classes = tp_all.sum(dim=0, keepdim=True)
    fp_classes = fp_all.sum(dim=0, keepdim=True)
    fn_classes = fn_all.sum(dim=0, keepdim=True)
    tn_classes = tn_all.sum(dim=0, keepdim=True)

    accuracy_classes = smp.metrics.accuracy(tp=tp_classes,
                                            fp=fp_classes,
                                            fn=fn_classes,
                                            tn=tn_classes, reduction=None).squeeze(0)
    precision_classes = smp.metrics.precision(tp=tp_classes,
                                              fp=fp_classes,
                                              fn=fn_classes,
                                              tn=tn_classes, reduction=None).squeeze(0)
    recall_classes = smp.metrics.recall(tp=tp_classes,
                                        fp=fp_classes,
                                        fn=fn_classes,
                                        tn=tn_classes, reduction=None).squeeze(0)
    iou_classes = smp.metrics.iou_score(tp=tp_classes,
                                        fp=fp_classes,
                                        fn=fn_classes,
                                        tn=tn_classes, reduction=None).squeeze(0)
    dice_classes = smp.metrics.f1_score(tp=tp_classes,
                                        fp=fp_classes,
                                        fn=fn_classes,
                                        tn=tn_classes, reduction=None).squeeze(0)
    f2score_classes = smp.metrics.fbeta_score(tp=tp_classes,
                                              fp=fp_classes,
                                              fn=fn_classes,
                                              tn=tn_classes,
                                              beta=2, reduction=None).squeeze(0)

    # 用 0.0 替换空类别的 NaN 指标，避免传播到整体指标
    if empty_classes is not None and len(empty_classes) > 0:
        for c in empty_classes:
            if torch.isnan(accuracy_classes[c]):
                accuracy_classes[c] = 0.0
            if torch.isnan(precision_classes[c]):
                precision_classes[c] = 0.0
            if torch.isnan(recall_classes[c]):
                recall_classes[c] = 0.0
            if torch.isnan(iou_classes[c]):
                iou_classes[c] = 0.0
            if torch.isnan(dice_classes[c]):
                dice_classes[c] = 0.0
            if torch.isnan(f2score_classes[c]):
                f2score_classes[c] = 0.0

    metrics_classes = {"accuracy": accuracy_classes,
                       "precision": precision_classes,
                       "recall": recall_classes,
                       "iou": iou_classes,
                       "dice": dice_classes,
                       "f2score": f2score_classes
                       }
    return metrics_classes


def calculate_metrics_all_classes(tp_all, fp_all, fn_all, tn_all, class_weights=None, empty_classes=None):
    metrics = {"accuracy": 0,
               "precision": 0,
               "recall": 0,
               "iou": 0,
               "dice": 0,
               "f2score": 0}

    reductions = ['micro', 'macro', 'weighted', 'macro-imagewise']
    if class_weights is None:
        reductions = ['micro', 'macro', 'macro-imagewise']

    # 构造 ignore_index：对于 macro/weighted，排除空类别（让 smp 不将其纳入平均）
    # smp 的 reduction 中 macro/macro-imagewise 会自动跳过 tp+fp+fn=0 的类别，
    # 但我们仍然确保空类别在 class_weights 中权重为 0
    ori_class_weights = class_weights
    if empty_classes is not None and len(empty_classes) > 0 and ori_class_weights is not None:
        ori_class_weights = ori_class_weights.copy() if isinstance(ori_class_weights, list) else list(ori_class_weights)
        for c in empty_classes:
            ori_class_weights[c] = 0.0

    results = {}
    for reduction in reductions:
        class_weights = ori_class_weights
        if reduction not in results:
            results[reduction] = {}

        if reduction in ['micro', 'macro', 'macro-imagewise']:
            class_weights = None

        metrics["accuracy"] = smp.metrics.accuracy(tp=tp_all,
                                                   fp=fp_all,
                                                   fn=fn_all,
                                                   tn=tn_all,
                                                   class_weights=class_weights,
                                                   reduction=reduction).item()
        metrics["precision"] = smp.metrics.precision(tp=tp_all,
                                                     fp=fp_all,
                                                     fn=fn_all,
                                                     tn=tn_all,
                                                     class_weights=class_weights,
                                                     reduction=reduction).item()
        metrics["recall"] = smp.metrics.recall(tp=tp_all,
                                               fp=fp_all,
                                               fn=fn_all,
                                               tn=tn_all,
                                               class_weights=class_weights,
                                               reduction=reduction).item()
        metrics["iou"] = smp.metrics.iou_score(tp=tp_all,
                                               fp=fp_all,
                                               fn=fn_all,
                                               tn=tn_all,
                                               class_weights=class_weights,
                                               reduction=reduction).item()
        metrics["dice"] = smp.metrics.dice_score(tp=tp_all,
                                                 fp=fp_all,
                                                 fn=fn_all,
                                                 tn=tn_all,
                                                 class_weights=class_weights,
                                                 reduction=reduction).item()
        metrics["f2score"] = smp.metrics.fbeta_score(tp=tp_all,
                                                     fp=fp_all,
                                                     fn=fn_all,
                                                     tn=tn_all,
                                                     beta=2,
                                                     class_weights=class_weights,
                                                     reduction=reduction).item()
        results[reduction].update(metrics)
    return results


class ModelSaver:
    """
    用加权综合分数保存最优模型，并保存最近一个epoch的模型、优化器、学习率控制器、指标等。
    综合考虑：
    - val_loss（转成越大越好的loss_score）
    - weighted dice
    - weighted iou
    - weighted f2
    - weighted recall
    - weighted precision

    最优模型保存依据：
        composite_score 更高 -> 保存
        若分数几乎相同，则 val_loss 更低 -> 保存
    """

    def __init__(self,
                 best_model_pth: PosixPath,
                 last_model_pth: PosixPath,
                 ema_best_model_pth: PosixPath = None,
                 ema_last_model_pth: PosixPath = None,
                 metric_weights: Optional[List[float]] = None,
                 higher_is_better: bool = True,
                 min_delta: float = 1e-6,
                 loss_transform: str = "exp",  # "exp" or "reciprocal"
                 loss_alpha: float = 1.0,
                 metric_reduction: str = "weighted",
                 class_names: List[str] = None,
                 arch: str = 'unet',
                 encoder_name: str = 'mobilenet_v2',
                 decoder_attention_type: Optional[str] = None,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 input_size: Union[List[int], List[int]] = None,
                 mean: List[float] = None,
                 std: List[float] = None):

        self.min_delta = min_delta
        self.higher_is_better = higher_is_better
        self.loss_transform = loss_transform
        self.loss_alpha = loss_alpha
        self.class_names = class_names if class_names is not None else None

        assert metric_reduction in ['micro', 'macro', 'weighted', 'macro-imagewise', 'weighted-imagewise'], \
            f"Unsupported metric_reduction: {metric_reduction}"
        self.metric_reduction = metric_reduction

        self.class_weights = None  # 需要在训练过程中计算得到，并传入 save() 函数
        # 默认权重
        self.metric_weights = metric_weights or {"loss": 0.20,
                                                 "accuracy": 0.0,
                                                 "precision": 0.0,
                                                 "recall": 0.0,
                                                 "iou": 0.4,
                                                 "dice": 0.4,
                                                 "f2": 0.0,
                                                 }

        # 归一化权重
        s = sum(self.metric_weights.values())
        if abs(s - 1.0) > 1e-6:
            self.metric_weights = {k: v / s for k, v in self.metric_weights.items()}

        # 模型信息记录
        self.best_pth = {'ori': str(best_model_pth),
                         'ema': str(ema_best_model_pth) if ema_best_model_pth is not None else None}
        self.last_pth = {'ori': str(last_model_pth),
                         'ema': str(ema_last_model_pth) if ema_last_model_pth is not None else None}

        init_score = -float("inf") if higher_is_better else float("inf")
        self.improved = {'ori': False,
                         'ema': False}
        self.best_score = {'ori': init_score,
                           'ema': init_score
                           }  # 同时跟踪原始模型和EMA模型的最佳分数
        self.best_loss = {'ori': float("inf"),
                          'ema': float("inf")}  # 同时跟踪原始模型和EMA
        self.best_epoch = {'ori': -1,
                           'ema': -1}  # 同时跟踪原始模型和EMA模型
        self.best_metrics_all_classes = {'ori': {},
                                         'ema': {}}  # 同时跟踪原始模型和EMA模型的所有指标
        self.best_metrics_per_classes = {'ori': {},
                                         'ema': {}}  # 同时跟踪原始模型和EMA模型的每类指标
        self.saved_model_types = []
        self.meta_info = {}
        self.meta_info.update({
            "arch": arch,
            "encoder_name": encoder_name,
            "decoder_attention_type": decoder_attention_type,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "input_size": input_size,
            "mean": mean,
            "std": std,
        })

    def _loss_to_score(self, val_loss):
        """
        将 val_loss 映射为 [0,1] 左右的“越大越好”分数。
        两种常见方式：
        1) exp(-alpha * loss)
        2) 1 / (1 + loss)
        """
        if self.loss_transform == "exp":  # loss_score = exp( - loss)
            return math.exp(-self.loss_alpha * float(val_loss))
        elif self.loss_transform == "reciprocal":  # loss_score = 1 / (1 + loss)
            return 1.0 / (1.0 + float(val_loss))
        else:
            raise ValueError(f"Unsupported loss_transform: {self.loss_transform}")

    def compute_score(self,
                      val_loss: float = None,
                      metrics_all_classes: Optional[Dict] = None):

        if val_loss is None or metrics_all_classes is None:
            return None

        accuracy = metrics_all_classes[self.metric_reduction].get("accuracy", 0.0)
        precision = metrics_all_classes[self.metric_reduction].get("precision", 0.0)
        recall = metrics_all_classes[self.metric_reduction].get("recall", 0.0)
        iou = metrics_all_classes[self.metric_reduction].get("iou", 0.0)
        dice = metrics_all_classes[self.metric_reduction].get("dice", 0.0)
        f2score = metrics_all_classes[self.metric_reduction].get("f2score", 0.0)

        loss_score = self._loss_to_score(val_loss)

        score = (self.metric_weights["loss"] * loss_score
                 + self.metric_weights["accuracy"] * float(accuracy)
                 + self.metric_weights["precision"] * float(precision)
                 + self.metric_weights["recall"] * float(recall)
                 + self.metric_weights["iou"] * float(iou)
                 + self.metric_weights["dice"] * float(dice)
                 + self.metric_weights["f2"] * float(f2score))

        return score

    def is_improved(self, composite_score, val_loss, model_type='ori'):
        if composite_score is None or val_loss is None:
            return False

        if self.higher_is_better:
            if composite_score > self.best_score[model_type] + self.min_delta:
                return True
            elif abs(composite_score - self.best_score[model_type]) <= self.min_delta and val_loss < self.best_loss[model_type]:
                return True
            else:
                return False
        else:
            if composite_score < self.best_score[model_type] - self.min_delta:
                return True
            elif abs(composite_score - self.best_score[model_type]) <= self.min_delta and val_loss < self.best_loss[model_type]:
                return True
            else:
                return False

    def save(self,
             epoch: int,
             model: nn.Module,
             optimizer: Optimizer,
             lr_ctrl: Scheduler,
             val_loss: float,
             metrics_per_classes: Optional[Dict] = None,
             metrics_all_classes: Optional[Dict] = None,
             # EMA 模型相关参数
             ema_model: nn.Module = None,
             ema_val_loss: Optional[float] = None,
             ema_metrics_per_classes: Optional[Dict] = None,
             ema_metrics_all_classes: Optional[Dict] = None,
             class_weights: Optional[List[float]] = None):
        metrics_per_classes = {'ori': metrics_per_classes if metrics_per_classes is not None else None,
                               'ema': ema_metrics_per_classes if ema_metrics_per_classes is not None else None
                               }
        metrics_all_classes = {'ori': metrics_all_classes if metrics_all_classes is not None else None,
                               'ema': ema_metrics_all_classes if ema_metrics_all_classes is not None else None
                               }
        val_loss = {'ori': val_loss,
                    'ema': ema_val_loss if ema_val_loss is not None else None}

        composite_score = {'ori': None, 'ema': None}
        for model_type in ['ori', 'ema']:
            composite_score[model_type] = self.compute_score(val_loss=val_loss[model_type],
                                                             metrics_all_classes=metrics_all_classes[model_type])

        # ------------------------------ 保存最近一个epoch的模型 -------------------------------------------------
        last_state_dict = {"epoch": epoch,
                           "model_state_dict": deepcopy(model.state_dict()),
                           "optimizer_state_dict": deepcopy(optimizer.state_dict()) if optimizer is not None else None,
                           "lr_ctrl_state_dict": deepcopy(lr_ctrl.state_dict()) if lr_ctrl is not None else None,
                           "composite_score": composite_score["ori"],
                           "val_loss": float(val_loss["ori"]),
                           "class_names": self.class_names if self.class_names is not None else None,
                           "metrics_per_classes": metrics_per_classes["ori"],
                           "metrics_all_classes": metrics_all_classes["ori"],
                           "composite_weights": self.metric_weights,
                           "class_weights": class_weights if class_weights is not None else None,
                           "loss_transform": self.loss_transform,
                           }
        last_state_dict.update(meta_info=self.meta_info)  # 将模型的meta信息也保存到state_dict中
        # 保存最近一个epoch的模型
        torch.save(last_state_dict, str(self.last_pth['ori']))

        # 保存最近一个epoch的EMA模型
        if ema_model is not None:
            last_ema_state_dict = {"epoch": epoch,
                                   "model_state_dict": deepcopy(ema_model.state_dict()) if ema_model is not None else None,
                                   "composite_score": composite_score["ema"],
                                   "val_loss": float(val_loss["ema"]),
                                   "class_names": self.class_names if self.class_names is not None else None,
                                   "metrics_per_classes": metrics_per_classes["ema"],
                                   "metrics_all_classes": metrics_all_classes["ema"],
                                   "composite_weights": self.metric_weights,
                                   "class_weights": class_weights if class_weights is not None else None,
                                   "loss_transform": self.loss_transform,
                                   }
            last_ema_state_dict.update(meta_info=self.meta_info)  # 将模型的meta信息也保存到state_dict中
            if self.last_pth['ema'] is not None:
                torch.save(last_ema_state_dict, str(self.last_pth['ema']))

        # ------------------------------ 保存最优模型 -------------------------------------------------
        for model_type in ['ori', 'ema']:
            self.improved[model_type] = self.is_improved(composite_score[model_type],
                                                         val_loss[model_type],
                                                         model_type=model_type)
            if self.improved[model_type]:
                self.best_score[model_type] = composite_score[model_type]
                self.best_loss[model_type] = float(val_loss[model_type])
                self.best_epoch[model_type] = epoch
                self.best_metrics_all_classes[model_type] = metrics_all_classes[model_type]
                self.best_metrics_per_classes[model_type] = metrics_per_classes[model_type]

                save_dict = {"best_epoch": epoch,
                             "model_state_dict": deepcopy(model.state_dict()) if model_type == 'ori' else deepcopy(ema_model.state_dict()),
                             "composite_score": self.best_score[model_type],
                             "val_loss": self.best_loss[model_type],
                             "class_names": self.class_names if self.class_names is not None else None,
                             "metrics_per_classes": self.best_metrics_per_classes[model_type],
                             "metrics_all_classes": self.best_metrics_all_classes[model_type],
                             "composite_weights": self.metric_weights,
                             "class_weights": class_weights if class_weights is not None else None,
                             "loss_transform": self.loss_transform,
                             }
                save_dict.update(meta_info=self.meta_info)  # 将模型的meta信息也保存到state_dict中
                # 保存模型
                torch.save(save_dict, str(self.best_pth[model_type]))
                log_write(f"🚀 Epoch {epoch}: save improved {model_type} model to path {self.best_pth[model_type]}")

                if model_type not in self.saved_model_types:
                    self.saved_model_types.append(model_type)

                self.improved[model_type] = False  # 重置 improved 标志，等待下一个 epoch 的评估

        return composite_score

    def load_best_ckpt(self, model: nn.Module,
                       model_type='ori',
                       ckpt_path: Optional[Union[str, PosixPath]] = None):
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
        else:
            ckpt_path = Path(self.best_pth[model_type])

        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path))
            self.best_score = ckpt.get("composite_score", self.best_score)
            self.best_metrics_all_classes = ckpt.get("metrics_all_classes", None)
            self.best_metrics_per_classes = ckpt.get("metrics_per_classes", None)
            self.class_weights = ckpt.get("class_weights", None)

            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                log_write(f"Load the best {model_type} model from {ckpt_path}")

        else:
            log_write(f"No best {model_type} model found at {ckpt_path}")

        return model

    def get_best_info(self, model_type='ori'):
        assert model_type in self.saved_model_types, f"Model type {model_type} not saved"

        best_metrics = self.best_metrics_all_classes.get(model_type)
        best_metrics_per_classes = self.best_metrics_per_classes.get(model_type)
        composite_score = self.best_score.get(model_type)
        best_epoch = self.best_epoch.get(model_type)

        return best_epoch, composite_score, best_metrics, best_metrics_per_classes


def validate_one_epoch(epoch: int,
                       model: nn.Module,
                       val_loader: DataLoader,
                       loss_fn: nn.Module,
                       num_classes: int,
                       class_weights: Optional[List[float]] = None,
                       device: torch.device = torch.device('cuda'),
                       epochs: int = 100,
                       model_type: str = 'ori',
                       ignore_index: int = -1,
                       empty_classes: Optional[List[int]] = None,
                       is_ov_model: bool = False):
    if is_ov_model:
        pass
    else:
        model.eval().to(device)
    val_loss = []
    # Accumulate tp/fp/fn/tn batch-wise
    tp_all, fp_all, fn_all, tn_all = [], [], [], []

    pbar = tqdm(val_loader, desc=f"{model_type} - Val Epoch {epoch}/{epochs - 1}", bar_format=TQDM_BAR_FORMAT)

    with torch.no_grad():
        for (images, targets) in pbar:
            if not is_ov_model:
                images, targets = images.to(device), targets.to(device)

            logits = model(images)
            loss = loss_fn(logits, targets) if loss_fn is not None else None
            val_loss.append(loss.item()) if loss is not None else None

            # postprocess
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).long()

            tp, fp, fn, tn = smp.metrics.get_stats(output=pred,
                                                   target=targets.long(),
                                                   mode='multiclass',
                                                   num_classes=num_classes,
                                                   ignore_index=ignore_index)
            tp_all.append(tp.cpu())
            fp_all.append(fp.cpu())
            fn_all.append(fn.cpu())
            tn_all.append(tn.cpu())
            # pbar.set_postfix({f"{model_type} model val_loss": f"{(sum(val_loss) / len(val_loss)):.4f}"} if loss_fn is not None else {})

        # 聚合所有批次的 tp/fp/fn/tn，得到形状为[num_images, num_classes] 的张量
        val_loss = sum(val_loss) / len(val_loss) if loss_fn is not None else None
        tp_all = torch.cat(tp_all, dim=0)
        fp_all = torch.cat(fp_all, dim=0)
        fn_all = torch.cat(fn_all, dim=0)
        tn_all = torch.cat(tn_all, dim=0)

        metrics_per_classes = calculate_metrics_per_class(tp_all, fp_all, fn_all, tn_all, empty_classes=empty_classes)

        # micro 平均指标
        metrics_all_classes = calculate_metrics_all_classes(tp_all,
                                                            fp_all,
                                                            fn_all,
                                                            tn_all,
                                                            class_weights=class_weights,
                                                            empty_classes=empty_classes)

        return metrics_per_classes, metrics_all_classes, val_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('-cp', '--checkpoint', type=str,
                        default='res/zhou-seg/best/timm-tf_efficientnet_lite1-no-att-no-roi-hard-samp-512-exp29/weights/ema_best.pth',
                        help='Path to model checkpoint (.pth) or OpenVINO IR model (.xml)')
    parser.add_argument('-d', '--dataset_cfg', type=str,
                        default='UltraSeg/config/dataset/zhou/zhou.yaml',
                        help='dataset config file')
    parser.add_argument('--split', default='test', help='split to use for validation')
    parser.add_argument('-cls', '--num_classes', type=int, default=19, help='number of classes (auto-detected for OpenVINO models if not set)')
    parser.add_argument('-ig', '--ignore_index', type=int, nargs='+', default=[-1],
                        help='ignore indices for metrics calculation')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--ov_device', default='CPU', help='OpenVINO inference device, e.g. CPU or GPU (only used for .xml models)')

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def print_metrics_table(metrics_per_classes, metrics_all_classes, class_names, logfile=None):
    metrics_row = ["accuracy", "precision", "recall", "iou", "dice", "f2score"]
    table = PrettyTable()
    table.field_names = ["reduction"] + metrics_row
    for key, value in metrics_all_classes.items():
        data_row = []
        for k in metrics_row:
            v = value[k]
            data_row.append(f"{v:.4f}")
        table.add_row([key] + data_row)

    table_cls = PrettyTable()
    table_cls.field_names = ["metric"] + class_names

    for key, value in metrics_per_classes.items():
        data_rows = [key]
        for class_idx in range(len(value)):
            data_rows.append(f"{value[class_idx]:.4f}")
        table_cls.add_row(data_rows)

    log_write(f"Metrics for all classes:\n{table}\n", logfile)
    log_write(f"Metrics per class:\n{table_cls}\n", logfile)


if __name__ == '__main__':
    opts = parse_args()

    ckpt_path = Path(opts.checkpoint)
    # auto-detect model format by file extension
    is_ov_model = ckpt_path.suffix == '.xml'

    if is_ov_model:
        if not HAS_OPENVINO:
            raise ImportError("openvino is not installed. Run: pip install openvino")

        model = OpenVINOModel(ckpt_path, device=opts.ov_device)
        input_size, mean, std, ov_num_classes = load_ov_model_meta(ckpt_path)

        if opts.num_classes is None:
            opts.num_classes = ov_num_classes

        # exp_dir: use parent of the .xml file as output directory
        exp_dir = ckpt_path.parents[1] if len(ckpt_path.parents) >= 2 else ckpt_path.parent

        log_write(f"Loaded OpenVINO IR model from {ckpt_path}\n"
                  f"  input_size={input_size}, num_classes={opts.num_classes}\n"
                  f"  device={opts.ov_device}\n")
    else:
        model, input_size, mean, std = load_model(ckpt_path)
        exp_dir = ckpt_path.parents[2]

        if opts.num_classes is None:
            opts.num_classes = 10  # fallback default for PyTorch checkpoints

    dataset_cfg = yaml_load(opts.dataset_cfg)
    dataset_name = Path(opts.dataset_cfg).stem
    data_classes = dataset_cfg['classes'][:opts.num_classes] if opts.num_classes is not None else dataset_cfg['classes'][:10]

    logfile = exp_dir / f'test_on_{dataset_name}_{opts.split}.log'
    logfile.parent.mkdir(parents=True, exist_ok=True)

    model_name = "OpenVINO" if is_ov_model else "PyTorch"
    log_write(f"Testing [{model_name}] on {dataset_name} {opts.split}, ignore_index: {opts.ignore_index}\n", logfile)
    log_write(f"Checkpoint: {ckpt_path}\n", logfile)

    if isinstance(input_size, int):
        input_size = [input_size, input_size]
    elif isinstance(input_size, (list, tuple)):
        import numpy as np
        flat = []

        def _extract(x):
            if isinstance(x, (int, float, np.integer, np.floating)):
                flat.append(int(x))
            elif isinstance(x, (list, tuple, np.ndarray)):
                for item in x:
                    _extract(item)
        _extract(input_size)
        if len(flat) >= 2:
            input_size = [flat[0], flat[1]]
        elif len(flat) == 1:
            input_size = [flat[0], flat[0]]
        else:
            input_size = [512, 512]
    else:
        input_size = [512, 512]

    log_write(f"DEBUG: final input_size for dataset={input_size}, types={[type(x) for x in input_size]}\n")

    from UltraSeg.core.dataset import Ultrasound_Dataset

    if is_ov_model:
        validation_dataset = Ultrasound_Dataset(data_root=dataset_cfg['data_root'],
                                                img_dir=dataset_cfg['img_dir'],
                                                mask_dir=dataset_cfg['mask_dir'],
                                                img_suffix=dataset_cfg['img_suffix'],
                                                mask_suffix=dataset_cfg['mask_suffix'],
                                                input_size=input_size,
                                                mean=mean,
                                                std=std,
                                                num_classes=dataset_cfg['num_classes'],
                                                classes=dataset_cfg['classes'],
                                                color_map=dataset_cfg['color_map'],
                                                mode=opts.split,
                                                augment_version=2)
    else:
        validation_dataset = create_dataset(dataset_cfg=dataset_cfg,
                                            split=opts.split,
                                            input_size=input_size[0] if isinstance(input_size, (list, tuple)) else input_size,
                                            mean=mean, std=std)

    val_loader = torch.utils.data.DataLoader(validation_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True,
                                             persistent_workers=IS_WINDOWS,
                                             prefetch_factor=4 if IS_WINDOWS else None)

    dev = torch.device(f'cuda:{opts.device}' if torch.cuda.is_available() and opts.device != 'cpu' else 'cpu')

    metrics_per_classes, metrics_all_classes, _ = validate_one_epoch(
        epoch=0,
        model=model,
        val_loader=val_loader,
        loss_fn=None,
        num_classes=opts.num_classes if opts.num_classes is not None else dataset_cfg.get('num_classes'),
        class_weights=None,
        device=dev,
        epochs=1,
        model_type='ema_best',
        empty_classes=opts.ignore_index,
        is_ov_model=is_ov_model)

    print_metrics_table(metrics_per_classes, metrics_all_classes, data_classes, logfile=logfile)
