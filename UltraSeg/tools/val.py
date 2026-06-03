import math
from os import path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Optional, List, Dict
import wandb
from pathlib import PosixPath, Path
from UltraSeg.logger.logger import TQDM_BAR_FORMAT
import segmentation_models_pytorch as smp
from UltraSeg.core.lr_scheduler import Scheduler


def calculate_metrics_per_class(tp_all, fp_all, fn_all, tn_all):
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

    metrics_classes = {"accuracy": accuracy_classes,
                       "precision": precision_classes,
                       "recall": recall_classes,
                       "iou": iou_classes,
                       "dice": dice_classes,
                       "f2score": f2score_classes
                       }
    return metrics_classes


def calculate_metrics_all_classes(tp_all, fp_all, fn_all, tn_all, class_weights=None):
    metrics = {"accuracy": 0,
               "precision": 0,
               "recall": 0,
               "iou": 0,
               "dice": 0,
               "f2score": 0}

    reductions = ['micro', 'macro', 'weighted', 'macro-imagewise']
    if class_weights is None:
        reductions = ['micro', 'macro', 'macro-imagewise']

    results = {}
    ori_class_weights = class_weights
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
        metrics["dice"] = smp.metrics.f1_score(tp=tp_all,
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
        self.metric_weights = metric_weights or {"loss": 0.10,
                                                 "accuracy": 0.0,
                                                 "precision": 0.1,
                                                 "recall": 0.15,
                                                 "iou": 0.30,
                                                 "dice": 0.35,
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
                print(f"[BestModelSaver] Saved best {model_type} model to path {self.best_pth[model_type]}")

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
                print(f"[BestModelSaver] Loaded model from {ckpt_path}, composite_score={self.best_score:.4f}")

        else:
            print(f"[BestModelSaver] No best model found at {self.best_pth}")

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
                       class_weights: Optional[List[float]] = None,
                       device: torch.device = torch.device('cuda'),
                       epochs: int = 100,
                       model_type: str = 'ori'):
    model.eval().to(device)
    val_loss = []
    # Accumulate tp/fp/fn/tn batch-wise
    tp_all, fp_all, fn_all, tn_all = [], [], [], []

    pbar = tqdm(val_loader, desc=f"{model_type} - Val Epoch {epoch}/{epochs - 1}", bar_format=TQDM_BAR_FORMAT)

    with torch.no_grad():
        for (images, targets) in pbar:
            images, targets = images.to(device), targets.to(device)

            logits = model(images)
            loss = loss_fn(logits, targets)
            val_loss.append(loss.item())

            # postprocess
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).long()

            tp, fp, fn, tn = smp.metrics.get_stats(output=pred,
                                                   target=targets.long(),
                                                   mode='multiclass',
                                                   num_classes=10,
                                                   ignore_index=-1)
            tp_all.append(tp.cpu())
            fp_all.append(fp.cpu())
            fn_all.append(fn.cpu())
            tn_all.append(tn.cpu())
            pbar.set_postfix({f"{model_type} model val_loss": f"{(sum(val_loss) / len(val_loss)):.4f}"})

        # 聚合所有批次的 tp/fp/fn/tn，得到形状为[num_images, num_classes] 的张量
        val_loss = sum(val_loss) / len(val_loss)
        tp_all = torch.cat(tp_all, dim=0)
        fp_all = torch.cat(fp_all, dim=0)
        fn_all = torch.cat(fn_all, dim=0)
        tn_all = torch.cat(tn_all, dim=0)

        metrics_per_classes = calculate_metrics_per_class(tp_all, fp_all, fn_all, tn_all)

        # micro 平均指标
        metrics_all_classes = calculate_metrics_all_classes(tp_all,
                                                            fp_all,
                                                            fn_all,
                                                            tn_all,
                                                            class_weights=class_weights)

        return metrics_per_classes, metrics_all_classes, val_loss
