import argparse
import json
import numpy as np
import optuna
import swanlab
from pathlib import Path
import os
import platform
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import sys
import torch
import gc
from prettytable import PrettyTable
import wcwidth
import shutil

IS_WINDOWS = platform.system() == 'Windows'

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))
from torch.utils.data import DataLoader, WeightedRandomSampler
from segmentation_models_pytorch import create_model
from UltraSeg.core.fileio import yaml_load, yaml_save, increment_path, dict_to_obj, object_to_dict
from UltraSeg.core.sweep import (build_search_space,
                                 derive_epoch_metric_key,
                                 export_trials_dataframe,
                                 find_incompatible_params,
                                 get_trial_params,
                                 suggest_hyperparams)
from UltraSeg.core.initialize import init_random_seed, set_random_seed
from UltraSeg.core.dataset import create_dataset
from UltraSeg.tools.val import validate_one_epoch, ModelSaver
from UltraSeg.tools.train_utils import train_one_epoch
from UltraSeg.core.losses import Loss
from UltraSeg.core.optimizer import get_optimizer
from UltraSeg.core.lr_scheduler import get_lr_scheduler
from UltraSeg.tools.evaluate import evaluate_model
from UltraSeg.core.ema import EMA

from UltraSeg.logger.logger import get_environment_info, get_experiment_info, log_write


def _to_python(value):
    """把 torch.Tensor / numpy 标量 / numpy 数组 / Path 转成 JSON 可序列化的原生类型。"""
    if isinstance(value, torch.Tensor):
        # 单元素张量取标量，多元素张量转成列表
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def sanitize_metrics(metrics: dict) -> dict:
    """过滤 None 值，并把 tensor / numpy 标量转成 python 原生类型，方便 SwanLab 与 JSON 记录。"""
    sanitized = {}
    for key, value in metrics.items():
        if value is None:
            continue
        sanitized[key] = _to_python(value)
    return sanitized


def train(config, trial: optuna.Trial = None):
    """训练一个实验。

    Args:
        config: 超参数配置对象（支持点号访问）。
        trial: 传入 Optuna trial 时，会在每个 epoch 上报优化指标并支持剪枝；
            为 None 时即为普通的单次训练（例如 base_train.py）。
    Returns:
        dict: 训练结束时记录的指标（与 SwanLab 中 best_* 系列的键一致），
            供 Optuna objective 读取优化目标。
    """

    try:
        os.stat(config.exp_dir)
    except Exception:
        os.makedirs(config.exp_dir)

    # logfile = str(config.exp_dir / 'extractor_train.log')
    weight_dir, cfg_dir, plot_dir = Path(config.exp_dir) / 'weights', Path(config.exp_dir) / 'cfg', Path(config.exp_dir) / 'plots'
    best_model_pth, last_model_pth = weight_dir / 'best.pth', weight_dir / 'last.pth'
    ema_best_model_pth, ema_last_model_pth = weight_dir / 'ema_best.pth', weight_dir / 'ema_last.pth'

    plot_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # epochs / batch_size 必须是整数：Optuna 采样或手写 hyper.yaml 都可能给出浮点值
    for key in ('epochs', 'batch_size'):
        value = getattr(config, key, None)
        if value is not None:
            setattr(config, key, int(value))

    # Load configs
    if config.model_cfg is None:
        model_cfg = {}
        model_cfg['arch'] = config.arch
        model_cfg['encoder_name'] = config.encoder_name
        model_cfg['encoder_init_weights'] = config.encoder_init_weights
    else:
        model_cfg = yaml_load(config.model_cfg)

    dataset_cfg = yaml_load(config.dataset_cfg)
    data_classes = dataset_cfg['classes']
    data_classes = [f"{item:^8}" for item in data_classes]

    # 读取空类别配置（无样本的类别），用于评估时忽略
    empty_classes = dataset_cfg.get('empty_classes', []) or []

    # save configs
    yaml_save(cfg_dir / 'model.yaml', model_cfg)
    yaml_save(cfg_dir / 'dataset.yaml', dataset_cfg)
    yaml_save(cfg_dir / 'hyper.yaml', config.hyper_cfg) if hasattr(config, 'hyper_cfg') else yaml_save(cfg_dir / 'hyper.yaml', config)

    # 设置随机种子, 保证算法的可复现性
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    seed = init_random_seed(seed=config.seed, device=device)
    set_random_seed(seed, deterministic=True)

    # Create dataset
    normalize_type = dataset_cfg.get('normalize_type')
    train_dataset = create_dataset(dataset_cfg,
                                   input_size=config.input_size,
                                   mean=dataset_cfg['mean'][normalize_type],
                                   std=dataset_cfg['std'][normalize_type],
                                   split='train',
                                   use_roi=config.use_roi,
                                   hard_samp=config.hard_samp,
                                   rare_classes=config.rare_classes,
                                   augment_version=config.augment_version,
                                   augment_cfg_save_path=cfg_dir / 'augment_cfg.json')
    val_dataset = create_dataset(dataset_cfg,
                                 input_size=config.input_size,
                                 mean=dataset_cfg['mean'][normalize_type],
                                 std=dataset_cfg['std'][normalize_type],
                                 split='val',
                                 use_roi=config.use_roi,
                                 hard_samp=False,
                                 augment_version=config.augment_version)
    if config.hard_samp:
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(train_dataset.sample_weights),
                                        num_samples=len(train_dataset.sample_weights) * 2,  # 进行过采样，迭代器的长度是原来的两倍
                                        replacement=True)
        # Windows: persistent_workers=True 避免每个epoch重复创建进程（spawn开销大）
        # Linux: persistent_workers=False 及时释放文件描述符，避免 "Too many open files"
        # prefetch_factor: Windows=4，Linux=2，平衡吞吐量和资源占用
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  sampler=sampler,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=IS_WINDOWS,
                                  prefetch_factor=4 if IS_WINDOWS else None)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=IS_WINDOWS,
                                  prefetch_factor=4 if IS_WINDOWS else None)

    # 空类别：将 class_rarity 对应位置设为 0，避免影响 weighted 平均指标
    if len(empty_classes) > 0:
        for c in empty_classes:
            train_dataset.class_rarity[c] = 0.0

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             persistent_workers=IS_WINDOWS,
                                             prefetch_factor=4 if IS_WINDOWS else None)

    # Create model saver
    model_saver = ModelSaver(best_model_pth=best_model_pth,
                             last_model_pth=last_model_pth,
                             ema_best_model_pth=ema_best_model_pth,
                             ema_last_model_pth=ema_last_model_pth,
                             metric_reduction="weighted",
                             arch=model_cfg['arch'],
                             encoder_name=model_cfg['encoder_name'],
                             decoder_attention_type=config.decoder_attention_type,
                             in_channels=3,
                             num_classes=dataset_cfg['num_classes'],
                             input_size=config.input_size,
                             mean=train_dataset.get_mean(),
                             std=train_dataset.get_std(),
                             )
    # Create model
    model = create_model(arch=model_cfg['arch'],
                         encoder_name=model_cfg['encoder_name'],
                         encoder_weights=model_cfg['encoder_init_weights'],
                         decoder_attention_type=config.decoder_attention_type,
                         in_channels=3,
                         classes=dataset_cfg['num_classes'])

    # Get environment info
    get_environment_info()
    # Get experiment info
    get_experiment_info(cfg_dir=cfg_dir, model=model, input_size=config.input_size, logfile=None)
    # ! Load checkpoint if specified  for training of stage 2. In stage 2, we use the best model from stage 1 as the initial model.
    # ! In stage 1, the config.load_from_ckpt should be set to None, and the model will be initialized with
    # ! imagenet weights (as specified by model_cfg['encoder_weights']).
    if config.load_from_ckpt is not None:
        model_saver.load_best_ckpt(model=model,
                                   ckpt_path=config.load_from_ckpt)
        shutil.copy(config.load_from_ckpt, best_model_pth)  # 将指定的 ckpt 复制到 best_model_pth，方便后续加载和管理

    # Create optimizer
    optimizer = get_optimizer(optimizer_type=config.optimizer,
                              model=model,
                              decoder_lr=config.decoder_lr,
                              encoder_lr_factor=config.encoder_lr_factor,
                              weight_decay=config.weight_decay,
                              momentum=config.momentum)

    # Create lr scheduler
    epochs = config.epochs
    warmup_epochs = epochs * config.warmup_epochs_ratio
    lr_scheduler = get_lr_scheduler(optimizer,
                                    warmup_epochs=warmup_epochs,
                                    total_epochs=epochs,
                                    warmup_type=config.warmup_type,
                                    warmup_start_factor=config.warmup_start_factor,
                                    main_lr_type=config.main_lr_type,
                                    warmup_enabled=True)

    # Create loss function
    loss_fn = Loss(losses=config.loss_type,
                   mode='multiclass',
                   alpha=config.alpha)

    scaler = torch.amp.GradScaler()

    # Create EMA model
    ema = EMA(model, decay=config.ema_decay) if config.ema_decay > 0 else None
    metrics_row = ["accuracy", "precision", "recall", "iou", "dice", "f2score"]

    # 训练过程指标历史：除 SwanLab 云端记录外，再落盘一份 JSON，
    # 便于离线分析（替代原先从 wandb output.log 中解析指标的做法）
    metrics_history = []
    first_train_loss, last_train_loss = None, None

    # Optuna 剪枝所用的优化指标：每个 epoch 上报的同名指标键
    epoch_metric_key = None
    if trial is not None:
        opt_metric = getattr(config, 'opt_metric', 'best_ema/metric(macro)/dice')
        epoch_metric_key = derive_epoch_metric_key(opt_metric)

    for epoch in range(epochs):
        log_metrics = {}
        train_loss = train_one_epoch(epoch=epoch,
                                     model=model,
                                     color_map=dataset_cfg['color_map'],
                                     ema=ema,
                                     train_loader=train_loader,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     scaler=scaler,
                                     device=device,
                                     epochs=epochs,
                                     use_roi=config.use_roi,
                                     mean=train_dataset.get_mean(),
                                     std=train_dataset.get_std(),
                                     save_path=plot_dir)

        lr = lr_scheduler.get_last_lr()
        # 学习率与验证指标合并到同一个 step 记录，避免产生多余的 step
        log_metrics.update({"learning_rate/encoder_decay": lr[0],
                            "learning_rate/encoder_no_decay": lr[1],
                            "learning_rate/decoder_decay": lr[2],
                            "learning_rate/decoder_no_decay": lr[3],
                            })
        metrics_per_classes, metrics_all_classes, val_loss = validate_one_epoch(epoch=epoch,
                                                                                model=model,
                                                                                val_loader=val_loader,
                                                                                loss_fn=loss_fn,
                                                                                num_classes=dataset_cfg['num_classes'],
                                                                                class_weights=train_dataset.class_rarity,
                                                                                device=device,
                                                                                epochs=epochs,
                                                                                model_type='ori',
                                                                                empty_classes=empty_classes)

        log_metrics.update({"epoch": epoch,
                            "loss/ori_train": train_loss,
                            "loss/ori_val": val_loss})
        lr_scheduler.step()

        ori_table = PrettyTable()
        ori_table.field_names = ["reduction"] + metrics_row
        ori_table._string_width = wcwidth.wcswidth
        for key, value in metrics_all_classes.items():  # reduction: acc: value
            data_row = []
            for k in metrics_row:
                v = value[k]
                data_row.append(f"{v:.4f}")
                new_key = f"ori_metric({key})/{k}"
                log_metrics.update({new_key: v})
            ori_table.add_row([key] + data_row)
        log_write(f"Epoch {epoch} - ori Model Validation Metrics:\n{ori_table}\n")

        ori_cls_table = PrettyTable()
        ori_cls_table.field_names = ["metric"] + data_classes
        ori_cls_table._string_width = wcwidth.wcswidth
        for key, value in metrics_per_classes.items():
            data_rows = [key]
            for class_idx in range(len(value)):
                new_key = f"ori_cls_metric({key})/c{class_idx}"
                data_rows.append(f"{value[class_idx]:.4f}")
                log_metrics.update({new_key: value[class_idx]})
            ori_cls_table.add_row(data_rows)
        log_write(f"Epoch {epoch} - ori Model Validation Metrics Per Class:\n{ori_cls_table}\n")

        if ema is not None:
            ema_metrics_per_classes, ema_metrics_all_classes, ema_val_loss = validate_one_epoch(epoch=epoch,
                                                                                                model=ema.model(),
                                                                                                val_loader=val_loader,
                                                                                                loss_fn=loss_fn,
                                                                                                num_classes=dataset_cfg['num_classes'],
                                                                                                class_weights=train_dataset.class_rarity,
                                                                                                device=device,
                                                                                                epochs=epochs,
                                                                                                model_type='ema',
                                                                                                empty_classes=empty_classes)
            log_metrics.update({"loss/ema_val": ema_val_loss})

            ema_table = PrettyTable()
            ema_table.field_names = ["reduction"] + metrics_row
            ema_table._string_width = wcwidth.wcswidth
            for key, value in ema_metrics_all_classes.items():  # reduction: acc: value
                data_row = []
                for k in metrics_row:
                    v = value[k]
                    data_row.append(f"{v:.4f}")
                    new_key = f"ema_metric({key})/{k}"
                    log_metrics.update({new_key: v})
                ema_table.add_row([key] + data_row)
            log_write(f"Epoch {epoch} - ema Model Validation Metrics:\n{ema_table}\n")

            ema_cls_table = PrettyTable()
            ema_cls_table.field_names = ["metric"] + data_classes
            ema_cls_table._string_width = wcwidth.wcswidth
            for key, value in ema_metrics_per_classes.items():
                data_rows = [key]
                for class_idx in range(len(value)):
                    new_key = f"ema_cls_metric({key})/c{class_idx}"
                    data_rows.append(f"{value[class_idx]:.4f}")
                    log_metrics.update({new_key: value[class_idx]})
                ema_cls_table.add_row(data_rows)
            log_write(f"Epoch {epoch} - ema Model Validation Metrics Per Class:\n{ema_cls_table}\n")

        composite_score = model_saver.save(epoch=epoch,
                                           model=model,
                                           optimizer=optimizer,
                                           lr_ctrl=lr_scheduler,
                                           val_loss=val_loss,
                                           metrics_per_classes=metrics_per_classes,
                                           metrics_all_classes=metrics_all_classes,
                                           ema_model=ema.model() if ema is not None else None,
                                           ema_val_loss=ema_val_loss if ema is not None else None,
                                           ema_metrics_per_classes=ema_metrics_per_classes if ema is not None else None,
                                           ema_metrics_all_classes=ema_metrics_all_classes if ema is not None else None,
                                           )
        log_metrics.update({"score/ori": composite_score['ori']})
        if ema is not None:
            log_metrics.update({"score/ema": composite_score['ema']})
        swanlab.log(sanitize_metrics(log_metrics), step=epoch)

        metrics_history.append(sanitize_metrics(log_metrics))
        if first_train_loss is None:
            first_train_loss = train_loss
        last_train_loss = train_loss

        # Optuna：上报当前 epoch 的优化指标，由 pruner 决定是否提前终止该 trial
        if trial is not None:
            epoch_value = log_metrics.get(epoch_metric_key)
            if epoch_value is not None:
                trial.report(float(epoch_value), step=epoch)
                if trial.should_prune():
                    log_write(f"Epoch {epoch} - trial {trial.number} pruned by Optuna.\n")
                    raise optuna.TrialPruned()

    # end of training, log best epoch and best score for both original model and ema model (if exists)
    log_write("\n=================================== ⭐️ Best Model Validation Metrics ========================================= \n")

    best_log = {}
    for model_type in model_saver.saved_model_types:
        best_epoch, best_score, best_metrics, best_metrics_per_classes = model_saver.get_best_info(model_type=model_type)
        best_log.update({f"best_{model_type}/epoch": best_epoch,
                         f"best_{model_type}/score": best_score})

        per_class_metrics = {}
        table_cls = PrettyTable()
        table_cls.field_names = ["metric"] + data_classes
        table_cls._string_width = wcwidth.wcswidth
        for key, value in best_metrics_per_classes.items():
            data_rows = [key]
            for class_idx in range(len(value)):
                new_key = f"best_{model_type}/cls_{key}/c{class_idx}"
                per_class_metrics.update({new_key: value[class_idx]})
                data_rows.append(f"{value[class_idx]:.4f}")
            table_cls.add_row(data_rows)
        log_write(f"Best {model_type} Model in Epoch {best_epoch} Validation Metrics Per Class:\n{table_cls}\n")

        best_log.update(per_class_metrics)

        metrics = {}
        table = PrettyTable()
        table.field_names = ["reduction"] + metrics_row
        table._string_width = wcwidth.wcswidth
        for key, value in best_metrics.items():
            data_row = []
            for k in metrics_row:
                v = value[k]
                data_row.append(f"{v:.4f}")
                new_key = f"best_{model_type}/metric({key})/{k}"
                metrics.update({new_key: v})
            table.add_row([key] + data_row)

        log_write(f"Best {model_type} Model in Epoch {best_epoch} Validation Metrics:\n{table}\n")
        best_log.update(metrics)

    swanlab.log(sanitize_metrics(best_log), step=epochs)

    # 落盘最佳指标与训练元信息，供 analyze_sweep.py 等离线分析脚本使用
    final_metrics = {
        'best': sanitize_metrics(best_log),
        'meta': {
            'epochs': epochs,
            'epochs_trained': len(metrics_history),
            'first_train_loss': first_train_loss,
            'last_train_loss': last_train_loss,
            'opt_metric': getattr(config, 'opt_metric', None),
            'exp_dir': str(config.exp_dir),
        },
    }
    with open(cfg_dir / 'final_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2, default=str)
    with open(cfg_dir / 'metrics_history.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_history, f, ensure_ascii=False, indent=2, default=str)

    #  Evaluate best model on validation set
    for model_type in model_saver.saved_model_types:
        model_saver.load_best_ckpt(model=model, model_type=model_type)
        evaluate_model(model=model,
                       val_loader=val_loader,
                       mean=val_dataset.get_mean(),
                       std=val_dataset.get_std(),
                       color_map=dataset_cfg['color_map'],
                       save_path=plot_dir,
                       max_batch=8,
                       device=device,
                       model_type=model_type,
                       use_roi=config.use_roi)

    # ============ 实验结束：显式释放资源，避免 "Too many open files" ============
    # 删除 DataLoader 引用，强制工作进程退出并释放文件描述符
    del train_loader, val_loader
    # 删除模型引用
    del model
    # 强制垃圾回收，及时释放所有资源
    gc.collect()
    # 清空 CUDA 缓存（如果使用 GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回训练结束时的指标，供 Optuna objective 读取优化目标
    return best_log


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model with Optuna hyperparameter tuning')
    parser.add_argument('--model_cfg', type=str, default=None, help='model config file')
    parser.add_argument('--arch', type=str, default='unet', help='architecture of the model')
    parser.add_argument('--encoder_name', type=str, default='timm-tf_efficientnet_lite1', help='encoder name for the model')
    parser.add_argument('--encoder_init_weights', type=str, default='imagenet', help='initialize encoder weights')
    parser.add_argument('--dataset_cfg', type=str, default='UltraSeg/config/dataset/zhuiguan/zhuiguan.yaml', help='dataset config file')
    parser.add_argument('--input_size', type=int, default=512, help='input size for training and validation')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('--att_type', type=str, default=None, help='decoder attention type for training, none or scse')
    parser.add_argument('--use_roi', action='store_true', help='use roi for training')
    parser.add_argument('-hs', '--hard_samp', action='store_true', help='use hard sampling for training')
    parser.add_argument('-av', '--augment_version', type=int, default=2, help='augment version for training')
    parser.add_argument('-rc', '--rare_classes', type=int, nargs='+',
                        default=None,  # 腕管：[1, 3, 6, 8, 11]
                        help='rare classes for training, e.g. --rare_classes 0 1 2')
    parser.add_argument('--sweep_cfg', type=str, default='UltraSeg/config/hyper/ema-sweep-zhuiguan.yaml', help='hyperparameters search space config file')
    parser.add_argument('--work-dir',
                        default=ROOT / 'res', help='the dir to save logs and models')
    parser.add_argument('--project',
                        default='zhuiguan-seg', help='the project name to save logs')
    parser.add_argument('-n', '--name', default='exp', help='save to work-dir/project/name, and SwanLab run name')
    parser.add_argument('-d', '--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--load_from_ckpt', type=str, default=None, help='load from checkpoint')
    parser.add_argument('-sc', '--sweep_count', type=int, default=10, help='number of optuna trials')
    parser.add_argument('--study_name', type=str, default=None, help='optuna study name, default: derived from model setting')
    parser.add_argument('--storage', type=str, default=None, help='optuna storage url, default: sqlite under work-dir')
    parser.add_argument('--sampler', type=str, default='tpe', choices=['tpe', 'random'], help='optuna sampler')
    parser.add_argument('--pruner', type=str, default='median', choices=['median', 'percentile', 'none'], help='optuna pruner')
    parser.add_argument('--prune_startup_trials', type=int, default=10, help='number of trials to wait before pruning')
    parser.add_argument('--prune_warmup_epochs', type=int, default=20, help='number of epochs to wait before pruning')
    parser.add_argument('--opt_metric', type=str, default='best_ema/metric(macro)/dice',
                        help='metric to maximize, e.g. score/ema, best_ema/metric(macro)/dice')
    parser.add_argument('--log_mode', type=str, default='online', choices=['online', 'offline', 'local', 'disabled'],
                        help='SwanLab logging mode, default: online')

    args = parser.parse_args()

    return args


def build_sweep_name(opts):
    att = opts.att_type if opts.att_type is not None else 'no-att'
    roi = 'roi' if opts.use_roi else 'no-roi'
    hard_samp = 'hard-samp' if opts.hard_samp else 'no-hard-samp'
    return f"{opts.encoder_name}-{att}-{roi}-{hard_samp}-{opts.input_size}"


def build_config(opts, hyper_params: dict, trial_number: int = None):
    """把 CLI 参数与 trial 采样得到的超参数组合成本次实验的配置对象。"""
    config = dict_to_obj(dict(hyper_params))

    # epochs / batch_size 统一成整数，保证落盘的 hyper.yaml 与 SwanLab config 也是整数
    for key in ('epochs', 'batch_size'):
        value = getattr(config, key, None)
        if value is not None:
            setattr(config, key, int(value))

    name = f"{build_sweep_name(opts)}-{opts.name}"
    if trial_number is not None:
        name = f"{name}-trial{trial_number}"
    exp_dir = increment_path(work_dir=opts.work_dir, project=opts.project, name=name)

    config.exp_dir = exp_dir
    config.model_cfg = opts.model_cfg
    config.arch = opts.arch
    config.encoder_name = opts.encoder_name
    config.encoder_init_weights = opts.encoder_init_weights
    config.dataset_cfg = opts.dataset_cfg
    config.device = opts.device
    config.input_size = opts.input_size
    config.decoder_attention_type = opts.att_type
    config.use_roi = opts.use_roi
    config.hard_samp = opts.hard_samp
    config.rare_classes = opts.rare_classes
    config.load_from_ckpt = opts.load_from_ckpt
    config.augment_version = opts.augment_version
    config.num_workers = opts.num_workers
    config.opt_metric = opts.opt_metric
    config.trial_number = trial_number

    hyper_cfg = object_to_dict(config)
    hyper_cfg['exp_dir'] = str(exp_dir)
    config.hyper_cfg = hyper_cfg

    return config


def make_objective(opts, search_space: dict, study_name: str = None):
    """构造 Optuna 的 objective：一次 trial 对应一次完整训练 + 一次 SwanLab 记录。"""

    def objective(trial: optuna.Trial):
        hyper_params = suggest_hyperparams(trial, search_space)
        config = build_config(opts, hyper_params, trial_number=trial.number)

        # 把实验目录记到 trial 上，方便从 Optuna 数据库 / CSV 反查对应实验
        trial.set_user_attr('exp_dir', str(config.exp_dir))

        run_config = dict(config.hyper_cfg)
        run_config.update({'trial_number': trial.number,
                           'study_name': study_name,
                           'opt_metric': opts.opt_metric})

        swanlab.init(project=opts.project,
                     name=Path(config.exp_dir).name,
                     group=study_name,       # 同一 study 的 trial 在 SwanLab 中归为一组
                     job_type='sweep',
                     log_dir=str(config.exp_dir),
                     config=run_config,
                     mode=opts.log_mode,
                     reinit=True)            # 同一进程内连续跑多个 trial 必须 reinit
        try:
            final_metrics = train(config, trial=trial)
        except optuna.TrialPruned:
            swanlab.finish(state='aborted')
            raise
        except Exception as error:
            swanlab.finish(state='crashed', error=str(error))
            raise
        swanlab.finish()

        if opts.opt_metric not in final_metrics:
            raise KeyError(f"Metric '{opts.opt_metric}' not found in the final metrics, "
                           f"available keys: {sorted(final_metrics.keys())}")
        return float(final_metrics[opts.opt_metric])

    return objective


def get_sampler(opts):
    seed = 42
    if opts.sampler == 'tpe':
        return optuna.samplers.TPESampler(seed=seed)
    return optuna.samplers.RandomSampler(seed=seed)


def get_pruner(opts):
    if opts.pruner == 'median':
        return optuna.pruners.MedianPruner(n_startup_trials=opts.prune_startup_trials,
                                           n_warmup_steps=opts.prune_warmup_epochs,
                                           interval_steps=1)
    if opts.pruner == 'percentile':
        return optuna.pruners.PercentilePruner(percentile=25.0,
                                               n_startup_trials=opts.prune_startup_trials,
                                               n_warmup_steps=opts.prune_warmup_epochs,
                                               interval_steps=1)
    return optuna.pruners.NopPruner()


def sweep_main(opts=None):
    opts = opts if opts is not None else parse_args()

    search_space = build_search_space(yaml_load(opts.sweep_cfg))
    study_name = opts.study_name if opts.study_name is not None else build_sweep_name(opts)

    sweep_dir = Path(opts.work_dir) / opts.project / 'sweeps'
    sweep_dir.mkdir(parents=True, exist_ok=True)
    storage = opts.storage if opts.storage is not None else f"sqlite:///{sweep_dir / f'{study_name}.db'}"

    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                direction='maximize',
                                sampler=get_sampler(opts),
                                pruner=get_pruner(opts),
                                load_if_exists=True)  # 已存在的同名 study 会继续追加 trial，方便中断后恢复

    # Optuna 不允许同一 study 内同名参数更换分布类型；搜索空间调整后复用旧 study 会
    # 在采样阶段直接报错（且 trial 尚未训练）。这里提前检查并给出可操作的提示。
    mismatches = find_incompatible_params(study, search_space)
    if mismatches:
        detail = ', '.join(f"{m['param']}: {m['recorded']} -> {m['current']}" for m in mismatches)
        db_path = storage[len('sqlite:///'):] if storage.startswith('sqlite:///') else storage
        raise ValueError(
            f"搜索空间与已有 study '{study_name}' 中记录的分布不兼容（{detail}）。\n"
            f"Optuna 不允许同一个 study 内同名参数更换分布类型，请二选一：\n"
            f"  1) 删除旧的 study 数据库后重跑：rm '{db_path}'\n"
            f"  2) 换用新的 study 名：--study_name <new_name>"
        )

    log_write(f"\nOptuna study: {study_name}\n")
    log_write(f"{'storage:':<20} {storage}\n")
    log_write(f"{'search space:':<20} {list(search_space.keys())}\n")
    log_write(f"{'objective:':<20} {opts.opt_metric} (maximize)\n")
    log_write(f"{'trials:':<20} {opts.sweep_count}\n\n")

    try:
        study.optimize(make_objective(opts, search_space, study_name=study_name),
                       n_trials=opts.sweep_count)
    except KeyboardInterrupt:
        log_write("\n⚠️ Sweep interrupted by user, saving current results...\n")

    # 汇总结果：最优超参数写入 yaml（可直接作为 base_train.py 的 --hyper_cfg），全部 trial 写入 csv
    best_params = get_trial_params(study.best_trial, search_space)
    yaml_save(sweep_dir / f'{study_name}-best_params.yaml', best_params)
    trials_df = export_trials_dataframe(study, search_space)
    trials_df.to_csv(sweep_dir / f'{study_name}-trials.csv', index=False)

    log_write("\n==================== Optuna sweep finished ====================\n")
    log_write(f"Best trial: {study.best_trial.number}, {opts.opt_metric} = {study.best_value:.6f}\n")
    for key, value in best_params.items():
        log_write(f"{key + ':':<28} {value}\n")
    log_write(f"📁 Best trial exp dir: {study.best_trial.user_attrs.get('exp_dir', 'N/A')}\n")
    log_write(f"📁 Best params saved at: {sweep_dir / f'{study_name}-best_params.yaml'}\n")
    log_write(f"📁 All trials saved at:  {sweep_dir / f'{study_name}-trials.csv'}\n")


if __name__ == '__main__':
    sweep_main()
