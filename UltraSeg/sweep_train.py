import argparse
import wandb
from pathlib import Path
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import sys
import torch
from prettytable import PrettyTable
import wcwidth
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))
from torch.utils.data import DataLoader, WeightedRandomSampler
from segmentation_models_pytorch import create_model
from UltraSeg.core.fileio import yaml_load, yaml_save, increment_path
from UltraSeg.core.initialize import init_random_seed, set_random_seed
from UltraSeg.core.dataset import create_dataset
from UltraSeg.tools.val import validate_one_epoch, ModelSaver
from UltraSeg.tools.evaluate import compute_class_weights_from_loader
from UltraSeg.tools.train_utils import train_one_epoch
from UltraSeg.core.losses import Loss
from UltraSeg.core.optimizer import get_optimizer
from UltraSeg.core.lr_scheduler import get_lr_scheduler
from UltraSeg.tools.evaluate import evaluate_model
from UltraSeg.core.ema import EMA

from UltraSeg.logger.logger import get_environment_info, get_experiment_info, log_write


def train(config):

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

    # save configs
    yaml_save(cfg_dir / 'model.yaml', model_cfg)
    yaml_save(cfg_dir / 'dataset.yaml', dataset_cfg)
    yaml_save(cfg_dir / 'hyper.yaml', config.hyper_cfg) if hasattr(config, 'hyper_cfg') else yaml_save(cfg_dir / 'hyper.yaml', config)

    # 设置随机种子, 保证算法的可复现性
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    seed = init_random_seed(seed=config.seed, device=device)
    set_random_seed(seed, deterministic=True)

    # Create dataset
    train_dataset = create_dataset(dataset_cfg,
                                   input_size=config.input_size,
                                   split='train',
                                   use_roi=config.use_roi,
                                   hard_samp=config.hard_samp,
                                   rare_classes=config.rare_classes,
                                   augment_version=config.augment_version)
    val_dataset = create_dataset(dataset_cfg,
                                 input_size=config.input_size,
                                 split='val',
                                 use_roi=config.use_roi,
                                 hard_samp=False,
                                 augment_version=config.augment_version)
    if config.hard_samp:
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(train_dataset.sample_weights),
                                        num_samples=len(train_dataset.sample_weights) * 2,  # 进行过采样，迭代器的长度是原来的两倍
                                        replacement=True)

        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  sampler=sampler,
                                  num_workers=8,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True)

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

    for epoch in range(epochs):
        wandb_summary = {}
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
        wandb.log({"learning_rate/encoder_decay": lr[0],
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
                                                                                model_type='ori')

        wandb_summary.update({"epoch": epoch,
                              "loss/ori_train": train_loss,
                              "loss/ori_val": val_loss,
                              "learning_rate/encoder_decay": lr[0],
                              "learning_rate/encoder_no_decay": lr[1],
                              "learning_rate/decoder_decay": lr[2],
                              "learning_rate/decoder_no_decay": lr[3]})
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
                wandb_summary.update({new_key: v})
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
                wandb_summary.update({new_key: value[class_idx]})
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
                                                                                                model_type='ema')
            wandb_summary.update({"loss/ema_val": ema_val_loss})

            ema_table = PrettyTable()
            ema_table.field_names = ["reduction"] + metrics_row
            ema_table._string_width = wcwidth.wcswidth
            for key, value in ema_metrics_all_classes.items():  # reduction: acc: value
                data_row = []
                for k in metrics_row:
                    v = value[k]
                    data_row.append(f"{v:.4f}")
                    new_key = f"ema_metric({key})/{k}"
                    wandb_summary.update({new_key: v})
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
                    wandb_summary.update({new_key: value[class_idx]})
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
        wandb_summary.update({"score/ori": composite_score['ori'],
                              "score/ema": composite_score['ema'] if ema is not None else None})
        wandb.log(wandb_summary)

    # end of training, log best epoch and best score for both original model and ema model (if exists)
    log_write("\n=================================== ⭐️ Best Model Validation Metrics ========================================= \n")

    for model_type in model_saver.saved_model_types:
        best_epoch, best_score, best_metrics, best_metrics_per_classes = model_saver.get_best_info(model_type=model_type)
        wandb.log({f"best_{model_type}/epoch": best_epoch,
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

        wandb.log(per_class_metrics)

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
        wandb.log(metrics)

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--model_cfg', type=str, default=None, help='model config file')
    parser.add_argument('--arch', type=str, default='unet', help='architecture of the model')
    parser.add_argument('--encoder_name', type=str, default='timm-tf_efficientnet_lite1', help='encoder name for the model')
    parser.add_argument('--encoder_init_weights', type=str, default='imagenet', help='initialize encoder weights')
    parser.add_argument('--dataset_cfg', type=str, default='UltraSeg/config/dataset/wan_guanBG.yaml', help='dataset config file')
    parser.add_argument('--input_size', type=int, default=512, help='input size for training and validation')
    parser.add_argument('--att_type', type=str, default=None, help='decoder attention type for training, none or scse')
    parser.add_argument('--use_roi', action='store_true', help='use roi for training')
    parser.add_argument('--hard_samp', action='store_true', help='use hard sampling for training')
    parser.add_argument('--augment_version', type=int, default=2, help='augment version for training')
    parser.add_argument('--rare_classes', type=int, nargs='+', 
                        default=None,  # 腕管：[1, 3, 6, 8, 11]
                        help='rare classes for training, e.g. --rare_classes 0 1 2')
    parser.add_argument('--sweep_cfg', type=str, default='UltraSeg/config/hyper/ema-sweep.yaml', help='hyperparameters config file')
    parser.add_argument('--work-dir',
                        default=ROOT / 'res', help='the dir to save logs and models')
    parser.add_argument('--project',
                        default='wan_guan-seg', help='the project name to save logs')
    parser.add_argument('--name', default='exp', help='save to work-dir/project/name, and wandb run name')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--load_from_ckpt', type=str, default=None, help='load from checkpoint')
    parser.add_argument('--sweep_count', type=int, default=60, help='sweep count for wandb agent')

    args = parser.parse_args()

    return args


def sweep_main():
    opts = parse_args()
    if opts.att_type is None:
        att = 'no-att'
    else:
        att = opts.att_type

    if opts.use_roi is False:
        roi = 'no-roi'
    else:
        roi = 'roi'

    if opts.hard_samp:
        hard_samp = 'hard-samp'
    else:
        hard_samp = 'no-hard-samp'

    opts.name = f"{opts.encoder_name}-{att}-{roi}-{hard_samp}-{opts.input_size}-{opts.name}"
    # setup output
    exp_dir = increment_path(work_dir=opts.work_dir, project=opts.project, name=opts.name)
    exp_folder_name = exp_dir.name

    with wandb.init(project=opts.project, name=exp_folder_name, dir=exp_dir) as run:
        run.config.exp_dir = exp_dir
        run.config.model_cfg = opts.model_cfg if opts.model_cfg is not None else None
        run.config.arch = opts.arch if opts.arch is not None else None
        run.config.encoder_name = opts.encoder_name if opts.encoder_name is not None else None
        run.config.encoder_init_weights = opts.encoder_init_weights if opts.encoder_init_weights is not None else None
        run.config.dataset_cfg = opts.dataset_cfg
        run.config.device = opts.device
        run.config.input_size = opts.input_size
        run.config.decoder_attention_type = opts.att_type
        run.config.use_roi = opts.use_roi
        run.config.hard_samp = opts.hard_samp
        run.config.rare_classes = opts.rare_classes
        run.config.load_from_ckpt = opts.load_from_ckpt if opts.load_from_ckpt is not None else None
        run.config.augment_version = opts.augment_version
        train(run.config)


if __name__ == '__main__':
    opts = parse_args()
    if opts.att_type is None:
        att = 'no-att'
    else:
        att = opts.att_type

    if opts.use_roi is False:
        roi = 'no-roi'
    else:
        roi = 'roi'

    if opts.hard_samp:
        hard_samp = 'hard-samp'
    else:
        hard_samp = 'no-hard-samp'

    sweep_name = f"{opts.encoder_name}-{att}-{roi}-{hard_samp}-{opts.input_size}"

    sweep_configuration = {
        "name": sweep_name,
        "method": "random",
        "metric": {"goal": "maximize",
                   "name": "score/ema"},
        "parameters": {}
    }
    sweep_parameters = yaml_load(opts.sweep_cfg)
    sweep_configuration["parameters"].update(sweep_parameters)

    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           entity="wanghan-tr-tuorenmedical",
                           project=opts.project)
    wandb.agent(sweep_id, function=sweep_main, count=opts.sweep_count)
    sweep_main()
