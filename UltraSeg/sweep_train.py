import argparse
import wandb
from pathlib import Path
import os
import sys
import torch
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

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
# from UltraSeg.logger.logger import get_environment_info, log_write


def train(config):

    try:
        os.stat(config.exp_dir)
    except Exception:
        os.makedirs(config.exp_dir)

    # logfile = str(config.exp_dir / 'extractor_train.log')
    weight_dir, cfg_dir, plot_dir = Path(config.exp_dir) / 'weights', Path(config.exp_dir) / 'cfg', Path(config.exp_dir) / 'plots'
    best_model_pth, last_model_pth = weight_dir / 'best.pth', weight_dir / 'last.pth'
    plot_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    model_cfg, dataset_cfg = yaml_load(config.model_cfg), yaml_load(config.dataset_cfg)
    # save configs
    yaml_save(cfg_dir / 'model.yaml', model_cfg)
    yaml_save(cfg_dir / 'dataset.yaml', dataset_cfg)
    yaml_save(cfg_dir / 'hyper.yaml', config)

    wandb.log({"save_dir": str(config.exp_dir), })
    # 设置随机种子, 保证算法的可复现性
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    seed = init_random_seed(seed=config.seed, device=device)
    set_random_seed(seed, deterministic=True)

    # Create dataset
    train_dataset = create_dataset(dataset_cfg, split='train')
    val_dataset = create_dataset(dataset_cfg, split='val')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             drop_last=False)
    class_weights = compute_class_weights_from_loader(train_loader,
                                                      dataset_cfg['num_classes'],
                                                      method="sqrt",
                                                      eps=1e-6,
                                                      normalize=True).tolist()
    print(f"Class weights: {class_weights}")

    # Create model saver
    model_saver = ModelSaver(best_model_pth=best_model_pth,
                             last_model_pth=last_model_pth,
                             metric_reduction="weighted",)
    # Create model
    model = create_model(arch=model_cfg['arch'],
                         encoder_name=model_cfg['encoder_name'],
                         encoder_weights=model_cfg['encoder_weights'],
                         in_channels=3,
                         classes=dataset_cfg['num_classes'])

    # Load checkpoint if specified
    if config.load_from_ckpt is not None:
        model_saver.load_best_ckpt(model=model,
                                   ckpt_path=config.load_from_ckpt)

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

    for epoch in range(epochs):
        train_loss = train_one_epoch(epoch=epoch,
                                     model=model,
                                     train_loader=train_loader,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     lr_scheduler=lr_scheduler,
                                     scaler=scaler,
                                     device=device,
                                     epochs=epochs)
        lr = lr_scheduler.get_last_lr()
        wandb.log({"learning_rate/encoder_decay": lr[0],
                   "learning_rate/encoder_no_decay": lr[1],
                   "learning_rate/decoder_decay": lr[2],
                   "learning_rate/decoder_no_decay": lr[3],
                   })
        lr_scheduler.step()
        metrics_per_classes, metrics_all_classes, val_loss = validate_one_epoch(epoch=epoch,
                                                                                model=model,
                                                                                val_loader=val_loader,
                                                                                loss_fn=loss_fn,
                                                                                class_weights=class_weights,
                                                                                device=device,
                                                                                epochs=epochs)
        wandb.log({"epoch": epoch,
                   "loss/train": train_loss,
                   "loss/val": val_loss})

        metrics = {}
        for key, value in metrics_all_classes.items():  # reduction: acc: value
            for k, v in value.items():
                new_key = f"metric_{key}/{k}"
                metrics.update({new_key: v})
        wandb.log(metrics)

        per_class_metrics = {}
        for key, value in metrics_per_classes.items():
            for class_idx in range(len(value)):
                new_key = f"per_class_{key}/class_{class_idx}"
                per_class_metrics.update({new_key: value[class_idx]})
        wandb.log(per_class_metrics)

        composite_score = model_saver.save(epoch=epoch,
                                           model=model,
                                           optimizer=optimizer,
                                           lr_ctrl=lr_scheduler,
                                           val_loss=val_loss,
                                           metrics_per_classes=metrics_per_classes,
                                           metrics_all_classes=metrics_all_classes)
        wandb.log({"composite_score": composite_score})

    best_metrics, best_metrics_per_classes, best_composite_score, best_epoch = model_saver.get_best_info()
    wandb.log({"best_composite_score": best_composite_score,
               "best_epoch": best_epoch,
               "best_metrics": best_metrics})

    best_per_class_metrics = {}
    for key, value in best_metrics_per_classes.items():
        for class_idx in range(len(value)):
            new_key = f"best_per_class_{key}/class_{class_idx}"
            best_per_class_metrics.update({new_key: value[class_idx]})
    wandb.log(best_per_class_metrics)

    best_metrics = {}
    for key, value in best_metrics.items():
        for k, v in value.items():
            new_key = f"best_metric_{key}/{k}"
            best_metrics.update({new_key: v})
    wandb.log(best_metrics)

    #  Evaluate best model on validation set
    model_saver.load_best_ckpt(model=model)
    evaluate_model(model=model,
                   val_loader=val_loader,
                   mean=val_dataset.get_mean(),
                   std=val_dataset.get_std(),
                   save_path=plot_dir,
                   max_batch=8,
                   device=device)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--model_cfg', type=str, default='UltraSeg/config/network/unet-mobilenetv2.yaml', help='model config file')
    parser.add_argument('--dataset_cfg', type=str, default='UltraSeg/config/dataset/wrist.yaml', help='dataset config file')
    parser.add_argument('--sweep_cfg', type=str, default='UltraSeg/config/hyper/unet_sweep_stage2.yaml', help='hyperparameters config file')
    parser.add_argument('--work-dir',
                        default=ROOT / 'res', help='the dir to save logs and models')
    parser.add_argument('--project',
                        default='unet-mobilenetv2_stage2', help='the project name to save logs')
    parser.add_argument('--name', default='tune', help='save to work-dir/project/name')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--load_from_ckpt', type=str, default="res/unet-mobilenetv2/exp2/weights/best.pth", help='load from checkpoint')
    parser.add_argument('--sweep_count', type=int, default=50, help='sweep count for wandb agent')
    args = parser.parse_args()

    return args


def main():
    opts = parse_args()
    # setup output
    exp_dir = increment_path(work_dir=opts.work_dir, project=opts.project, name=opts.name)
    exp_folder_name = exp_dir.name

    with wandb.init(project=opts.project, name=exp_folder_name) as run:
        run.config.exp_dir = exp_dir
        run.config.model_cfg = opts.model_cfg
        run.config.dataset_cfg = opts.dataset_cfg
        run.config.device = opts.device
        run.config.load_from_ckpt = opts.load_from_ckpt
        train(run.config)


if __name__ == '__main__':
    opts = parse_args()
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize",
                   "name": "best_composite_score"},
        "parameters": {}
    }
    sweep_parameters = yaml_load(opts.sweep_cfg)
    sweep_configuration["parameters"].update(sweep_parameters)

    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           entity="wanghan-tr-tuorenmedical",
                           project=opts.project)
    wandb.agent(sweep_id, function=main, count=opts.sweep_count)
    main()
