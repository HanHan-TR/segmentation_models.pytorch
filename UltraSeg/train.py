import argparse
from pathlib import Path
import os
import sys

import torch

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--model_cfg', type=str, default='UltraSeg/config/network/model.yaml', help='model config file')
    parser.add_argument('--dataset_cfg', type=str, default='UltraSeg/config/dataset/coco.yaml', help='dataset config file')
    parser.add_argument('--hyper_cfg', type=str, default='UltraSeg/config/hyper/hyper.yaml', help='hyperparameters config file')
    parser.add_argument('--work-dir',
                        default=ROOT / 'res/extractor', help='the dir to save logs and models')
    parser.add_argument('--name', default='tune', help='save to work-dir/project/name')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(opts):
    # setup output
    exp_dir = increment_path(work_dir=opts.work_dir, project=opts.backbone_name, name=opts.name)

    try:
        os.stat(exp_dir)
    except Exception:
        os.makedirs(exp_dir)

    logfile = str(exp_dir / 'extractor_train.log')
    weight_dir, cfg_dir = exp_dir / 'weights', exp_dir / 'cfg'
    weight_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    model_cfg, dataset_cfg, hyper_cfg = yaml_load(opts.model_cfg), yaml_load(opts.dataset_cfg), yaml_load(opts.hyper_cfg)
    # save configs
    yaml_save(model_cfg, cfg_dir / 'model.yaml')
    yaml_save(dataset_cfg, cfg_dir / 'dataset.yaml')
    yaml_save(hyper_cfg, cfg_dir / 'hyper.yaml')

    # 设置随机种子, 保证算法的可复现性
    device = torch.device(f"cuda:{opts.device}" if torch.cuda.is_available() else "cpu")
    seed = init_random_seed(seed=hyper_cfg['seed'], device=device)
    set_random_seed(seed, deterministic=hyper_cfg['deterministic'])

    # Create dataset
    train_dataset = create_dataset(dataset_cfg, split='train')
    val_dataset = create_dataset(dataset_cfg, split='val')
    # Create model
    model = create_model(arch=model_cfg['arch'],
                         encoder_name=model_cfg['encoder_name'],
                         encoder_weights=model_cfg['encoder_weights'],
                         in_channels=3,
                         classes=dataset_cfg['num_classes'],)


if __name__ == '__main__':
    opts = parse_args()
    main(opts)
