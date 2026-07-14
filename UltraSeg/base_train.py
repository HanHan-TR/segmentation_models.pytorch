import argparse
import shutil
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

from UltraSeg.core.fileio import yaml_load, increment_path, dict_to_obj, object_to_dict
from UltraSeg.sweep_train import train


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--model_cfg', type=str, default=None, help='model config file')
    parser.add_argument('--dataset_cfg', type=str,
                        default='UltraSeg/config/dataset/wan_guanBG.yaml',
                        help='dataset config file')
    parser.add_argument('--hyper_cfg', type=str,
                        default='res/wrist-seg/best/timm-tf_efficientnet_lite1-no-att-no-roi-hard-samp-512-p22/cfg/hyper.yaml',
                        help='hyperparameters config file')
    parser.add_argument('--work-dir',
                        default=ROOT / 'res', help='the dir to save logs and models')
    parser.add_argument('--project',
                        default='wan_guan-seg', help='the project name to save logs')
    parser.add_argument('--name', default='exp', help='save to work-dir/project/name')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--load_from_ckpt', type=str, default=None, help='load from checkpoint')

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    opts = parse_args()
    # setup output
    exp_dir = increment_path(work_dir=opts.work_dir, project=opts.project, name=opts.name)
    # exp_folder_name = exp_dir.name
    config = dict_to_obj(yaml_load(opts.hyper_cfg))

    config.exp_dir = exp_dir
    config.device = opts.device
    if opts.dataset_cfg is not None:
        config.dataset_cfg = opts.dataset_cfg

    if opts.model_cfg is not None:
        config.model_cfg = opts.model_cfg

    config.load_from_ckpt = opts.load_from_ckpt if opts.load_from_ckpt is not None else None

    hyper_cfg = object_to_dict(config)
    config.hyper_cfg = hyper_cfg

    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="wanghan-tr-tuorenmedical",
        # Set the wandb project where this run will be logged.
        project=opts.project,
        name=exp_dir.name,
        dir=exp_dir,
    )
    train(config)


if __name__ == '__main__':
    main()
