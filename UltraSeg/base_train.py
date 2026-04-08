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
from UltraSeg.core.ema import EMA
from UltraSeg.sweep_train import train


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--model_cfg', type=str, default='UltraSeg/config/network/unet-mobilenetv2.yaml', help='model config file')
    parser.add_argument('--dataset_cfg', type=str, default='UltraSeg/config/dataset/wrist.yaml', help='dataset config file')
    parser.add_argument('--hyper_cfg', type=str, default='res/unet-mobilenetv2/exp10/cfg/hyper.yaml', help='hyperparameters config file')
    parser.add_argument('--input_size', type=int, default=512, help='input size for training and validation')
    parser.add_argument('--att_type', type=str, default=None, help='decoder attention type for training, none or scse')
    parser.add_argument('--use_roi', action='store_true', help='use roi for training')
    parser.add_argument('--use_dual', action='store_true', help='use cutmix for training')
    parser.add_argument('--work-dir',
                        default=ROOT / 'res', help='the dir to save logs and models')
    parser.add_argument('--project',
                        default='test', help='the project name to save logs')
    parser.add_argument('--name', default='exp', help='save to work-dir/project/name')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--load_from_ckpt', type=str, default=None, help='load from checkpoint')

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def dict_to_obj(dictionary):
    """
    将字典转换为可以通过点号访问的对象

    Args:
        dictionary (dict): 输入字典

    Returns:
        object: 可以通过点号访问的对象
    """
    class DictObject:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, dict_to_obj(value))
                else:
                    setattr(self, key, value)

    return DictObject(dictionary)


def main():
    opts = parse_args()
    # setup output
    exp_dir = increment_path(work_dir=opts.work_dir, project=opts.project, name=opts.name)
    # exp_folder_name = exp_dir.name
    config = dict_to_obj(yaml_load(opts.hyper_cfg))

    config.exp_dir = exp_dir
    config.model_cfg = opts.model_cfg
    config.dataset_cfg = opts.dataset_cfg
    config.device = opts.device
    config.decoder_attention_type = opts.att_type
    config.use_roi = opts.use_roi
    config.use_cutmix = opts.use_dual
    config.load_from_ckpt = opts.load_from_ckpt if opts.load_from_ckpt is not None else None
    train(config)


if __name__ == '__main__':
    main()
