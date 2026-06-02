import platform
import sys
import os
import torch
from datetime import datetime
from typing import Optional
from pathlib import Path, PosixPath
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from UltraSeg.core.fileio import yaml_load, object_to_dict

TQDM_BAR_FORMAT = '{l_bar}{bar:16}{r_bar}'


def log_write(message='', logfile: Optional[str] = None):
    if logfile is None:
        print(message)
    else:
        with open(logfile, 'a') as f:
            f.write(message)


def get_environment_info(logfile=None):
    """收集并返回训练环境信息字典"""
    env_info = {}

    # 1. 操作系统信息
    env_info["Platform"] = platform.system()  # Windows/Linux/macOS
    env_info["OS Version"] = platform.platform()

    # 2. Python环境
    env_info["Python Version"] = sys.version.split()[0]  # 取主版本号

    # 3. PyTorch环境
    env_info["PyTorch Version"] = torch.__version__

    # 4. GPU和CUDA信息
    if torch.cuda.is_available():
        env_info["CUDA Available"] = "Yes"
        env_info["CUDA Version"] = torch.version.cuda
        env_info["Number of GPUs"] = torch.cuda.device_count()
        # env_info["Current Device"] = torch.cuda.current_device()
        env_info["Device Name"] = torch.cuda.get_device_name()
        env_info["GPU Memory (MB)"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f}"
    else:
        env_info["CUDA Available"] = "No"

    result = "\n".join(f"{key + ':':<25} {value}" for key, value in env_info.items())
    result += '\n'

    title = "=============================== environment info ====================================== \n"
    log_write(title, logfile=logfile)
    log_write(result, logfile=logfile)


def get_experiment_info(cfg_dir: PosixPath, logfile: Optional[str] = None):
    dataset_cfg = yaml_load(cfg_dir / 'dataset.yaml')
    title = "=============================== dataset info ====================================== \n"
    log_write(title, logfile=logfile)

    for key, value in dataset_cfg.items():
        if isinstance(value, list):
            log_write(f"{key}:", logfile=logfile)
            for item in value:
                if isinstance(item, list):
                    item_str = ', '.join(str(i) for i in item)
                    log_write(f"  - {item_str}", logfile=logfile)
                else:
                    log_write(f"  - {item}", logfile=logfile)
        else:
            log_write(f"{key  + ':':<20} {value}", logfile=logfile)

    title = "=============================== model info ====================================== \n"
    model_cfg = yaml_load(cfg_dir / 'model.yaml')
    log_write(title, logfile=logfile)
    for key, value in model_cfg.items():
        log_write(f"{key  + ':':<28} {value}", logfile=logfile)

    config_dict = yaml_load(cfg_dir / 'hyper.yaml')
    title = "=============================== training hyperparameters ====================================== \n"
    log_write(title, logfile=logfile)
    config_dict.pop('model_cfg')
    config_dict.pop('dataset_cfg')
    for key, value in config_dict.items():
        if key != 'exp_dir':
            log_write(f"{key  + ':':<28} {value}", logfile=logfile)

    log_write(f"\nThis running is saved at: {config_dict['exp_dir']}\n", logfile=logfile)

    exp_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_write(f"Training Start Time:   {exp_time}\n", logfile=logfile)
